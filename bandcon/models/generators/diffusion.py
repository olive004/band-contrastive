# digress_minimal_conditional.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------


def sinusoidal_embedding(t: torch.LongTensor, dim: int):
    """[B] -> [B, dim]"""
    device, half = t.device, dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half,
                      device=device).float() / max(1, half))
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def one_hot(x: torch.LongTensor, num_classes: int):
    return F.one_hot(x, num_classes=num_classes).float()


def sample_categorical(probs: torch.Tensor):
    """
    Gumbel-Max sampling: probs [..., K] -> Long with same leading dims (class ids)
    """
    u = torch.rand_like(probs).clamp_(1e-6, 1 - 1e-6)
    g = -torch.log(-torch.log(u))
    return torch.argmax(torch.log(probs.clamp_min(1e-20)) + g, dim=-1)

# -------------------------
# Discrete diffusion schedule
# -------------------------


def make_beta_schedule(T: int, schedule="cosine", beta_start=1e-3, beta_end=5e-2):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # https://openreview.net/forum?id=-NEXDKk8gZ
        t = torch.linspace(0, T, T + 1)
        f = torch.cos(((t / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = (f / f[0]).clamp(min=1e-8)
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(1e-8, 0.999)
        # return betas.clamp(1e-5, 0.2)
    else:
        raise ValueError("Unknown schedule")


class DiscreteSchedule:
    """
    Class-uniform discrete corruption:
      q(x_t | x_{t-1}) = (1 - beta_t) * I + beta_t * (1/K) * 1
      => q(x_t | x0) = a_bar_t * 1hot(x0) + (1 - a_bar_t) * (1/K)
    """

    def __init__(self, T: int, schedule="cosine"):
        self.T = T
        self.betas = make_beta_schedule(T, schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        return self

# -------------------------
# Graph denoiser (Transformer with condition + edge bias)
# -------------------------


class AttentionWithBias(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d = d_model
        self.h = heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_bias):
        """
        x: [B, N, d_model]
        attn_bias: [B, H, N, N] additive to attention logits
        """
        B, N, d, H = x.size(0), x.size(1), self.d, self.h
        q = self.q(x).view(B, N, H, d // H).transpose(1, 2)
        k = self.k(x).view(B, N, H, d // H).transpose(1, 2)
        v = self.v(x).view(B, N, H, d // H).transpose(1, 2)
        scale = (d // H) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, d)
        return self.proj(out)


class GraphDenoiser(nn.Module):
    """
    Predicts x0 for nodes/edges from (x_t_nodes, x_t_edges, t, c).
    Conditioning: graph-level vector c projected and added to node states.
    """

    def __init__(self, K_node, K_edge, c_dim,
                 d_model=256, n_layers=6, n_heads=8, t_dim=128):
        super().__init__()
        self.Kn, self.Ke = K_node, K_edge

        self.node_embed = nn.Embedding(K_node, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(c_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # Edge-type bias per head
        self.edge_bias = nn.Embedding(K_edge, n_heads)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=AttentionWithBias(d_model, n_heads),
                norm1=nn.LayerNorm(d_model),
                ff=nn.Sequential(
                    nn.Linear(
                        d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model),
                ),
                norm2=nn.LayerNorm(d_model),
            )) for _ in range(n_layers)
        ])

        self.node_out = nn.Linear(d_model, K_node)
        self.edge_pair = nn.Sequential(
            nn.Linear(
                2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, K_edge),
        )

        self.t_dim = t_dim
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, x_nodes_t, x_edges_t, t, c_emb_vec):
        """
        x_nodes_t: [B, N] Long; x_edges_t: [B, N, N] Long; t: [B] Long
        c_emb_vec: [B, d_model] already projected condition (or null)
        """
        B, N = x_nodes_t.size()
        # [B,N,d]
        h = self.node_embed(x_nodes_t)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.t_dim))     # [B,d]
        # inject condition
        h = h + t_emb.unsqueeze(1) + c_emb_vec.unsqueeze(1)

        # attention bias from edge types (per-head)
        # [B,N,N,H]
        eb = self.edge_bias(x_edges_t)
        attn_bias = eb.permute(
            0, 3, 1, 2).contiguous()                 # [B,H,N,N]

        for lyr in self.layers:
            h2 = lyr.attn(lyr.norm1(h), attn_bias)
            h = h + h2
            h = h + lyr.ff(lyr.norm2(h))

        # [B,N,Kn]
        node_logits = self.node_out(h)

        hi = h.unsqueeze(2).expand(B, N, N, self.d_model)
        hj = h.unsqueeze(1).expand(B, N, N, self.d_model)
        # [B,N,N,2d]
        pair = torch.cat([hi, hj], dim=-1)
        # [B,N,N,Ke]
        edge_logits = self.edge_pair(pair)

        # enforce no self-loops + symmetry
        edge_logits = edge_logits.clone()  # Ensure no in-place operation on a view
        edge_logits[:, torch.arange(N), torch.arange(N)] = -1e9
        edge_logits[:, torch.arange(N), torch.arange(N), 0] = 1e9
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        return node_logits, edge_logits

# -------------------------
# Conditional DiGress (graph-level conditioning)
# -------------------------


class DiGressConditional(nn.Module):
    """
    Minimal conditional DiGress:
      - Condition c ∈ R^{c_dim} via projection and addition to node states.
      - Classifier-free guidance: p_uncond during training; guidance_w at sampling.
      - API:
         * loss(x, c) where x=(x_nodes0, x_edges0)
         * sample(num_graphs, N, c, device, guidance_w=0.0)
    """

    def __init__(self, K_node, K_edge, c_dim, cfg):
        super().__init__()
        self.Kn, self.Ke = K_node, K_edge
        self.T = cfg.T
        self.edge_loss_weight = cfg.edge_loss_weight
        self.p_uncond = cfg.p_uncond

        self.sched_node = DiscreteSchedule(cfg.T)
        self.sched_edge = DiscreteSchedule(cfg.T)

        self.denoiser = GraphDenoiser(K_node, K_edge, c_dim,
                                      d_model=cfg.d_model, n_layers=cfg.n_layers,
                                      n_heads=cfg.n_heads, t_dim=cfg.t_dim)
        # For unconditional pass
        self.null_cond = nn.Parameter(torch.zeros(1, cfg.d_model))
        self.cond_proj = self.denoiser.cond_mlp  # reuse

    def to(self, device):
        super().to(device)
        self.sched_node.to(device)
        self.sched_edge.to(device)
        return self

    # ---------- q(x_t | x0) ----------
    @staticmethod
    def _q_xt_given_x0_probs(x0_oh, a_bar_t, K):
        B = a_bar_t.size(0)
        a = a_bar_t.view(B, *([1] * (x0_oh.ndim - 1)))
        return a * x0_oh + (1 - a) * (1.0 / K)

    def q_sample_nodes(self, x0_nodes, t):
        a_bar = self.sched_node.alphas_bar[t]                          # [B]
        probs = self._q_xt_given_x0_probs(
            one_hot(x0_nodes, self.Kn), a_bar, self.Kn)
        return sample_categorical(probs)

    def q_sample_edges(self, x0_edges, t):
        device = x0_edges.device
        B, N = x0_edges.size(0), x0_edges.size(1)
        a_bar = self.sched_edge.alphas_bar[t]                           # [B]
        probs = self._q_xt_given_x0_probs(
            one_hot(x0_edges, self.Ke), a_bar, self.Ke)

        iu = torch.triu_indices(N, N, offset=1, device=device)
        # [B,M,Ke]
        probs_u = probs[:, iu[0], iu[1], :]
        samp_u = sample_categorical(probs_u)                            # [B,M]
        xt = torch.zeros(B, N, N, dtype=torch.long, device=device)
        xt[:, iu[0], iu[1]] = samp_u
        xt[:, iu[1], iu[0]] = samp_u
        return xt

    # ---------- posterior q(x_{t-1} | x_t, x0) ----------
    @staticmethod
    def _q_posterior_logits(xt, x0_probs, beta_t, a_bar_tm1, K):
        B = xt.size(0)
        q_tm1 = a_bar_tm1.view(B, *([1] * (x0_probs.ndim - 1))) * x0_probs + \
            (1 - a_bar_tm1.view(B, *([1] * (x0_probs.ndim - 1)))) * (1.0 / K)
        p_same = (1 - beta_t) + beta_t / K
        p_diff = beta_t / K
        Kdim = x0_probs.size(-1)
        onehot_xt = F.one_hot(xt, Kdim).float()
        like = p_diff.view(B, *([1] * (x0_probs.ndim - 1))) + \
            onehot_xt * (p_same - p_diff).view(B, *([1] * (x0_probs.ndim - 1)))
        return (q_tm1.clamp_min(1e-20).log() + like.clamp_min(1e-20).log())

    # ---------- helper: denoiser with CFG mixing ----------
    def _predict_logits(self, x_nodes_t, x_edges_t, t, c, guidance_w: float):
        """
        Returns (node_logits, edge_logits) with classifier-free guidance mixing.
        c: [B, c_dim]
        """
        B = x_nodes_t.size(0)
        # [B,d_model]
        c_proj = self.cond_proj(c)
        if guidance_w == 0.0:
            return self.denoiser(x_nodes_t, x_edges_t, t, c_proj)

        # conditional + unconditional (null)
        node_logits_c, edge_logits_c = self.denoiser(
            x_nodes_t, x_edges_t, t, c_proj)
        null = self.null_cond.expand(B, -1)
        node_logits_u, edge_logits_u = self.denoiser(
            x_nodes_t, x_edges_t, t, null)
        # CFG: (1+w)*cond - w*uncond   (logit-space)
        node_logits = (1 + guidance_w) * node_logits_c - \
            guidance_w * node_logits_u
        edge_logits = (1 + guidance_w) * edge_logits_c - \
            guidance_w * edge_logits_u
        return node_logits, edge_logits

    # ---------- loss ----------
    def loss(self, x, c):
        """
        x: tuple (x_nodes0 [B,N] Long, x_edges0 [B,N,N] Long) zero-diag, symmetric
        c: [B, c_dim] float
        """
        x_nodes0, x_edges0 = x
        device = x_nodes0.device
        B, N = x_nodes0.size(0), x_nodes0.size(1)
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # corruption
        x_nodes_t = self.q_sample_nodes(x_nodes0, t)
        x_edges_t = self.q_sample_edges(x_edges0, t)

        # classifier-free guidance dropout of condition during training
        if self.training and self.p_uncond > 0.0:
            keep = (torch.rand(B, device=device) >=
                    self.p_uncond).float().unsqueeze(1)
            c_eff = keep * c  # zeros when dropped
        else:
            c_eff = c

        # project condition (zeros => learned null via parameter when we call denoiser)
        # [B,d_model]
        c_proj = self.cond_proj(c_eff)
        # For dropped ones, swap to null explicitly so it learns unconditional:
        dropped_mask = (c_eff.abs().sum(dim=1, keepdim=True) == 0)
        c_proj = torch.where(
            dropped_mask, self.null_cond.expand_as(c_proj), c_proj)

        node_logits, edge_logits = self.denoiser(
            x_nodes_t, x_edges_t, t, c_proj)

        # node CE
        node_loss = F.cross_entropy(node_logits.reshape(-1, self.Kn),
                                    x_nodes0.reshape(-1))

        # edge CE off-diagonal
        eye = torch.eye(N, device=device, dtype=torch.bool)
        mask_off = (~eye).view(1, N * N).expand(B, -1)
        edge_loss = F.cross_entropy(
            edge_logits.view(B, N * N, self.Ke)[mask_off],
            x_edges0.view(B, N * N)[mask_off]
        )

        return node_loss + self.edge_loss_weight * edge_loss

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, num_graphs: int, N: int, c: torch.Tensor, device, guidance_w: float = 0.0):
        """
        Sample graphs conditioned on c.
        c: [B, c_dim] (if B=1 it will be broadcast to num_graphs)
        Returns: (x_nodes, x_edges) with shapes [B,N], [B,N,N]
        """
        if c.size(0) != num_graphs:
            if c.size(0) == 1:
                c = c.expand(num_graphs, -1)
            else:
                c = c[:num_graphs]

        B = num_graphs
        Kn, Ke = self.Kn, self.Ke

        # init x_T ~ uniform
        x_nodes_t = torch.randint(0, Kn, (B, N), device=device)
        iu = torch.triu_indices(N, N, offset=1, device=device)
        x_edges_t = torch.zeros(B, N, N, dtype=torch.long, device=device)
        x_edges_t[:, iu[0], iu[1]] = torch.randint(
            0, Ke, (B, iu.size(1)), device=device)
        x_edges_t[:, iu[1], iu[0]] = x_edges_t[:, iu[0], iu[1]]

        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            node_logits, edge_logits = self._predict_logits(
                x_nodes_t, x_edges_t, t, c, guidance_w=guidance_w
            )
            node_p0 = node_logits.softmax(-1)
            edge_p0 = edge_logits.softmax(-1)

            beta_n = self.sched_node.betas[t]
            a_bar_tm1_n = torch.where(
                t > 0,
                self.sched_node.alphas_bar[torch.clamp(t - 1, min=0)],
                torch.ones_like(beta_n)
            )
            node_post_logits = self._q_posterior_logits(
                x_nodes_t, node_p0, beta_n, a_bar_tm1_n, Kn
            )
            x_nodes_t = sample_categorical(node_post_logits.softmax(-1))

            beta_e = self.sched_edge.betas[t]
            a_bar_tm1_e = torch.where(
                t > 0,
                self.sched_edge.alphas_bar[torch.clamp(t - 1, min=0)],
                torch.ones_like(beta_e)
            )
            edge_post_logits = self._q_posterior_logits(
                x_edges_t, edge_p0, beta_e, a_bar_tm1_e, Ke
            )
            # sample only upper triangle for symmetry
            probs_u = edge_post_logits[:, iu[0], iu[1], :].softmax(-1)
            samp_u = sample_categorical(probs_u)
            x_edges_t = torch.zeros(B, N, N, dtype=torch.long, device=device)
            x_edges_t[:, iu[0], iu[1]] = samp_u
            x_edges_t[:, iu[1], iu[0]] = samp_u

        return x_nodes_t, x_edges_t

# -------------------------
# Tiny example
# -------------------------


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N = 8, 20
    Kn, Ke = 8, 4      # 0 = no-edge
    c_dim = 16
    T = 200

    model = DiGressConditional(Kn, Ke, c_dim, T=T, d_model=192, n_layers=4, n_heads=4,
                               edge_loss_weight=1.0, p_uncond=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # Dummy clean graphs + conditions
    x_nodes0 = torch.randint(0, Kn, (B, N), device=device)
    iu = torch.triu_indices(N, N, offset=1, device=device)
    x_edges0 = torch.zeros(B, N, N, dtype=torch.long, device=device)
    x_edges0[:, iu[0], iu[1]] = torch.randint(
        0, Ke, (B, iu.size(1)), device=device)
    x_edges0[:, iu[1], iu[0]] = x_edges0[:, iu[0], iu[1]]
    c = torch.randn(B, c_dim, device=device)

    model.train()
    for step in range(200):
        loss = model.loss((x_nodes0, x_edges0), c)   # <- loss(x, c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        nodes_s, edges_s = model.sample(
            num_graphs=4, N=N, c=c[:1], device=device, guidance_w=2.0)
        print("Samples:", nodes_s.shape, edges_s.shape)

okokok

# diffusion_conditional_vectors.py

# ---------- Utilities ----------


def timestep_embedding(timesteps: torch.LongTensor, dim: int):
    """
    Sinusoidal timestep embeddings (like in transformer/NeRF).
    timesteps: (B,) long on device
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    # log space from 1 to 10000
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def make_beta_schedule(T: int, schedule: str = "cosine", beta_start=1e-4, beta_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # https://openreview.net/forum?id=-NEXDKk8gZ
        t = torch.linspace(0, T, T+1)
        f = torch.cos(((t / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = (f / f[0]).clamp(min=1e-8)
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(1e-8, 0.999)
    else:
        raise ValueError("Unknown schedule")
    return betas


class DiffusionSchedule:
    def __init__(self, T: int, schedule: str = "cosine"):
        self.T = T
        betas = make_beta_schedule(T, schedule=schedule)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alphas_bar)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(
            device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(
            device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def register(self, betas, alphas, alphas_bar):
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # posterior q(x_{t-1} | x_t, x_0)
        alphas_bar_prev = torch.cat(
            [torch.tensor([1.0]), alphas_bar[:-1]], dim=0)
        self.posterior_variance = betas * \
            (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * \
            torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_mean_coef2 = (torch.sqrt(
            alphas) * (1.0 - alphas_bar_prev)) / (1.0 - alphas_bar)

# ---------- Conditional MLP Noise Predictor ----------


class ConditionalMLP(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, hidden: int = 512, depth: int = 4, time_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.time_dim = time_dim
        # small encoders for c and t
        self.c_proj = nn.Sequential(
            nn.Linear(c_dim, hidden),
            nn.SiLU(),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
        )
        in_dim = x_dim + hidden + hidden  # x + c_proj + t_proj
        layers = []
        h = hidden
        layers.append(nn.Linear(in_dim, h))
        for _ in range(depth - 1):
            layers += [nn.SiLU(), nn.Dropout(dropout), nn.Linear(h, h)]
        layers += [nn.SiLU(), nn.Linear(h, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, c):
        """
        x_t: (B, x_dim)
        t:   (B,) long
        c:   (B, c_dim)
        """
        te = timestep_embedding(t, self.time_dim)
        ct = self.c_proj(c)
        tt = self.t_proj(te)
        inp = torch.cat([x_t, ct, tt], dim=-1)
        return self.net(inp)

# ---------- Forward process + training loss ----------


class ConditionalDDPM(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, T: int = 1000, schedule: str = "cosine",
                 hidden: int = 512, depth: int = 4, time_dim: int = 128, dropout: float = 0.0,
                 p_uncond: float = 0.1):
        super().__init__()
        self.model = ConditionalMLP(
            x_dim, c_dim, hidden, depth, time_dim, dropout)
        self.diff = DiffusionSchedule(T, schedule)
        # for classifier-free guidance (drop some c during training)
        self.p_uncond = p_uncond

        # a learned "null" embedding for unconditional context (helps guidance)
        self.null_c = nn.Parameter(torch.zeros(1, c_dim))

    def to(self, device):
        super().to(device)
        self.diff.to(device)
        return self

    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t ~ q(x_t | x0)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.diff.alphas_bar[t].unsqueeze(1).sqrt()
        sqrt_1mab = self.diff.sqrt_one_minus_alphas_bar[t].unsqueeze(1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def p_mean_variance(self, x_t, t, c, guidance_w: float = 0.0):
        """
        Predict posterior mean and log-variance using epsilon model.
        guidance_w: classifier-free guidance scale (0 = conditional only, typical 1..3)
        """
        eps_cond = self.model(x_t, t, c)

        if guidance_w != 0.0:
            # unconditional pass using learned null context
            c_uncond = self.null_c.expand_as(c)
            eps_uncond = self.model(x_t, t, c_uncond)
            eps = (1 + guidance_w) * eps_cond - guidance_w * eps_uncond
        else:
            eps = eps_cond

        alpha_t = self.diff.alphas[t].unsqueeze(1)
        alpha_bar_t = self.diff.alphas_bar[t].unsqueeze(1)
        sqrt_recip_alpha = self.diff.sqrt_recip_alphas[t].unsqueeze(1)

        # x0 estimate from predicted noise
        x0_pred = (
            x_t - eps * self.diff.sqrt_one_minus_alphas_bar[t].unsqueeze(1)) / alpha_bar_t.sqrt()
        # posterior mean
        mean = (
            self.diff.posterior_mean_coef1[t].unsqueeze(1) * x0_pred +
            self.diff.posterior_mean_coef2[t].unsqueeze(1) * x_t
        )
        log_var = self.diff.posterior_log_variance_clipped[t].unsqueeze(1)
        return mean, log_var, x0_pred, eps

    @torch.no_grad()
    def sample(self, num_samples: int, x_dim: int, c: torch.Tensor, device=None, guidance_w: float = 0.0):
        """
        Draw samples x_0 ~ p_theta(x_0 | c).
        c: (B, c_dim). If num_samples != B, c will be tiled or truncated to match.
        """
        device = device or next(self.parameters()).device
        T = self.diff.T
        if c.size(0) != num_samples:
            if c.size(0) == 1:
                c = c.expand(num_samples, -1)
            else:
                c = c[:num_samples]
        x_t = torch.randn(num_samples, x_dim, device=device)
        for i in reversed(range(T)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            mean, log_var, _, _ = self.p_mean_variance(
                x_t, t, c, guidance_w=guidance_w)
            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + (0.5 * log_var).exp() * noise
            else:
                x_t = mean
        return x_t

    def loss(self, x0, c):
        """
        Standard DDPM MSE loss between true noise and predicted noise.
        With probability p_uncond, replace c by learned null context (for guidance at sampling).
        """
        B, x_dim = x0.shape
        device = x0.device
        t = torch.randint(0, self.diff.T, (B,),
                          device=device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)

        # classifier-free guidance dropout of c
        if self.p_uncond > 0.0 and self.training:
            drop_mask = (torch.rand(B, device=device) <
                         self.p_uncond).float().unsqueeze(1)
            c_eff = torch.where(drop_mask.bool(), self.null_c.expand_as(c), c)
        else:
            c_eff = c

        noise_pred = self.model(x_t, t, c_eff)
        return F.mse_loss(noise_pred, noise)

# ---------- Minimal training / usage skeleton ----------

# if __name__ == "__main__":
#     # Example dummy usage with random vectors
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     x_dim = 32     # dimension of data vector x
#     c_dim = 10     # dimension of condition vector c
#     T = 1000

#     model = ConditionalDDPM(x_dim=x_dim, c_dim=c_dim, T=T, schedule="cosine",
#                             hidden=512, depth=4, time_dim=128, p_uncond=0.1).to(device)

#     # Dummy dataset
#     N = 4096
#     x_data = torch.randn(N, x_dim)
#     # Create a synthetic conditional signal correlated with x (just for demo)
#     W = torch.randn(x_dim, c_dim)
#     c_data = (x_data @ W).tanh() + 0.05 * torch.randn(N, c_dim)

#     ds = torch.utils.data.TensorDataset(x_data, c_data)
#     loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

#     opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

#     model.train()
#     for step, (xb, cb) in enumerate(loader):
#         xb = xb.to(device)
#         cb = cb.to(device)
#         loss = model.loss(xb, cb)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         if step % 50 == 0:
#             print(f"step {step:04d} | loss {loss.item():.4f}")
#         if step == 500:  # short demo
#             break

#     model.eval()
#     with torch.no_grad():
#         # Condition for sampling (use mean c here as an example)
#         c_seed = c_data[:1].to(device)
#         samples = model.sample(num_samples=8, x_dim=x_dim, c=c_seed, device=device, guidance_w=2.0)
#         print("Sampled batch shape:", samples.shape)
