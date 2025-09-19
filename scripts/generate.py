import argparse, json, os
from bandcon.eval.reports import dummy_generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/dummy.ckpt")
    ap.add_argument("--target_y", type=float, default=0.75)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)
    out = dummy_generate(args.checkpoint, args.target_y, args.n)
    print(json.dumps(out, indent=2))
