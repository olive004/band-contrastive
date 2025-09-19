def dummy_generate(checkpoint_path: str, target_y: float, n: int = 8):
    # Generates n dummy 'designs' with random scores near target for sanity checks
    import random
    random.seed(0)
    samples = [max(1e-3, min(0.999, target_y + random.uniform(-0.05, 0.05))) for _ in range(n)]
    return {"checkpoint": checkpoint_path, "target_y": target_y, "samples": samples}
