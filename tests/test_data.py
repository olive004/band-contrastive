from bandcon.data.datasets import DummyVectorDataset
def test_dummy_dataset():
    ds = DummyVectorDataset({"n_samples_max": 10, "vector_dim": 28, "noise":0.01, "n_nodes":8, "name":"vectors_dummy"})
    x, y = ds[0]
    assert x.numel() == 28 and 0 < float(y) < 1
