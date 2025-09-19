from bandcon.eval.metrics import simple_eval
def test_simple_eval():
    res = simple_eval([0.3,0.5], [0.10, 0.05])
    assert 0.0 <= list(res.values())[0]["@±10%"] <= 1.0
