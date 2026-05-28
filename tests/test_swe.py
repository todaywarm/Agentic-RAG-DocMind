def test_swebug():
    from swe_bench import add
    assert add(1,1)==2, "1+1 should equal 2"#复现遇到的bug 同时也可以作为后续的pass test
