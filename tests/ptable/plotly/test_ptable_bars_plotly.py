def test_ptable_bars_plotly():
    from pymatviz.ptable.plotly import ptable_bars_plotly

    data = {"Fe": {"1": 10.0, "2": 20.0}, "O": {"a": 5.0}}
    fig = ptable_bars_plotly(data, colorbar=False)
    # at least two bar traces present
    assert any(t.type == "bar" for t in fig.data)
    # no error on rendering JSON
    fig_json = fig.to_plotly_json()
    assert "data" in fig_json and len(fig_json["data"]) >= 2
