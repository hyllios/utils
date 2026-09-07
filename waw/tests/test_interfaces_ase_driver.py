

def test_two_point_direction_warns():
    """N_i=2 leaves the interpolation unconstrained between k=0 and k=1/2
    (single harmonic) -- the NiI2 Gamma-A artifact, 436 meV on a real model."""
    import warnings
    from waw.interfaces.ase.driver import _warn_on_two_point_directions

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_on_two_point_directions((6, 6, 2))
    assert len(w) == 1 and "only 2 k-points along c" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_on_two_point_directions((2, 6, 2))
    assert len(w) == 1 and "along a, c" in str(w[0].message)

    for grid in [(6, 6, 6), (6, 6, 1), (1, 1, 1), (4, 4, 4)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_on_two_point_directions(grid)
        assert not w, f"{grid} should not warn"     # N=1 is dispersionless, not underdetermined
