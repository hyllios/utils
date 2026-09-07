

def test_open_ended_window_bound_does_not_squash_the_y_axis():
    """Windows are routinely given sentinel bounds like (-1e3, 6.4) meaning
    'no lower bound'. Taking the y-range from those put every band of the
    w90 tutorials into a sliver at the top of a 1000 eV axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.linspace(0, 1, 40)
    dft = np.stack([-3 + 0.4 * np.cos(2 * np.pi * x), 1 + 0.3 * x], axis=1)
    wann = dft + 0.01

    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, dft, wann,
                               outer_window=(-1e3, 6.4), frozen_window=(-1e3, 2.0),
                               fermi_energy=0.0)
    lo, hi = ax.get_ylim()
    assert lo > -10.0 and hi < 10.0, f"y-axis blown out by the sentinel: {(lo, hi)}"
    assert lo <= wann.min() and hi >= wann.max()    # the model is fully visible
    plt.close('all')


def test_shaded_window_is_clipped_to_the_visible_range():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.linspace(0, 1, 20)
    dft = np.zeros((20, 1))
    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, dft, outer_window=(-1e6, 1e6))
    lo, hi = ax.get_ylim()
    assert ax.patches, 'the window was not shaded at all'
    for patch in ax.patches:             # no shaded span may exceed the axis
        y0 = patch.get_y(); y1 = y0 + patch.get_height()
        assert y0 >= lo - 1e-9 and y1 <= hi + 1e-9, (y0, y1, lo, hi)
    plt.close('all')


def test_explicit_ylim_still_wins():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.linspace(0, 1, 20)
    dft = np.zeros((20, 1))
    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, dft, outer_window=(-1e3, 5.0), ylim=(-7.0, 9.0))
    assert ax.get_ylim() == (-7.0, 9.0)
    plt.close('all')


def test_all_nan_dft_falls_back_to_the_wannier_range():
    """The DFT-unavailable path passes an all-NaN array; the axis must still
    frame the interpolation instead of collapsing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.linspace(0, 1, 20)
    wann = np.stack([-2 + 0 * x, 3 + 0 * x], axis=1)
    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, np.full_like(wann, np.nan), wann,
                               outer_window=(-1e3, 6.0))
    lo, hi = ax.get_ylim()
    assert lo <= -2.0 and hi >= 3.0 and lo > -50.0
    plt.close('all')


def test_range_ignores_semicore_dft_bands_the_model_excludes():
    """A bands run returns every band pw.x computed, including the semicore
    states exclude_bands dropped. Tutorial 14 has those at -56 and -26 eV
    while the model spans ~4 eV around E_F: ranging over the DFT set squashed
    the comparison as badly as the sentinel window did."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.linspace(0, 1, 30)
    wann = np.stack([-1 + 0.2 * np.cos(2 * np.pi * x), 2 + 0.3 * x], axis=1)
    deep = np.stack([np.full_like(x, -56.0), np.full_like(x, -26.0)], axis=1)
    dft = np.concatenate([deep, wann + 0.005], axis=1)

    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, dft, wann,
                               outer_window=(-1e3, 4.0), fermi_energy=0.0)
    lo, hi = ax.get_ylim()
    assert lo > -20.0, f"semicore bands dragged the axis down to {lo}"
    assert lo <= wann.min() and hi >= wann.max()
    plt.close('all')


def test_lines_are_broken_at_kpath_discontinuities():
    """Standard paths are not one connected walk -- ASE writes breaks as
    commas ('GXMGZRAZ,XR,MA') and puts both sides at the same coordinate.
    Joining them drew a band plunging vertically through tutorial 14 at X
    and M."""
    import numpy as np
    from waw.vis import break_at_path_jumps

    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0])          # one break at index 2->3
    v = np.arange(5.0).reshape(5, 1)
    xb, vb = break_at_path_jumps(x, v)
    assert np.isnan(xb[3]) and np.isnan(vb[3, 0])
    assert np.array_equal(xb[~np.isnan(xb)], x)      # no real point lost
    assert np.array_equal(vb[~np.isnan(vb[:, 0]), 0], v[:, 0])

    y = np.linspace(0.0, 1.0, 5)                     # connected path: untouched
    xb2, vb2 = break_at_path_jumps(y, v)
    assert np.array_equal(xb2, y) and np.array_equal(vb2, v)


def test_window_plot_breaks_wannier_lines_at_jumps():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from waw.vis import plot_wannierization_windows

    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    wann = np.array([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
    _, ax = plt.subplots()
    plot_wannierization_windows(ax, x, np.full_like(wann, np.nan), wann)
    line = [l for l in ax.lines if l.get_linestyle() == '-' and len(l.get_xdata()) > 3][0]
    assert np.isnan(np.asarray(line.get_ydata(), dtype=float)).any(), \
        'the Wannier line was drawn straight across the path break'
    plt.close('all')
