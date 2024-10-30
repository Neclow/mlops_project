def set_size(width, layout="h", fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Heavily inspired by: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
        Report: 390 pt
    layout: string
        h: horizontal layout
        v: vertical layout
        s: square layout
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio for aesthetic figures
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if layout == "h":
        fig_height_in = fig_width_in * golden_ratio
    elif layout == "v":
        fig_height_in = fig_width_in / golden_ratio
    elif layout == "s":
        fig_height_in = fig_width_in
    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
