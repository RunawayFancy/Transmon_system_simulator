# -*- coding: utf-8 -*-
"""
@author: Jiheng Duan
@copyright qutip.visualization

The following code is addopted from qutip.

[1] J.R. Johansson, P.D. Nation, Franco Nori,
QuTiP: An open-source Python framework for the dynamics of open quantum systems,
Computer Physics Communications,
Volume 183, Issue 8,
2012,
Pages 1760-1772,
ISSN 0010-4655,
https://doi.org/10.1016/j.cpc.2012.02.021.


[2] Lambert, Neill, et al. "QuTiP 5: The Quantum Toolbox in Python." 
arXiv preprint arXiv:2412.04705 (2024).
"""

import itertools as it
import numpy as np
from numpy import pi, array, sin, cos, angle, log2, sqrt

from packaging.version import parse as parse_version

from qutip import Qobj, isket, ket2dm, tensor, vector_to_operator, settings

from qutip.core.superop_reps import _to_superpauli, isqubitdims
import qutip.visualization


try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(mpl.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)
except:
    pass

#%% 2D matrix histogram 

def hinton(rho, xlabels=None, ylabels=None, title=None, ax=None, cmap=None,
           label_top=True, color_style="scaled", xtk_rotate = None):
    """Draws a Hinton diagram for visualizing a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

    xlabels : list of strings or False
        list of x labels

    ylabels : list of strings or False
        list of y labels

    title : string
        title of the plot (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    cmap : a matplotlib colormap instance
        Color map to use when plotting.

    label_top : bool
        If True, x-axis labels will be placed on top, otherwise
        they will appear below the plot.

    color_style : string
        Determines how colors are assigned to each square:

        -  If set to ``"scaled"`` (default), each color is chosen by
           passing the absolute value of the corresponding matrix
           element into `cmap` with the sign of the real part.
        -  If set to ``"threshold"``, each square is plotted as
           the maximum of `cmap` for the positive real part and as
           the minimum for the negative part of the matrix element;
           note that this generalizes `"threshold"` to complex numbers.
        -  If set to ``"phase"``, each color is chosen according to
           the angle of the corresponding matrix element.

    xtk_rotate : float
        Rotate the x ticks by amount `xtk_rotate`
    
    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    Examples
    --------
    >>> import qutip
    >>>
    >>> dm = qutip.rand_dm(4)
    >>> fig, ax = qutip.hinton(dm)
    >>> fig.show()
    >>>
    >>> qutip.settings.colorblind_safe = True
    >>> fig, ax = qutip.hinton(dm, color_style="threshold")
    >>> fig.show()
    >>> qutip.settings.colorblind_safe = False
    >>>
    >>> fig, ax = qutip.hinton(dm, color_style="phase")
    >>> fig.show()
    """

    # Apply default colormaps.
    # TODO: abstract this away into something that makes default
    #       colormaps.
    cmap = (
        (cm.Greys_r if settings.colorblind_safe else cm.RdBu)
        if cmap is None else cmap
    )

    # Extract plotting data W from the input.
    if isinstance(rho, Qobj):
        if rho.isoper:
            W = rho.full()

            # Create default labels if none are given.
            if xlabels is None or ylabels is None:
                labels = _cb_labels(rho.dims[0])
                xlabels = xlabels if xlabels is not None else list(labels[0])
                ylabels = ylabels if ylabels is not None else list(labels[1])

        elif rho.isoperket:
            W = vector_to_operator(rho).full()
        elif rho.isoperbra:
            W = vector_to_operator(rho.dag()).full()
        elif rho.issuper:
            if not isqubitdims(rho.dims):
                raise ValueError("Hinton plots of superoperators are "
                                 "currently only supported for qubits.")
            # Convert to a superoperator in the Pauli basis,
            # so that all the elements are real.
            sqobj = _to_superpauli(rho)
            nq = int(log2(sqobj.shape[0]) / 2)
            W = sqobj.full().T
            # Create default labels, too.
            if (xlabels is None) or (ylabels is None):
                labels = list(map("".join, it.product("IXYZ", repeat=nq)))
                xlabels = xlabels if xlabels is not None else labels
                ylabels = ylabels if ylabels is not None else labels

        else:
            raise ValueError(
                "Input quantum object must be an operator or superoperator."
            )

    else:
        W = rho

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    if not (xlabels or ylabels):
        ax.axis('off')
    if title:
        ax.set_title(title)

    ax.axis('equal')
    ax.set_frame_on(False)

    height, width = W.shape

    w_max = 1.25 * max(abs(np.array(W)).flatten())
    if w_max <= 0.0:
        w_max = 1.0

    # Set color_fn here.
    if color_style == "scaled":
        def color_fn(w):
            w = np.abs(w) * np.sign(np.real(w))
            return cmap(int((w + w_max) * 256 / (2 * w_max)))
    elif color_style == "threshold":
        def color_fn(w):
            w = np.real(w)
            return cmap(255 if w > 0 else 0)
    elif color_style == "phase":
        def color_fn(w):
            return cmap(int(255 * (np.angle(w) / 2 / np.pi + 0.5)))
    else:
        raise ValueError(
            "Unknown color style {} for Hinton diagrams.".format(color_style)
        )

    ax.fill(array([0, width, width, 0]), array([0, 0, height, height]),
            color=cmap(128))
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            _blob(
                _x - 0.5, height - _y + 0.5, W[y, x], w_max,
                min(1, abs(W[y, x]) / w_max), color_fn=color_fn, ax=ax)

    # color axis
    vmax = np.pi if color_style == "phase" else abs(W).max()
    norm = mpl.colors.Normalize(-vmax, vmax)
    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
    mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)

    xtics = 0.5 + np.arange(width)
    # x axis
    ax.xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        if xtk_rotate: xtk_ang = xtk_rotate
        else: xtk_ang = 0
        ax.set_xticklabels(xlabels, rotation=xtk_ang)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=14)

    # y axis
    ytics = 0.5 + np.arange(height)
    ax.yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=14)

    return fig, ax

def _cb_labels(left_dims):
    """Creates plot labels for matrix elements in the computational basis.

    Parameters
    ----------
    left_dims : flat list of ints
        Dimensions of the left index of a density operator. E. g.
        [2, 3] for a qubit tensored with a qutrit.

    Returns
    -------
    left_labels, right_labels : lists of strings
        Labels for the left and right indices of a density operator
        (kets and bras, respectively).
    """
    # FIXME: assumes dims, such that we only need left_dims == dims[0].
    basis_labels = list(map(",".join, it.product(*[
        map(str, range(dim))
        for dim in left_dims
    ])))
    return [
        map(fmt.format, basis_labels) for fmt in
        (
            r"$\langle{}|$",
            r"$|{}\rangle$",
        )
    ]

# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, color_fn, ax=None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = array([x - hs, x + hs, x + hs, x - hs])
    ycorners = array([y - hs, y - hs, y + hs, y + hs])

    if ax is not None:
        handle = ax
    else:
        handle = plt

    handle.fill(xcorners, ycorners,
             color=color_fn(w))

#%% 3D matrix histogram 

def matrix_histogram(M, xlabels=None, ylabels=None, title=None, limits=None, xy_tk_rotate=None,
                     colorbar=True, fig=None, ax=None, options=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot (optional)

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    xy_tk_rotate : list/array with three float
        Rotate xticks_label/y/z with corresponding angle value.

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    colorbar : bool (default: True)
        show colorbar

    options : dict
        A dictionary containing extra options for the plot.
        The names (keys) and values of the options are
        described below:

        'zticks' : list of numbers
            A list of z-axis tick locations.

        'cmap' : string (default: 'jet')
            The name of the color map to use.

        'cmap_min' : float (default: 0.0)
            The lower bound to truncate the color map at.
            A value in range 0 - 1. The default, 0, leaves the lower
            bound of the map unchanged.

        'cmap_max' : float (default: 1.0)
            The upper bound to truncate the color map at.
            A value in range 0 - 1. The default, 1, leaves the upper
            bound of the map unchanged.

        'bars_spacing' : float (default: 0.1)
            spacing between bars.

        'bars_alpha' : float (default: 1.)
            transparency of bars, should be in range 0 - 1

        'bars_lw' : float (default: 0.5)
            linewidth of bars' edges.

        'bars_edgecolor' : color (default: 'k')
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.

        'shade' : bool (default: True)
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.

        'azim' : float
            The azimuthal viewing angle.

        'elev' : float
            The elevation viewing angle.

        'proj_type' : string (default: 'ortho' if ax is not passed)
            The type of projection ('ortho' or 'persp')

        'stick' : bool (default: False)
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.

        'cbar_pad' : float (default: 0.04)
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).

        'cbar_to_z' : bool (default: False)
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).

        'figsize' : tuple of two numbers
            The size of the figure.

    Returns :
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    # default options
    default_opts = {'figsize': None, 'cmap': 'jet', 'cmap_min': 0.,
                    'cmap_max': 1., 'zticks': None, 'bars_spacing': 0.2,
                    'bars_alpha': 1., 'bars_lw': 0.5, 'bars_edgecolor': 'k',
                    'shade': False, 'azim': -35, 'elev': 35,
                    'proj_type': 'ortho', 'stick': False,
                    'cbar_pad': 0.04, 'cbar_to_z': False}

    # update default_opts from input options
    if options is None:
        pass
    elif isinstance(options, dict):
        # check if keys in options dict are valid
        if set(options) - set(default_opts):
            raise ValueError("invalid key(s) found in options: "
                             f"{', '.join(set(options) - set(default_opts))}")
        else:
            # updating default options
            default_opts.update(options)
    else:
        raise ValueError("options must be a dictionary")

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() + 0.5
    ypos = ypos.T.flatten() + 0.5
    zpos = np.zeros(n)
    dx = dy = (1 - default_opts['bars_spacing']) * np.ones(n)
    dz = np.real(M.flatten())

    if isinstance(limits, list) and len(limits) == 2:
        z_min = limits[0]
        z_max = limits[1]
    else:
        z_min = min(dz)
        z_max = max(dz)
        if z_min == z_max:
            z_min -= 0.1
            z_max += 0.1

    if default_opts['cbar_to_z']:
        norm = mpl.colors.Normalize(min(dz), max(dz))
    else:
        norm = mpl.colors.Normalize(z_min, z_max)
    cmap = _truncate_colormap(default_opts['cmap'],
                              default_opts['cmap_min'],
                              default_opts['cmap_max'])
    colors = cmap(norm(dz))

    if ax is None:
        fig = plt.figure(figsize=default_opts['figsize'])
        ax = _axes3D(fig,
                     azim=default_opts['azim'] % 360,
                     elev=default_opts['elev'] % 360)
        ax.set_proj_type(default_opts['proj_type'])

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,
             edgecolors=default_opts['bars_edgecolor'],
             linewidths=default_opts['bars_lw'],
             alpha=default_opts['bars_alpha'],
             shade=default_opts['shade'])
    # remove vertical lines on xz and yz plane
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.xaxis._axinfo["grid"]['linewidth'] = 0

    if title:
        ax.set_title(title)
    if xy_tk_rotate: xtk_angle, ytk_angle = xy_tk_rotate
    else: xtk_angle, ytk_angle = [0,0]
    # x axis
    _update_xaxis(default_opts['bars_spacing'], M, ax, xlabels, xtk_angle)

    # y axis
    _update_yaxis(default_opts['bars_spacing'], M, ax, ylabels, ytk_angle)

    # z axis
    _update_zaxis(ax, z_min, z_max, default_opts['zticks'])

    # stick to xz and yz plane
    _stick_to_planes(default_opts['stick'],
                     default_opts['azim'], ax, M,
                     default_opts['bars_spacing'])

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75,
                                         pad=default_opts['cbar_pad'])
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    # removing margins
    _remove_margins(ax.xaxis)
    _remove_margins(ax.yaxis)
    _remove_margins(ax.zaxis)

    return fig, ax

def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    truncates portion of a colormap and returns the new one
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(
            n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def _remove_margins(axis):
    """
    removes margins about z = 0 and improves the style
    by monkey patching
    """
    def _get_coord_info_new(renderer):
        mins, maxs, centers, deltas, tc, highs = \
            _get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs

    _get_coord_info_old = axis._get_coord_info
    axis._get_coord_info = _get_coord_info_new

def _update_yaxis(spacing, M, ax, ylabels, ytk_angle):
    """
    updates the y-axis
    """
    ytics = [x + (1 - (spacing / 2)) for x in range(M.shape[1])]
    if parse_version(mpl.__version__) >= parse_version("3.8"):
        ax.axes.yaxis.set_major_locator(plt.FixedLocator(ytics))
    else:
        ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(ylabels, rotation=ytk_angle)
    else:
        ax.set_yticklabels([str(y + 1) for y in range(M.shape[1])], rotation=ytk_angle)
        ax.set_yticklabels([str(i) for i in range(M.shape[1])], rotation=ytk_angle)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([y + (1 - (spacing / 2)) for y in range(M.shape[1])])

def _update_zaxis(ax, z_min, z_max, zticks):
    """
    updates the z-axis
    """
    if parse_version(mpl.__version__) >= parse_version("3.8"):
        ax.axes.zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    else:
        ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(1, 0.5))

    if isinstance(zticks, list):
        ax.set_zticks(zticks)
    ax.set_zlim3d([min(z_min, 0), z_max])

def _update_xaxis(spacing, M, ax, xlabels, xtk_angle):
    """
    updates the x-axis
    """
    xtics = [x + (1 - (spacing / 2)) for x in range(M.shape[1])]
    if parse_version(mpl.__version__) >= parse_version("3.8"):
        ax.axes.xaxis.set_major_locator(plt.FixedLocator(xtics))
    else:
        ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))

    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels, rotation=xtk_angle)
    else:
        ax.set_xticklabels([str(x + 1) for x in range(M.shape[0])], rotation=xtk_angle)
        ax.set_xticklabels([str(i) for i in range(M.shape[0])], rotation=xtk_angle)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xticks([x + (1 - (spacing / 2)) for x in range(M.shape[0])])

def _stick_to_planes(stick, azim, ax, M, spacing):
    """adjusts xlim and ylim in way that bars will
    Stick to xz and yz planes
    """
    if stick is True:
        azim = azim % 360
        if 0 <= azim <= 90:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(1 - .5,)
        elif 90 < azim <= 180:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 180 < azim <= 270:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 270 < azim < 360:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(1 - .5,)