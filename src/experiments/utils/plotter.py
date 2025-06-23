import os
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.config import SHOULD_SAVE_PLOT, SHOULD_SHOW_PLOT

def show_plot():
    if SHOULD_SHOW_PLOT:
        plt.show()

def save_plot(*, figure, filename, directory):
    """
    Save `figure` to `directory/filename` if saving is enabled.
    """
    if SHOULD_SAVE_PLOT:
        if not filename.endswith('.png'): 
            filename += '.png'
        os.makedirs(directory, exist_ok=True)
        figure.savefig(os.path.join(directory, filename), bbox_inches='tight')

def plot_series(
    *,
    x,
    ys,
    labels=None,
    title=None,
    xlabel=None,
    ylabel=None,
    log_y=False,
    filename,
    directory,
):
    """
    Plot multiple series on a single figure, with optional saving & display.

    Parameters:
    - x: sequence of x values
    - ys: list of sequences of y values (each same length as x)
    - labels: optional list of labels for each series (for legend)
    - title: optional plot title
    - xlabel, ylabel: optional axis labels
    - log_y: bool, if True sets y-axis to log scale
    - filename: base filename (no extension needed)
    - directory: path to save the figure
    """
    fig, ax = plt.subplots()

    # plot each series
    for i, y in enumerate(ys):
        lbl = labels[i] if labels and i < len(labels) else None
        # filtrar x,y para eliminar los y==None
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys_f = [yi for yi in y            if yi is not None]
        ax.plot(xs, ys_f, marker='o', label=lbl)

    # titles and labels
    if title:    ax.set_title(title)
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)
    if log_y:    ax.set_yscale('log')

    # force all x values to appear
    ax.set_xticks(x)

    # add gridlines at every major tick
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # legend if needed
    if labels: ax.legend()

    plt.tight_layout()

    # save & show
    save_plot(figure=fig, filename=filename, directory=directory)
    show_plot()


def plot_heatmap(
    *,
    x, y,
    title=None,
    xlabel=None,
    ylabel=None,
    hexbin=False,
    gridsize=30,
    cmap='Reds',
    bins_x=None,
    bins_y=None,
    class_labels_x=None,
    class_labels_y=None,
    filename,
    directory,
):
    """
    Plot a heatmap comparing two metrics, in continuous (hexbin) or
    categorical modes, with optional saving & display.

    Parameters:
    - x, y: 1D arrays of the same length. 
    - hexbin: if True, use hexagonal binning (continuous density).
    - gridsize, cmap: hexbin options.
    - bins_x, bins_y: lists of bin edges for categorical mode.
    - class_labels_x, class_labels_y: labels for categorical ticks.
    - filename: base filename for saving.
    - directory: directory to save the figure (if SHOULD_SAVE_PLOT).
    
    """
    fig, ax = plt.subplots()

    if hexbin:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Frecuencia')

    else:
        if bins_x is None or bins_y is None:
            raise ValueError("Provide bins_x and bins_y for categorical mode")
        H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y])
        mesh = ax.pcolormesh(xedges, yedges, H.T, cmap=cmap)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label('Frecuencia')

        # annotate & set ticks
        for i in range(len(bins_x)-1):
            for j in range(len(bins_y)-1):
                cnt = int(H[i,j])
                cx = (xedges[i] + xedges[i+1]) / 2
                cy = (yedges[j] + yedges[j+1]) / 2
                ax.text(cx, cy, str(cnt), ha='center', va='center')
        xt = (xedges[:-1] + xedges[1:]) / 2
        yt = (yedges[:-1] + yedges[1:]) / 2
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        if class_labels_x:
            ax.set_xticklabels(class_labels_x)
        if class_labels_y:
            ax.set_yticklabels(class_labels_y)

    # titles and labels
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    plt.tight_layout()

    # Show and save
    save_plot(figure=fig, filename=filename, directory=directory)
    show_plot()
        
def plot_histogram(
    *,
    x,
    ys,
    labels=None,
    colors=None,
    width_factor=0.8,
    title=None,
    xlabel=None,
    ylabel=None,
    filename,
    directory,
):
    """
    Plot one or more bar‐series at positions x, with dynamic spacing
    and value‐labels on top of each bar.
    """

    x = np.array(x)
    n_series = len(ys)

    # Figure out the minimal horizontal gap between x‐values
    if len(x) > 1:
        xs = np.sort(x)
        min_sep = np.min(np.diff(xs))
    else:
        min_sep = 1.0

    # Set the total group width as that minimal gap scaled
    group_w = min_sep * width_factor

    # Derive per‐series bar widths & offsets
    if n_series > 1:
        single_w = group_w / n_series
        offsets  = (np.arange(n_series) - (n_series-1)/2) * single_w
    else:
        single_w = group_w
        offsets  = [0]

    # Plotting
    fig, ax = plt.subplots()
    for i, y in enumerate(ys):
        xi = x + offsets[i]
        bars = ax.bar(
            xi,
            y,
            width=single_w * 0.9,                        # tiny inner gap
            color=(colors[i] if colors else None),
            label=(labels[i] if labels else None)
        )
        # Annotate each bar with its height
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x‐position: center of bar
                h,                                  # y‐position: top of bar
                f"{h:.0f}",                         # label text
                ha='center',
                va='bottom',
                fontsize='small'
            )

    ax.set_xticks(x)
    if labels:
        ax.legend()
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    plt.tight_layout()

    # Show and save
    save_plot(figure=fig, filename=filename, directory=directory)
    show_plot()
