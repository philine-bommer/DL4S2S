import os
from math import log10, ceil, floor 
import numpy as np
import matplotlib.pyplot as plt


def set_fig_size(width='thesis', width_fraction=1, subplots=(1, 1), subplot_fraction=1, ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 399.47578
    else:
        width_pt = width

    #Width of figure (in pts)
    fig_width_pt = width_pt * width_fraction   
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if ratio is None:
        # golden ratio
        ratio = (5**.5 - 1) / 2
    
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (fig_width_in * ratio * (subplots[0] / subplots[1])) / subplot_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def label_axis(ax, values, mpl_ax, v_min=None, v_max=None):
    v = list(values)
    
    if v_min is None:
        v_min = np.min(v)
    if v_max is None:
        v_max = np.max(v)

    exp = -int(floor(log10(abs(v_max))))

    bottom = round(v_min, exp) 
    top = round(v_max, exp)

    if bottom > v_min:
        bottom  -= 10**(-exp)
    
    if top < v_max:
        top += 10**(-exp)
    
    if ax == 'x':
        mpl_ax.set_xlim(v_min, v_max)
        mpl_ax.set_xticks(np.linspace(bottom, top, 5))
    elif ax == 'y':
        mpl_ax.set_ylim(v_min, v_max)
        mpl_ax.set_yticks(np.linspace(bottom, top, 5))


''' Based on paper by Guo et. al. 2017 -  On Calibration of Modern Neural Networks'''


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(ax, bin_data, 
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel="Count"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.9)
   
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def _reliability_diagram_combined(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
                                 title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
    ax[1].set_yticklabels(new_ticks)    

    plt.show()

    if return_fig: return fig


def reliability_diagram(true_labels, pred_labels, confidences, num_bins=10,
                        draw_ece=True, draw_bin_importance=False, 
                        draw_averages=True, title="Reliability Diagram", 
                        figsize=(6, 6), dpi=72, return_fig=False):
    """Draws a reliability diagram and confidence histogram in a single plot.
    
    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average 
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin. 
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence. 
    The closer these two lines are together, the better the calibration.
    
    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance,
                                         draw_averages, title, figsize=figsize, 
                                         dpi=dpi, return_fig=return_fig)


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_bin_importance=False, 
                         num_cols=4, dpi=72, return_fig=False):
    """Draws reliability diagrams for one or more models.
    
    Arguments:
        results: dictionary where the key is the model name and the value is
            a dictionary containing the true labels, predicated labels, and
            confidences for this model
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        num_cols: how wide to make the plot
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    ncols = num_cols
    nrows = (len(results) + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 4)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
                           figsize=figsize, dpi=dpi, constrained_layout=True)

    for i, (plot_name, data) in enumerate(results.items()):
        y_true = data["true_labels"]
        y_pred = data["pred_labels"]
        y_conf = data["confidences"]
        
        bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins)
        
        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[row, col], bin_data, draw_ece, 
                                     draw_bin_importance, 
                                     title="\n".join(plot_name.split()),
                                     xlabel="Confidence" if row == nrows - 1 else "",
                                     ylabel="Expected Accuracy" if col == 0 else "")

    for i in range(i + 1, nrows * ncols):
        row = i // ncols
        col = i % ncols        
        ax[row, col].axis("off")
        
    plt.show()

    if return_fig: return fig

def label_axes_grid(fig, axes, xlims=None, ylims=None, xlabel=None, ylabels=None, ylabel=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, grid_x=True, grid_y=True):
    for row_i, row_axes in enumerate(axes):
        for col_i, ax in enumerate(row_axes):
            if xticks is not None: ax.set_xticks(xticks)
            if yticks is not None: ax.set_yticks(yticks)
            if xlims is not None: ax.set_xlim(xlims)
            if ylims is not None: ax.set_ylim(ylims)
            if xticklabels is not None: ax.set_xticklabels(xticklabels)
            if yticklabels is not None: ax.set_yticklabels(yticklabels)
            if xlabel is not None:
                if row_i == len(axes)-1:
                    if isinstance(xlabel, list):
                        ax.set_xlabel(xlabel[col_i])
            if ylabels is not None:
                if col_i == 0:
                    if isinstance(ylabels, list):
                        ax.set_ylabel(ylabels[row_i])

            ax.grid(visible=False, axis='both')
            if grid_x: ax.grid(visible=True, axis='x')
            if grid_y: ax.grid(visible=True, axis='y')

            if row_i < len(axes)-1:
                ax.set_xticklabels([])
            if col_i != 0:
                ax.set_yticklabels([])

    if isinstance(xlabel, str):
        fig.text(0.5, 0.05, xlabel, ha='center')
    if isinstance(ylabel, str):
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

def plot_class_dist(preds,
                    targets,
                    xticklabs = ['NAO+', 'SB', 'AR', 'NAO-']):
    """Plot distribution of classes for each leadtime comparing predicted vs target labels

    Args:
        preds (np.ndarray): predicted class labels
        targets (np.ndarray): traget class labels
        xticklabs (list): list of class names

    Returns:
        fig-object: figure object
    """
    fig, axes = plt.subplots(6,2, figsize=set_fig_size(subplots=(6,2)))
    y_max = 0
    for row in range(6):
        # how often was each class predicted in this time step
        counts = [sum(i == preds[:, row]) for i in range(4)]
        y_max = max(counts) if max(counts) > y_max else y_max
        axes[row][0].bar(x=np.arange(4), height=counts)
        axes[row][0].set_title(f'Predicted: Leadtime = week {row+1}')
        # how often does each class occur in this time step
        counts = [sum(i == targets[:, row]) for i in range(4)]
        y_max = max(counts) if max(counts) > y_max else y_max
        axes[row][1].bar(x=np.arange(4), height=counts)
        axes[row][1].set_title(f'Actual: Leadtime = week {row+1}')

    label_axes_grid(fig, axes, ylims=(0, y_max+10), xticks=np.arange(4), xticklabels=xticklabs, xlabel='Classes', ylabel='Number of samples', yticks=np.arange(0, y_max+10, 50), grid_x=False)


    plt.show()
    return fig

def plot_joint_class_dist(pred_dict,
                    targets,
                    cm_list,
                    xticklabs = ['NAO+', 'SB', 'AR', 'NAO-']):
    """Plot distribution of classes for each leadtime comparing predicted vs target labels

    Args:
        preds (np.ndarray): predicted class labels
        targets (list): traget class labels
        xticklabs (list): list of class names

    Returns:
        fig-object: figure object
    """
    # fig, axes = plt.subplots(6,len(pred_dict)+1, figsize=plot_utils.set_fig_size(subplots=(6,len(pred_dict)+1)), sharey = True)
    fig, axes = plt.subplots(6,len(pred_dict), figsize=(20,15))#, sharey = True)
    y_max = 0
    col  = 0
    for keys, preds in pred_dict.items():
        # preds = pred_dict[keys]
        print(preds.shape)
        for row in range(6):
            # how often does each class occur in this time step
            counts_tar = [sum(i == targets[:, row]) for i in range(4)]
            y_max = max(counts_tar) if max(counts_tar) > y_max else y_max
            axes[row][col].bar(x=np.arange(4), height=counts_tar, alpha = 0.5, color ='#000000', label = 'target')
            # how often was each class predicted in this time step
            counts = [sum(i == preds[:, row]) for i in range(4)]
            y_max = max(counts) if max(counts) > y_max else y_max
            axes[row][col].bar(x=np.arange(4), height=counts, alpha = 0.5, color =cm_list[col], label = 'predictions')
            # axes[row][col].legend()
            if col == 0:
                # counts_tar = [sum(i == targets[:, row]) for i in range(4)]
                # y_max = max(counts_tar) if max(counts_tar) > y_max else y_max
                # axes[row][-1].bar(x=np.arange(4), height=counts_tar)
                axes[row][col].set_ylabel(f'Leadtime= week {row+1}')
            if row == 0:
                axes[row][col].set_title(f'{keys} predictions')
                # axes[row][-1].set_title(f'Target distribution')
        col +=1 

    label_axes_grid(fig, axes, ylims=(0, y_max+10), xticks=np.arange(4), xticklabels=xticklabs, xlabel='Classes', ylabel='Number of samples', yticks=np.arange(0, y_max+10, 50), grid_x=False)


    plt.show()
    return fig