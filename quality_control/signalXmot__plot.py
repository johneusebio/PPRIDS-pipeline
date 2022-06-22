def signalXmot__plot(data, labels=None, xlabel=None, ylabel=None, title=None):
    """
    Create and save a bar plot for QC metrics.
    """
    if labels is None:
        labels = [str(x+1) for x in range(len(data))]
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,.9,.8])
    ax.bar(labels, data)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axhline(y=0, color="black")
    return fig
