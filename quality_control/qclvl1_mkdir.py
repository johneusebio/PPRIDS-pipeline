def qclvl1__mkdir(path):
    path_plot     = Path(os.path.join(path, "plots"))
    path_plot_ica = Path(os.path.join(path, "plots", "ICA"))
    path_metric   = Path(os.path.join(path, "metrics"))
    
    path_plot.mkdir(parents=True, exist_ok=True)
    path_metric.mkdir(parents=True, exist_ok=True)
    path_plot_ica.mkdir(parents=True, exist_ok=True)
    return path_metric, path_plot, path_plot_ica