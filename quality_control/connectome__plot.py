def connectome__plot(mat, region_coords, savepath):
    """
    Plot and save the connectome plot between ROIs.
    """
    fig = plt.figure(figsize=(10,10))
    plotting.plot_connectome(mat, region_coords, edge_threshold="80%", colorbar=True, figure=fig)
    fig.savefig(savepath)
    return