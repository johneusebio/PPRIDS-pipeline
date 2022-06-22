def fcmat__plot(mat, atlas_labels, savepath):
    """
    Plot and save the FC matrix between ROIs.
    """
    
    fig = plt.figure(figsize=(10,10))
    np.fill_diagonal(mat, 0)
    plotting.plot_matrix(mat, labels=atlas_labels, colorbar=True, vmax=0.8, vmin=-0.8, figure=fig)
    fig.savefig(savepath)
    return