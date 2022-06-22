def rXdist__plot(r, dist, savepath):
    """
    Plot the correlation between each pair of ROIs as a function of the 
    euclidean distance between them.
    """
    
    fig = plt.figure(figsize=(20,10))
    ax  = fig.add_axes([0.05,0.05,.9,.9])
    ax.scatter(dist, r, color="black")
    ax.axhline(y=0, color="r", linestyle="dashed")
    ax.set_xlabel("Euclidean Distance Between ROIs (vox)")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("FC as a Function of Euclidean Distance Between ROIs")
    fig.savefig(savepath)
    return