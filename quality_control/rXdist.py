def rXdist(X, coords):
    """
    Compute the correlation and euclidean distance between each pair of ROIs.
    """
    n_uppertriangle = elem_uppertriangle(X)
    
    r_tab    = np.zeros([n_uppertriangle])   # correlation (r) b/w rois A & B
    a_tab    = np.zeros([n_uppertriangle,3]) # coordinates of roi A (rows)
    b_tab    = np.zeros([n_uppertriangle,3]) # coorindates of roi B (columns)
    dist_tab = np.zeros([n_uppertriangle])   # Euclidean dist b/w rois A & B
    
    dims = X.shape
    count=0
    for rr in range(dims[0]):
        for cc in range(dims[1]):
            if rr >= cc:
                continue
            r_tab   [count]  = X[rr,cc]
            a_tab   [count,] = list(coords[rr])
            b_tab   [count,] = list(coords[cc])
            dist_tab[count]  = abs(euc_dist(a_tab[count,], b_tab[count,]))
            
            count+=1
    return r_tab, a_tab, b_tab, dist_tab