def rXdist__diff(X2, X1, coords):
    r_tab1, a_tab1, b_tab1, dist_tab1 = rXdist(X1, coords)
    r_tab2,      _,      _,         _ = rXdist(X2, coords)
    
    r_tab_diff = r_tab2 - r_tab1
    return r_tab_diff, a_tab1, b_tab1, dist_tab1