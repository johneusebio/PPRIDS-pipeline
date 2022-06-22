def elem_uppertriangle(X):
    if X.ndim != 2:
        raise Exception("X must be a 2D numpy array.")
    dims = X.shape
    return int(dims[0]*(dims[1]-1)/2)