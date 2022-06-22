def cor_motxsignal(mot_params, nifti, mask=None, motparam_names=None):
    """
    Plot the correlation between motion parameters and neural signal.
    """
    
    mot_params   = mot_params.to_numpy()
    glob_tcourse = t_series__mask(nifti, mask)
    
    cor_mat = np.zeros(mot_params.shape[1])
    for param in range(mot_params.shape[1]):
        cor_mat[param], _ = pearsonr(glob_tcourse, mot_params[:,param])
    
    return cor_mat