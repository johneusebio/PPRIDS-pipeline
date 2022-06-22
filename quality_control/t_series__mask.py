def t_series__mask(nii, mask=None):
    """
    Functions returns the mean time course within the specified binary nifti mask ROI.
    
    If no mask is provided, all 0 voxels are excluded and the raw mean time course for 
    the entire nifti file is returned.
    """
    if mask is not None:
        glob = apply_mask(nii, mask)
        return np.mean(glob, axis=1)
        
    nii[nii==0] = "nan"
    return np.nanmean(nii, axis=(0,1,2))