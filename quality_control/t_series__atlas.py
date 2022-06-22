def t_series__atlas(nifti, atlas):
    """
    Function returns the time series for each ROI in the specified brain atlas.
    """
    
    atlas_filename = atlas["maps"]
    
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory="nilearn_cache")
    return masker.fit_transform(nifti)