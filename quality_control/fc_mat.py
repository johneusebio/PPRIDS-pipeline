def fc_mat(nifti, atlas):
    roi_t = t_series__atlas(nifti, atlas)
    correlation_measure = ConnectivityMeasure(kind="correlation")
    return correlation_measure.fit_transform([roi_t])[0], roi_t