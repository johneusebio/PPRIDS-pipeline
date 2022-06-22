def auto_labels(nifti):
    """
    This function automatically generates labels for the provided atlas
    (provided in nifti format) as a set of integers corresponging to each ROI.
    :param nifti: nifti object, as provided by nibabel nib.load()
    :return: list of labels
    """
    
    lab = (np.unique(nifti.get_fdata())).astype("int")
    lab = lab[lab != 0]
    
    return lab.astype("str")