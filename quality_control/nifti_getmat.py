def nifti_getmat(nifti):
    """
    Import nifti file and returns the 3- or 4-d matrix data.
    :param nifti: full path to nifti file
    :return: nfiti matrix data
    """
    img = nib.load(nifti)
    return img.get_fdata()