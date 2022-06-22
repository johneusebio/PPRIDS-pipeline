def load_atlas(atlas=None, labels=None, rois=None):
    """
    Load atlas provided in nifti format. Labels and rois must correspond
    to each of the regions defined in the provided nifti format.
    
    If no atlas is provided (default=None), it will load MSDL brain atlas 
    provided by nilearn.
    
    If an atlas is specified without a list of labels, they will be numbered. 
    :param atlas: specified nifti file of brain atlas (default=None)
    :param labels: file path to csv file containing ROI labels (default=None)
    :param rois: path to csv file containing ROI coordinates
    """
    
    if atlas is None:
        return datasets.fetch_atlas_msdl()
    # if not isfile(atlas):
    #     raise Exception("Must provide a valid atlas file path or leave it blank.")
    # labels = auto_labels(atlas) if labels is None else load_labels(labels)
    # if rois is None:
    #     raise Exception("Must provide roi coordinates.")
    # else:
    #     rois=np.genfromtxt(rois, delimiter=',')
    # return {"maps": atlas, "labels": labels, "rois": rois}