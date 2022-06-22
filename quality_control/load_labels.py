def load_labels(csv_path):
    """
    Returns a list of provided in a csv file.
    :param csv_path: path to csv file (str)
    :return: list of labels
    """
    labels = np.genfromtxt(csv_path, delimiter=',', dtype="str")
    if labels.ndim != 1:
        raise Exception("labels must be a 1-D file (i.e., single-column)")
    return list(labels)