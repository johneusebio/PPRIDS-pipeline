def import_motion(filepath, names=None):
    """
    Import the motion parameters.
    """
    # TODO: set names=0 to use the first line of the text file
    if names is None:
        return pd.read_csv(filepath, header=None, sep=r"\s+")
    return pd.read_csv(filepath, names=names, sep=r"\s+")