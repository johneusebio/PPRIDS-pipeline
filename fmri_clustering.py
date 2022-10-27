# clustering

import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cityblock

# the following is to customize the davies_bouldin_score to accept other distance functions
from sklearn.utils import check_X_y
from sklearn.utils import _safe_indexing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster._unsupervised import check_number_of_labels

def get_niimat(x):
    """Returns matrix array of nifti image

    Args:
        x (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image

    Raises:
        TypeError: x is not one of the accepted types

    Returns:
        np.ndarray: np.ndarray of nifti image data
    """
    if isinstance(x, str):
        x = nib.load(x)
    if isinstance(x, nib.nifti1.Nifti1Image):
        x = x.get_fdata()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError("x must be a one of the following types: str, nib.nifti1.Nifti1Image, np.ndarray")

## K means

def __newshape_nii_4Dto2D(dims):
    """Returns the dimensions of a 4D nifti file after being reshaped into a 2D array

    Args:
        dims (list): dimensions of the 4D nifti file

    Returns:
        list: the reshaped 2D dimensions
    """
    if len(dims)!=4:
        raise Exception("The 'dims' must have a length of 4.")
    return (np.prod(dims[:3]), dims[3])

def nii_4Dto2D(nii):
    """reshape a 4D nifti array into a 2D array

    Args:
        nii ([type]): 4D nifti image

    Returns:
        np.ndarray: 2D reshaped nii image
    """
    nii = get_niimat(nii)
    return np.reshape(nii, __newshape_nii_4Dto2D(nii.shape), order="F")

def kmeans(X:np.ndarray, k:int, 
           seed:int=None, max_iter:int=300, n_init:int=10, 
           just_cvi:bool=False, cvi_metric="euclidean"):
    """k means clustering of a 2D array, X
    
    Valid values for cvi_metric are:
    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.
    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Args:
        X (np.ndarray): matrix of data to be clustered
        k (int): number of k clusters
        seed (int, optional): Seed to use in the random initialization of centroids. Defaults to None.
        max_iter (int, optional): Maximum number of iterations to run. Defaults to 300.
        n_init (int, optional): Number of initializations to run. Defaults to 10.
        just_cvi (bool, optional): Only returns the Cluster Validity Index, defined as 
            the Davies-Bouldin Score. Defaults to False.
        cvi_metric : str or callable, default='euclidean'
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scipy.spatial.distance.pdist for its metric parameter, or
            a metric listed in ``pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
            If metric is "precomputed", X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from X as input and return a value indicating
            the distance between them.

    Returns:
        y_means, km: the fitted centroids, the kmeans object
    """
    km = KMeans(n_clusters=k, random_state=seed, max_iter=max_iter, n_init=n_init)
    km.fit(X)
    
    if just_cvi:
        return davies_bouldin_score__custom_metric(X, km.labels_, cvi_metric)
    
    y_kmeans = km.fit_predict(X)
    return y_kmeans, km, davies_bouldin_score__custom_metric(X, km.labels_, cvi_metric)

def davies_bouldin_score__custom_metric(X, labels, metric="euclidean"):
    """Compute the Davies-Bouldin score.
    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    .. versionadded:: 0.20
    
    Valid values for metric are:
    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.
    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in ``pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid], metric=metric))

    centroid_distances = pairwise_distances(centroids, metric=metric)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)