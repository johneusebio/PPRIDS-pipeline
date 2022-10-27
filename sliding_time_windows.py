# sliding time windows

import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

import input_import as inm


def which(X):
    X_bool = pd.Series(X)
    return X_bool[X_bool].index.values

def taper_twin(len:int, sigma=0):
    """Taper the sliding time window with a gaussian filter of the specified sigma.

    Args:
        len (int): Length of the time window, in volumes.
        sigma (int, optional): Sigma of the gaussian filter. Defaults to 0.

    Returns:
        list: weights of the tapered time window
    """
    X = np.ones(len)
    if sigma==0:
        return X
    return gaussian_filter1d(X, sigma=sigma, mode="constant")

def n_twin(n:int, win_sz:int, step_sz:int):
    """compute the number of time windows within the time series, given the specified window size and step size.

    Args:
        n (int): number of time points in the time series.
        win_sz (int): Width of the time window.
        step_sz (int): Step size of the time window.

    Returns:
        int: total number of time windows in the time series
    """
    return int(np.floor((n - win_sz) / step_sz))

def twin_trange(iter:int, win_sz:int, step_sz:int):
    """Returns the indices for the start and end of the current time window.

    Args:
        iter (int): The nth time window you're currently on.
        win_sz (int): Width of the time window.
        step_sz (int): Step size of the time window.

    Returns:
        range: range of the start-end of the current nth time window 
    """
    start=iter*step_sz
    return range(start, start+win_sz)

def fc_triu(X:np.ndarray):
    """Returns the upper-triangle of a 3D numpy array.

    Args:
        X (np.ndarray): 3D numpy array, each the third axis (axis=2) represents distinct matrices

    Raises:
        Exception: X is not 3-dimensional

    Returns:
        np.ndarray: the upper-triangle of X, all other values set to 0.
    """
    if (X.ndim!=3):
        raise Exception("X must be a 3D numpy.ndarray.")
    X_triu = np.empty(X.shape)
    for ii in range(X.shape[2]):
        X_triu[:,:,ii] = np.triu(X[:,:,ii], k=1)
    return X_triu

def cov2cor(X:np.ndarray):
    """Convert covariance matrix to a correlation matrix

    Args:
        X (np.ndarray): Covariance matrix.

    Returns:
        np.ndarray: correlation matrix
    """
    X_diag = np.reshape(np.diag(X), (np.diag(X).shape[0],1))
    Xij_cov = np.multiply(X_diag, np.transpose(X_diag))
    return X/np.sqrt(Xij_cov)

def correl_weight(X:np.ndarray, w:list=None):
    """Perform weighted correlation

    Args:
        X (np.ndarray): Matrix of values to correlate
        w (list, optional): List of weights to apply to each value during correlation. Defaults to None.

    Returns:
        [type]: [description]
    """
    if w is None:
        w = np.ones(X.shape[1])
    X_cov = np.cov(X,  aweights=w)
    return cov2cor(X_cov)

def fc_twin(tab:np.ndarray, win_sz:int, step_sz:int, sigma:int=0, normalize=True):
    """Compute the sliding time window FC matrices for a given input

    Args:
        tab (np.ndarray): Should be a 2D array: rows = timepoints; columns = ROIs
        win_sz (int): Size of the sliding time window, in TRs
        step_sz (int): Step size of the time window, in TRs
        sigma (int): the sigma (in TRs) for convolution of time window with Gaussian kernel (default=0)

    Returns:
        [np.ndarray]: A 3D array of correlation matrices, with each time window being represented on the 3rd axis
    """
    ntwin    = n_twin(tab.shape[0], win_sz, step_sz)
    twin_mat = np.empty((tab.shape[1], tab.shape[1], ntwin))
    
    for iter in range(ntwin):
        trange = twin_trange(iter, win_sz, step_sz)
        if normalize:
            twin_mat[:,:,iter] = np.arctanh(correl_weight(np.transpose(tab[trange, :]), taper_twin(win_sz, sigma)))
        else:
            twin_mat[:,:,iter] = correl_weight(np.transpose(tab[trange, :]), taper_twin(win_sz, sigma))
    return twin_mat

def fc(X_path, outdir):
    fc_mat = correl_weight(np.transpose(np.load(X_path)))
    np.save(os.path.join(outdir, "fc_mat.npy"), fc_mat)
    return fc_mat
    
def dfc(X_path:str, win_sz:int, step_sz:int, outdir:str=None, sigma=0, normalize=True):
    """Compute dynamic functional connectivity from ROI time series

    Args:
        X_path (str): File path to the ROI time series
        win_sz (int): window size, in volumes
        step_sz (int): step size, in volumes
        outdir (str, optional): Output directory to save the dynamic functional connecity matrices. Defaults to None.
        sigma (int, optional): Sigma value for the gaussian kernel used to taper the sliding time window. (default=0)

    Returns:
        np.ndarray: Dynamic functional connectivity matrices
    """
    dfc_mat = fc_twin(np.load(X_path), win_sz, step_sz, sigma, normalize)
    if outdir is not None:
        np.save(os.path.join(outdir, "dfc_mat.npy"), dfc_mat)
        return
    return dfc_mat

def downsample_fc(X:np.ndarray, ind:np.ndarray, fill_diag=np.nan):
    """_summary_

    Args:
        X (np.ndarray): Square covariance matrix to downsample.
        ind (np.ndarray): 1D array containing the grouping you wish to downsample to. Must be equal length to the number of rows/columns in X.
        fill_diag (_type_, optional): Value to fill the main diagonal of the matrix X. Set to None to leave the matrix intact. Defaults to np.nan.

    Returns:
        np.ndarray: Downsampled covariance matrix.
    """
    if any(len(ind) != x for x in X.shape):
        raise ValueError("The length of `ind` must be equal to the number of rows/columns in the square matrix X.")
    
    if fill_diag is not None:
        np.fill_diagonal(X, val=fill_diag)
    
    n = len(np.unique(ind))
    X_ds = np.zeros((n,n))
    ind_ls = [which(ind==ii) for ii in np.unique(ind)]
        
    for count_1, ii_1 in enumerate(ind_ls):
        for count_2, ii_2 in enumerate(ind_ls):
            tmp_vals = X[ii_1,:][:,ii_2]
            X_ds[count_1, count_2] = np.nanmean(tmp_vals)
    if fill_diag is not None:
        np.fill_diagonal(X_ds, val=fill_diag)
    return X_ds

def dfc_para(input_path:str):
    """Parallelize the computation of the dynamic functional connectivity matrices on multiple sets of ROI time series.
    
    Each line of the input file should be formatted as:
    
    PATH=[path to ROI time series], WIN_SZ=[sliding time window width, in volumes], STEP_SZ=[step size, in volumes], OUTDIR=[output directory path], SIGMA=[sigma value for gaussian convolution]

    Args:
        input (str): Input file.
    """
    complete_order={}
    
    input_ff = inm.interpret_input(input_path, keywords=["PATH", "WIN_SZ", "STEP_SZ", "OUTDIR", "SIGMA"])
    nrow     = len(input_ff.index)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(dfc, input_ff.loc[row,:"PATH"], input_ff.loc[row,:"WIN_SZ"], input_ff.loc[row,:"STEP_SZ"], input_ff.loc[row,:"OUTDIR"], input_ff.loc[row,:"SIGMA"]): row for row in range(nrow)}
        
        for i, f in enumerate(concurrent.futures.as_completed(futures), start=0):
            subj = futures[f] # deal with async nature of submit
            print((f"subj idx: {subj}"))
            
            # count how many tasks are done (or just initialize a counter at the top to avoid looping)
            states = [i._state for i in futures]
            print(states)
            idx = states.count("FINISHED")
            complete_order[subj] = idx-1
            print(f"completed order: {complete_order}")
            print("")
    return

## identify exemplar time windows

def dfc_exemplarSD(X: np.ndarray, outdir:str=None, plot_peaks:bool=False):
    """Identify exemplar volumes in the dFC matrices by identifying local maxima in the STDs of the matrices across time.

    Args:
        X (np.ndarray): A 3D array of square FC matrices stacked along axis 2.
        outdir (str, optional): Directory to save results. Defaults to None.
        plot_peaks (bool, optional): Saves plot of STD values across time and labels local peaks. Requires an outdir. Defaults to False.

    Raises:
        Exception: X must be a 3D array.
        Exception: Each FC in X must be a square matrix.

    Returns:
        list, list: Returns STD values across time and the indices for the local maxima.
    """
    
    if len(X.shape)!=3:
        raise Exception("X must be a 3D array")
    if X.shape[0]!=X.shape[1]:
        raise Exception("Each FC mat in X must be a square matrix.")
    
    utri_ind  = np.triu_indices(X.shape[0], k=1)
    
    X_std = [np.std(X[:,:,ii][utri_ind]) for ii in range(X.shape[2])]
    X_std_locpeaks = argrelextrema(np.array(X_std), np.greater)[0]
    
    if outdir is not None:
        np.savetxt(os.path.join(outdir, "dfc_std.csv"), X_std         , delimiter=",")
        np.savetxt(os.path.join(outdir, "dfc_std_peaks.csv"), X_std_locpeaks.astype(int), fmt='%i', delimiter=",")
        
        if plot_peaks:
            plt_stds(X_std, X_std_locpeaks, os.path.join(outdir, "dfc_std.png"))
    
    return X_std, X_std_locpeaks

def plt_stds(stds, peaks, savepath:str=None):
    plt.plot(range(len(stds)), stds)
    plt.plot(peaks, stds[peaks], "ro")
    
    if savepath is not None:
        plt.savefig(savepath)
        return
    
    plt.show()
    return

def dfc_exemplarSD_para(input_file:str):
    complete_order = {}
    
    input_ff = inm.interpret_input(input_file, ["INPUT", "OUTPUT"])
    nrow     = len(input.index)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(dfc_exemplarSD, input_ff.loc[row, "INPUT"], input_ff.loc[row, "OUTPUT"]): row for row in range(nrow)}
        
        for i, f in enumerate(concurrent.futures.as_completed(futures), start=0):
            subj = futures[f]  # deal with async nature of submit
            print(f"subj idx: {subj}")
            print(subj)

            # count how many tasks are done (or just initialize a counter at the top to avoid looping)
            states = [i._state for i in futures]
            idx = states.count("FINISHED")
            complete_order[subj] = idx - 1
            print(f"completed order: {complete_order}")
            print("")
    
    return
