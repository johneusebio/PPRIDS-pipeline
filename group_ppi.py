# importing packages
import concurrent.futures
import os
import pathlib

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nipype.interfaces.fsl import \
    TemporalFilter  # temporal filtering of the time series
from scipy.stats import dgamma, gamma, zscore

# importing own functions
import input_import as inm

# from sklearn.linear_model import LinearRegression


## Miscellaneous

def get_center(X:np.ndarray):
    """Find the center of a numerical list

    Args:
        X (np.ndarray): A 1D numerical vector
    """
    return((X.min() + X.max()) /2)

def makezero(vector: np.ndarray, center=np.mean):
    """Centers the provided vector to 0.as_integer_ratio

    Args:
        vector (list): The vector to center.
        mid (str): The method used to center the vector. "min" = set the minimum value of the vector to 0. "center" = set the half-way point between the minimum and maximum to 0. "mean" = set the mean of the vector to 0.

    Raises:
        Exception: "'mid' must be one of the following: ['min', 'center', 'mean']"

    Returns:
        [list]: [The 0-centered vector.]
    """
    
    return vector - center(vector)

def is_multiple(arr: list, base: int or float=1):
    """Determine if numbers are a multiple of a given base value.

    Args:
        arr (list): List of numeric values
        base (int or float, optional): Base value to test list elements against. Defaults to 1.

    Returns:
        [type]: List of logical values indicating if elements in the list are a multiple of the given base.
    """
    if not isinstance(arr, list):
        arr = [arr]
    return [x % base == 0 for x in arr]

def ons2trs(design: np.ndarray, tr: int, nvol: int):
    """Convert onsets to TR events (list of 1s and 0s; ON and OFF). Not that onsets and TRs must be given in the same units of time.

    Args:
        design (np.ndarray): design matrix of event onsets.
        tr (int): Duration of TRs (seconds)
        nvol (int): Number of volumes int the scan

    Returns:
        [type]: Returns list of TR events (list of 1s and 0s; ON and OFF)
    """
    psy_ons = np.zeros((nvol, 1))
    ons = design[:,0]/tr

    if not all(is_multiple(design[:,0], tr)) or not all(is_multiple(design[:,1], tr)):
        raise ValueError("Condition onset and duration must be multiples of TR length")
    if not all(is_multiple(design[:,2], 2)):
        raise ValueError("")

    for count, vol0 in enumerate(ons):
        for voli in range(design[count,2]):
            psy_ons[vol0+voli] = design[count, 1]
    return psy_ons

def get_field(X: np.ndarray, field: str):
    """Extract the values of a field contained in a matrix of dictionary or dictionary-like objects.

    Args:
        X (np.ndarray): A matrix where each element contains a series of dictionaries or dictionary-like objects you wish to extract data from.
        field (str): The field you want to extract the data from.

    Returns:
        np.ndarray: Matrix containing the values within the field of interest. Is the same dimensions as results.
    """
    try:
        # n_elem = len(getattr(X[0,0,0], field))
        n_elem = len(X[0,0,0].get(field))
    except:
        n_elem = 1 # to handle isntances where indexing a number rather than a list
    out_shape=list(X.shape)
    out_shape.append(n_elem)
    
    out_arr = np.empty(tuple(out_shape))
    
    for rr in range(n_elem):
        try:
            # out_arr[:,:,:,rr]=np.vectorize(lambda i: getattr(i, field)[rr])(X)
            out_arr[:,:,:,rr]=np.vectorize(lambda i: i.get(field)[rr])(X)
        except:
            # out_arr[:,:,:,rr]=np.vectorize(lambda i: [getattr(i, field)][rr])(X)
            out_arr[:,:,:,rr]=np.vectorize(lambda i: [i.get(field)][rr])(X)
    return out_arr

def hz2sigma(Hz_thr: int, TR: int or float = 1):
    """AI is creating summary for hz2sigma

    Args:
        Hz_thr (int): Frequency (Hz)
        TR (intorfloat, optional): TR duration, in seconds. Leave this at 1 to get sigma in seconds. Defaults to 1.

    Returns:
        float: sigma
    """    

    # sigma = ((1/Hz_thr) / sqrt(8*log(2))) / TR

    return 1/(2*TR*Hz_thr)

# Working with nifti images

def roi_tcourse(nii: np.ndarray, mask: np.ndarray):
    """Extract BOLD timecourse of given ROI

    Args:
        nii (np.ndarray): 4D matrix of nifti image
        mask (np.ndarray): 3D matrix of ROI. Must match the dimensions of the nii matrix.

    Returns:
        [type]: vector of ROI timecourse
    """
    masked = nii[mask==1]
    return np.nanmean(masked, axis=0)

def save_nii(arr: np.ndarray, savepath: str, hdr: nib.nifti1.Nifti1Header=None, aff=None, update_dims: bool=False):
    """Save an np.ndarray as a .nii file.

    Args:
        arr (np.ndarray): The array to be output
        savepath (str): The filepath you wish to save your array to
        hdr (nib.nifti1.Nifti1Header, optional): The header you wish to use for the nifti file. Defaults to None.
        update_dims (bool, optional): Would you like to update the provided hdr to match the arr dimensions?. Defaults to False.

    Returns:
        [str]: The filepath you saved the array to.
    """
    if update_dims:
        hdr.set_data_shape(arr.shape)
    img = nib.Nifti1Image(arr, header=hdr, affine=aff)
    nib.save(img, savepath) 
    return savepath

# temporal filtering

def __filt_1d_singlepass(sample_rate:float, freq:float or int, tpoints:int):
    return int(np.round(freq / sample_rate * tpoints))

def __filt_mkF(timepoints, lowidx, highidx):
    F = np.zeros((timepoints))
    F[highidx:lowidx] = 1
    return ((F+F[::-1]) > 0).astype(int) #F[::-1] reverses the list

def bp_filt__1D(X, TR, BP_freq):
    # source: https://neurostars.org/t/bandpass-filtering-different-outputs-from-fsl-and-nipype-custom-function/824
    LP_freq, HP_freq = BP_freq

    sampling_rate = 1./TR
    timepoints = len(X)
    
    # defaults
    highidx = 0
    lowidx  = timepoints // 2 + 1 # "/" replaced by "//"
    
    if LP_freq > 0:
        lowidx = __filt_1d_singlepass(freq=LP_freq, sample_rate=sampling_rate, tpoints=timepoints)
    if HP_freq > 0:
        highidx = __filt_1d_singlepass(freq=HP_freq, sample_rate=sampling_rate, tpoints=timepoints)

    F = __filt_mkF((timepoints), lowidx, highidx)

    if np.all(F==1):
        return  X
    return np.real(np.fft.ifftn(np.fft.fftn(X) * F))

def temporal_filtering__unused(img: str, out_dir: str, TR: int or float, hp_hz: int or float=0.01, lp_hz: int or float=-1):
    # set the filter cutoffs to negative values to skip
    out = os.path.join(out_dir, f"bp_{os.path.basename(img)}")
    TF = TemporalFilter(in_file=img, out_file=out, highpass_sigma=hz2sigma(hp_hz, TR), lowpass_sigma=hz2sigma(lp_hz, TR))
    TF.run()
    return out

# HRF

def eventsAsIntegers(task, nTR):
    time_course = np.zeros(nTR)
    for onset, duration, amplitude in task:
        # Make onset and duration integers
        onset = int(round(onset))
        duration = int(round(duration))
        time_course[onset:onset + duration] = amplitude
    return time_course

def events2neural(task_fname, tr, n_trs):
    """ Return predicted neural time course from event file `task_fname`
    This is from the stimuli package (saved in my OneDrive)

    Parameters
    ----------
    task_fname : str
        Filename of event file
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run

    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    task = np.loadtxt(task_fname)
    # Check that the file is plausibly a task file
    if task.ndim != 2 or task.shape[1] != 3:
        raise ValueError(f"Incorrect number of dimensions. Is '{task_fname}' a task file?")
    # Convert onset, duration seconds to TRs
    task[:, :2] = task[:, :2] / tr
    # Neural time course from onset, duration, amplitude for each event
    return eventsAsIntegers(task=task, nTR=n_trs)

def design2psy(design: np.ndarray, tr: int or float, nvol: int, hrf_func: str):
    """Generate predicted neural time-course from a design matrix

    Args:
        design (np.ndarray): Design matrix
        tr (int or float): TR duration (seconds)
        nvol (int): Number of volumes in the task run
        hrf_func (str): HRF to use ['gamma', 'dgamma']

    Returns:
        [np.ndarray]: predicted BOLD time course
    """
    hrf_f = hrf_curve(tr=tr, hrf_func=hrf_func)
    neural_prediction = events2neural(task_fname=design, tr=tr, n_trs=nvol)
    return convolve_hrf(neural_prediction, hrf_f)

def hrf_curve(tr: int, hrf_func: str):
    """Downsample HRF to TR frequency

    Args:
        tr (int): duration of TR (seconds)
        hrf_func (str): the HRF function to use ['gamma', 'dgamma']

    Returns:
        [type]: Downsampled HRF timecourse.
    """
    times = np.arange(0, 30, tr)
    return hrf(times, hrf_func)

def convolve_hrf(neural_prediction: list, hrf_tr: list):
    """Convolve HRF with the predicted neural timecourse

    Args:
        neural_prediction (list): The neural prediction output from events2neural().
        hrf_f (list): Downsampled HRF timecourse

    Raises:
        ValueError: "len(convolved) must be equal to (len(neural_prediction) + len(hrf_f) -1)"

    Returns:
        [type]: The predicted BOLD response, given the neural prediction and the HRF.
    """
    convolved = np.convolve(neural_prediction, hrf_tr)
    if len(convolved) != (len(neural_prediction) + len(hrf_tr) -1):
        raise ValueError("len(convolved) must be equal to (len(neural_prediction) + len(hrf_f) -1)")
    n_to_rm = len(hrf_tr) - 1
    return convolved[:-n_to_rm]

def hrf(times: list, hrf_func: str="dgamma"):
    """Return values for HRF at given times

    Args:
        times (list): The timecourse (in seconds) of TRs taken throughout the scan.
        hrf_func (str, optional): The HRF to use. Can be "gamma" or "dgamma" (double-gamma). Defaults to "dgamma".

    Raises:
        ValueError: "The provided response function (func) must be either 'gamma' or dgamma' (default)"

    Returns:
        [type]: The HRF, downsampled to match the TR frequency.
    """
    if hrf_func == "dgamma":
        peak_values = dgamma.pdf(times, 6) # dgamma pdf for the peak
        undershoot_values = dgamma.pdf(times, 12) # dgamma pdf for the undershoot
    elif hrf_func == "gamma":
        peak_values = gamma.pdf(times, 6) # gamma pdf for the peak
        undershoot_values = gamma.pdf(times, 12) # gamma pdf for the undershoot
    else:
        raise ValueError("The provided response function (func) must be either 'gamma' or dgamma' (default)")
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

# PPI

def norm_range(x: list):
    """Normalize vector of numbers to a range of (-1,1)

    Args:
        x (list): Vector of numbers to normalize

    Returns:
        [type]: Normalized vector with range of (-1,1)
    """
    # normalize values to a range of (-1, 1)
    return 2.*(x - np.min(x))/np.ptp(x)-1

def glm__add_intercept(X:np.ndarray):
    return np.c_[np.ones(X.shape[0]), X]

def glm_fit(y: np.ndarray, X: np.ndarray):
    ols = sm.OLS(y, glm__add_intercept(X))
    ols_results = ols.fit()
    return {"tvalues": ols_results.tvalues, "se": ols_results.bse}

def ppi_regressor(psych: list, phys: list):
    """Generate psychophysiological (PPI) regressor

    Args:
        psych (list): Psychological regressor
        phys (list): Physiological regressor

    Returns:
        [type]: PPI regressor
    """
    return makezero(psych, get_center) * makezero(phys, np.mean)

def ppi_bpfilter(preds, TR, BP_regs, BP_freq):
    for ii, reg in enumerate(BP_regs):
        if not reg:
            continue
        preds[:,ii] = bp_filt__1D(preds[:,ii], TR, BP_freq)
    return preds

def ppi_predictors(nii, roi, design, TR, ntps, hrf_func, BP_regs, BP_freq):
    psy   = design2psy(design=design, tr=TR, nvol=ntps, hrf_func=hrf_func) # psy
    psy_d = np.gradient(psy)                                               # psy derivative
    phys  = roi_tcourse(nii=nii.get_fdata(), mask=roi.get_fdata())                                 # phys
    ppi   = ppi_regressor(psych=psy, phys=phys)                            # ppi
    
    # assemble them into an array
    preds = np.column_stack((psy, psy_d, phys, ppi))
    return ppi_bpfilter(preds=preds, TR=TR, BP_regs=BP_regs, BP_freq=BP_freq)

def save_ppi(ppi_obj, out, hdr, aff):
    save_nii(get_field(ppi_obj, "tvalues"), os.path.join(out, "ppi_tstat.nii.gz"), hdr=hdr, aff=aff)
    save_nii(get_field(ppi_obj, "se"     ), os.path.join(out, "ppi_se.nii.gz"   ), hdr=hdr, aff=aff)
    return

def save_design(design_mat, out, header=""):
    out_path = os.path.join(out, "design.txt")
    np.savetxt(out_path, design_mat, delimiter=", ", header=header)
    return

def run_ppi(nii_path:str, roi_path:str, design: str, out:str=None, 
            hrf_func:str="dgamma", BP_hz:list=[-1,-1], BP_regs:list=[1,0,0,0]):
    if out is not None:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)

    nii_obj = nib.load(nii_path)
    roi_obj = nib.load(roi_path)

    # creating predictors
    predictors = ppi_predictors(nii=nii_obj, roi=roi_obj, 
                                design=design, TR=nii_obj.header.get_zooms()[-1], ntps=nii_obj.shape[3], 
                                hrf_func=hrf_func, BP_regs=BP_regs, BP_freq=BP_hz)

    # run GLM for PPI analysis
    fit = np.apply_along_axis(func1d=glm_fit, axis=3, arr=nii_obj.get_fdata(), X=predictors)
    # return fit
    if out is None:
        return fit, predictors

    # add baseline to the design matrix
    predictors = np.c_[np.zeros(predictors.shape[0]), predictors]

    save_ppi(fit, out=out, hdr=nii_obj.header, aff=nii_obj.affine)
    save_design(predictors, out=out, header="BASE, PSY, dPSY, PHYS, PPI")
    return

# Parallelization

def __parallel_idxComplete(futures, i):
    states = [i._state for i in futures]
    print(states)
    print(i)
    return states.count("FINISHED")

def __parallel_async(futures, f):
    subj = futures[f]  # deal with async nature of submit
    print(f"subj idx: {subj}")
    print(subj)
    
    return subj

def ppi_parallel(input_dict_file:str, hrf_func:str="dgamma",
                 BP_hz:list=[-1,-1], BP_regs: list=[1,0,0,0],
                 max_workers:int=None, debug:bool=False):
    # sourcery skip: default-mutable-arg

    input_dict  = inm.interpret_input(input_dict_file, ["FUNC", "ROI", "TASK", "OUTPUT"])
    
    complete_order = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_ppi,
                                   nii_path=input_dict.FUNC[row], roi_path=input_dict.ROI[row],
                                   design=input_dict.TASK[row], out=input_dict.OUTPUT[row],
                                   hrf_func=hrf_func, BP_hz=BP_hz, BP_regs=BP_regs): row for row in range(len(input_dict.index))}

        for i, f in enumerate(concurrent.futures.as_completed(futures), start=0):
            # count how many tasks are done (or just initialize a counter at the top to avoid looping)
            subj = __parallel_async(futures, f)
            idx  = __parallel_idxComplete(futures, i)
            complete_order[f"line {subj}"] = idx - 1
    if debug:
        return list(futures), complete_order
    else:
        return

## whole-brain PPI matrix

def t2r(t, df):
    """Convert a t-score into a Person's correlation r-score

    Args:
        t (float): t-score
        df (float): degrees of freedom

    Returns:
        float: Pearson's correlation r score
    """
    return np.sqrt(t**2/(t**2 + df))

def r2z(r):
    """Convert Pearsons correlation r-score to a z-score

    Args:
        r (float): Pearson's correlation r-score

    Returns:
        float: z-score
    """
    return 0.5*(np.log(1+r) - np.log(1-r))

def z2r(z):
    """Convert z-score to Pearson's correlation r-score

    Args:
        z (float): z-score

    Returns:
        float: Pearson's correlation r-score
    """
    return np.tanh(z)

def txt2list(filepath:str):
    """Generate a list containing lines of a given text file.

    Args:
        filename (str): filepath

    Returns:
        list: list of lines from tect file
    """
    with open(filepath) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

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

def roi_mean(nii, roi):
    """Returns mean value within a given roi mask

    Args:
        nii (str): path to brain image (3D or 4D)
        roi (str): path to ROI mask with the same dimensions as the brain image (nii)

    Returns:
        np.ndarray: array of mean ROI value(s)
    """
    nii = get_niimat(nii)
    roi = get_niimat(roi)
    return np.nanmean(nii[roi==1])

def atlas2roi(atlas, roi:int):
    """Extract specified ROI from the provided atlas.

    Args:
        atlas (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image or brain atlas
        roi (int): value of ROI within the atlas image

    Returns:
        np.ndarray: ROI mask indices
    """
    atlas = get_niimat(atlas)
    return np.where(atlas==roi, 1, 0)

def atlas_mean(nii, atlas):
    """Returns dictionary containing mean values within each ROI of the provided atlas image

    Args:
        nii (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image
        atlas (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image

    Returns:
        dict: Dictionary containing the ROI IDs ("rois") and the ROI means ("means")
    """
    atlas = get_niimat(atlas)
    nii   = get_niimat(nii)
    rois  = roi_ls(atlas)
    
    return {"rois":rois, "means":[roi_mean(nii, atlas2roi(atlas, roi)) for roi in rois]}

def roi_ls(nii):
    """Returns list of ROIs within a provided brain atlas

    Args:
        nii (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image of atlas

    Returns:
        list: list of ROI values within the provided brain atlas
    """
    nii = get_niimat(nii)
    rois = np.unique(nii)
    return rois[rois!=0]

def whole_ppi2rmat(input_dict:str, atlas, rois:list=None, as_r=True, output:str=None):
    """
    Convert a series of PPI analyses into an asymmetrical whole-brain connectivity matrix.

    Args:
        input_dict (str): text file where each line contains the file path of a 3D PPI output (t-map)
        atlas (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image of atlas
        rois (list, optional): List of ROIs to use. Defaults using all ROIs in the provided atlas (None).
        as_r (bool, optional): Output the matrix using Pearson's correlation r-scores. Defaults to True.

    Returns:
        [type]: [description]
    """
    
    nii_ls = txt2list(input_dict)
    if rois is None:
        rois = roi_ls(get_niimat(atlas ))
    
    covmat = np.empty((len(rois), len(rois)))
    for count, (nii, roi) in enumerate(zip(nii_ls, rois)):
        row           = atlas_mean(nii, atlas)["means"]
        row[count]    = 0
        covmat[count,:] = row
    if as_r:
        covmat = z2r(covmat)
    if output is not None:
        np.savetxt(output, covmat, delimiter=",")
        return
    return covmat
