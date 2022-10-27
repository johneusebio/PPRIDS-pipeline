# ROI time series

import concurrent.futures
import os
import subprocess
import tempfile as temp

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, maskers

import input_import as inm


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

def get_peak(nii:str, mask:str=None, absolute:bool=False, native_space:bool=True):
    fslcmd = f"fslstats '{nii}'"
    if mask is not None:
        fslcmd = fslcmd+f" -k '{mask}'"
    if absolute:
        fslcmd = fslcmd+" -a"
    fslcmd = fslcmd+" -x"
    
    fslout = subprocess.Popen([fslcmd], shell=True, stdout=subprocess.PIPE)
    peak_coords =  fslout.stdout.read()
    peak_coords = [int(x) for x in peak_coords.split()]
    
    if native_space:
        x,y,z = peak_coords
        peak_coords = image.coord_transform(x,y,z, nib.load(nii).affine)
    return peak_coords

def roisphere_tseries(nii:nib.nifti1.Nifti1Image, seeds, mask=None, radius=None, allow_overlap=False, standardize=False, out:str=None): 
    msk = maskers.NiftiSpheresMasker(seeds=seeds, mask_img=mask, radius=radius, allow_overlap=allow_overlap, standardize=standardize)
    roi_tcourses = msk.fit_transform(nii)
    
    if out is not None:
        pd.DataFrame(roi_tcourses).to_csv(out, header=False, index=False)
    return roi_tcourses

def roimask_tseries(nii, roi):
    """Returns the mean time series within a given ROI mask

    Args:
        nii (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image
        roi (str | nib.nifti1.Nifti1Image | np.ndarray): nifti image of ROI mask in same space and dimensions as brain

    Returns:
        np.ndarray: mean time series of ROI
    """
    nii = get_niimat(nii)
    roi = get_niimat(roi)
    return np.nanmean(nii[roi==1], axis=0)

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

def atlas_tseries(nii_path, atlas_path, atlas_data=None, spm_path=None, roi_rad=None, outdir=None):
    """Returns time series of averaged voxels within each ROI of the provided atlas

    Args:
        nii (str | nib.nifti1.Nifti1Image | np.ndarray): brain nifti image, 4D
        atlas (str | nib.nifti1.Nifti1Image | np.ndarray): atlas nifti image, 3D
        output (str, optional): Filepath to save the time series to an npy file. Defaults to None.

    Returns:
        np.ndarray: Array of time series for each ROI. Not returned if an output path is provided.
    """
    try:
        # print("one")
        nii = nib.load(nii_path)
        # print("two")
        if atlas_data == "None":
            atlas_data = None
        if atlas_data is not None:
            # if the want to specify unique ROI order and/or exclude certain ROIs in the atlas
            atlas_pd = pd.read_csv(atlas_data, header=0)

            rois = atlas_pd.loc[:,"value"]
            roi_nms = atlas_pd.loc[:,"region"]
        else:
            # use the default order indicated by ROI values
            rois = roi_ls(get_niimat(atlas_path))
            roi_nms = rois

        # create empty table for ROI time series & roi coordinates
        tab_tseries = np.zeros((nii.shape[3], len(rois)))
        roi_coords = pd.DataFrame(columns=["roi_label", "x", "y", "z", "rad_mm"], 
                                index=range(len(rois)))

        tmpdir = temp.TemporaryDirectory()

        print(nii_path)
        for count, roi in enumerate(rois):
            print(f"-{count}:{roi}")
            roi_msk = image.math_img(f"img=={roi}", img=atlas_path)
            roi_msk.to_filename(os.path.join(tmpdir.name, f"roi{roi}.nii.gz"))
            if spm_path is not None:
                coords = get_peak(nii=spm_path, mask=os.path.join(tmpdir.name, f"roi{roi}.nii.gz"), absolute=True, native_space=True)
                tab_tseries[:,count] = roisphere_tseries(nii, seeds=[coords], radius=roi_rad)[:,0]
            else:
                tab_tseries[:,count] = roimask_tseries(nii, roi_msk)

        print("seven")
        tmpdir.cleanup()
        roi_tseries = pd.DataFrame(data=tab_tseries, columns=roi_nms)

        print("eight")
        if outdir is not None:
            np.save(os.path.join(outdir, "roi_tseries.csv"), roi_tseries)
            
            if atlas_data is not None:
                atlas_pd.to_csv(os.path.join(outdir, "atlas_data.csv"), index=False)
            if spm_path is not None:
                roi_coords.to_csv(os.path.join(outdir, "roi_coords.csv"), index=False)
        return roi_tseries
    except:
        print("error")
        return {"roi_count": count, "roi_ind": roi, "roi_list": rois}

def atlas_tseries_para(input_file:str, max_workers=None, trouble_shooting=False):
    complete_order={}
    
    input_ff = inm.interpret_input(input_file, keywords=["NIFTI", "ATLAS_PATH", "ATLAS_DATA", "SPM", "ROI_RAD", "OUTDIR"])
    nrow     = len(input_ff.index)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(atlas_tseries, 
                                   input_ff.loc[row,"NIFTI"], 
                                   input_ff.loc[row,"ATLAS_PATH"], 
                                   input_ff.loc[row,"ATLAS_DATA"], 
                                   input_ff.loc[row,"SPM"], 
                                   int(input_ff.loc[row,"ROI_RAD"]), 
                                   input_ff.loc[row,"OUTDIR"]): row for row in range(nrow)}
        
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
    if trouble_shooting:
        return futures
    return
