import os
import pickle
import decimal
import pathlib
import numpy as np
import pandas as pd
import nibabel as nib
import Preproc_subj as ppo
from time import sleep

# Config Defaults
default_config = { 
    "SKULLSTRIP" : "1",
    "SLICETIME"  : "1",
    "MOTCOR"     : "1",
    "NORM"       : "1",
    "SMOOTH"     : "6", # gaussian smoothing kernel size in mm
    "MOTREG"     : "1",
    "GSR"        : "1",
    "NUISANCE"   : "3",
    "TEMPLATE"   : "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz",
    "SCRUB"      : "UNION" # union, intersect, dvars, fd, none
}

step_order={
    "BASELINE"  :0,
    "SKULLSTRIP":1,
    "SLICETIME" :2,
    "MOTCOR"    :3,
    "NORM"      :4,
    "NUISANCE"  :5,
    "SCRUB"     :6,
    "SMOOTH"    :7
}

# Support Functions

def create_dirstruct(output):
    pathlib.Path(os.path.join(output, "func"           )).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, "motion"         )).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, "spat_norm"      )).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, "quality_control")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, "anat", "segment")).mkdir(parents=True, exist_ok=True)

    return

def check_exist(filepath, type="any"):
    # TODO: Simplify with branching `return` statements
    if type in ["file", "f"]:
        from os.path import isfile as exists
    elif type in ["dir", "d"]:
        from os.path import isdir as exists
    elif type=="any":
        from os.path import exists
    else:
        raise Exception("type '" + type + "' is invalid.")

    return(exists(filepath))
    
def parse_line(text, keywords, split="="):
    key, val = text.split(split)
    if key not in keywords:
        raise Exception("Input doesn't contain any of the specified key words")
    return(key, val)

def strip_chars(text, first, last):
    text = "{}".format(text[1:] if text.startswith(first) else text)
    text = "{}".format(text[:-1] if text.endswith(last) else text)
    return(text)

def parse_input(text, keywords, split="=", first="[", last="]"):
    key, val = parse_line(text, keywords, split)
    val = strip_chars(val, first, last)
    return(key, val)
    
def check_defaults(config_dict, defaults):
    default_keys=[key for key in defaults.keys()]

    restore_defaults=[key not in config_dict.keys() for key in default_keys]
    restore_defaults=list(filter(lambda x: restore_defaults[x], range(len(restore_defaults))))
    restore_defaults=[default_keys[i] for i in restore_defaults]

    for key in restore_defaults:
        config_dict[key] = defaults[key]

    return(config_dict)

def voxel_size(img, excl_time=True):
    nii = nib.load(img)
    vox_sz = nii.header.get_zooms()

    if excl_time:
        vox_sz=vox_sz[0:3]
    return(vox_sz)
    
def interpret_input_line(line, keywords):
    input_dict = {}
    items = line.split(",")
    
    for item in items:
        item=item.strip()
        try:
            key, val = parse_input(item, keywords) # keywords must be enterest as a list
            input_dict[key] = val
        except:
            print("Error:", item)

    # check if all files exist 
    if not check_exist(input_dict["FUNC"], "file"):
        raise Exception(input_dict["FUNC"] + " is not a valid filepath.")
    if not check_exist(input_dict["ANAT"], "file"):
        raise Exception(input_dict["ANAT"] + " is not a valid filepath.")
       
    return(input_dict)
    
def interpret_input(filepath):
    input = open(filepath, "r")
    lines = input.readlines()

    keywords = ["FUNC", "ANAT", "OUTPUT"]
    input_df = pd.DataFrame(index=range(len(lines)), columns=keywords)

    for line, ind in zip(lines, range(len(lines))):
        line=line.strip()
        input_dict=interpret_input_line(line, keywords)

        input_df.loc[ind,"FUNC"  ] = input_dict["FUNC"]
        input_df.loc[ind,"ANAT"  ] = input_dict["ANAT"]
        input_df.loc[ind,"OUTPUT"] = input_dict["OUTPUT"]

    return(input_df)
    
def interpret_config(filepath):
    config=open(filepath, "r")
    lines =config.readlines()
    config_dict = {}
        
    for line in lines:
        line=line.strip()
        try:
            key, val = parse_input(line, default_config.keys()) # keywords must be enterest as a list
            config_dict[key] = val
        except:
            print("Error:", line)

    config_dict = check_defaults(config_dict, default_config)
    return(config_dict)

def getTR(filepath):    
    img = nib.load(filepath)
    tr  = img.header.get_zooms()[3]
    return(str(tr)) 
    
def rm_ext(filename):
    filename=os.path.basename(filename)
    newpath =os.path.splitext(filename)[0]
    
    if "." in newpath:
        newpath=rm_ext(newpath)
    return(newpath)  
    
def get_ext(filename):
    filename    =os.path.basename(filename)
    newname, ext=os.path.splitext(filename)
    
    if "." in newname:
        ext = get_ext(newname) + ext
    return(ext)
    
def gauss_mm2sigma(mm):
    return mm/2.35482004503

def which(list, x=True):
    return [iter for iter, elem in enumerate(list) if elem == x]

def cp_rename(filepath, newpath):
    os.system("cp {} {}".format(filepath, newpath))
    return newpath

# Preprocessing Functions

def brainroi(img, out_dir):
    roi_img = os.path.join(out_dir, "roi_"+os.path.basename(img))
    os.system("robustfov -i {} -r {}".format(img, roi_img))

    return(roi_img)

def skullstrip(img, out_dir):
    out_file = "brain_" + os.path.basename(img)
    skullstr = os.path.join(out_dir, out_file)

    command="bet {} {} -R".format(img, skullstr)
    os.system(command)
    
    return(skullstr)

def slicetime(img, out_dir):
    tr=getTR(img)
    tshift_path=os.path.join(out_dir, "t_" + rm_ext(img)+".nii.gz")
    
    command="3dTshift -TR {}s -prefix {} {}".format(tr, tshift_path, img)
    os.system(command)

    return(tshift_path)

def motcor(img, func_dir, motion_dir):
    motcor_path =os.path.join(func_dir, "m_"+os.path.basename(img))
    _1dfile_path=os.path.join(motion_dir, "1d_"+rm_ext(os.path.basename(img))+".1D")

    command="3dvolreg -base 0 -prefix {} -1Dfile {} {}".format(motcor_path, _1dfile_path, img)
    os.system(command)

    return(motcor_path, _1dfile_path)
    
# new spatial normalization
def spatnorm(f_img, a_img, template, func_dir, anat_dir, norm_dir):
    # lin warp func to struct
    print("       + Linear-warping functional to structural...")
    l_func_omat=os.path.join(norm_dir, "func2str.mat")
    command="flirt -ref {} -in {} -omat {} -dof 6".format(a_img, f_img, l_func_omat)
    os.system(command)
    
    # lin warp struct to template
    print("       + Linear-warping structural to standard template...")
    l_anat_img =os.path.join(anat_dir, "l_" + os.path.basename(a_img))
    l_anat_omat=os.path.join(norm_dir, "aff_str2std.mat")
    command="flirt -ref {} -in {} -omat {} -out {}".format(template, a_img, l_anat_omat, l_anat_img)
    os.system(command)
    
    # non-lin warp struct to template
    print("       + Non-linear-warping structural to standard template...")
    nl_anat_img  =os.path.join(anat_dir, "n" + os.path.basename(l_anat_img))
    cout_anat_img=os.path.join(anat_dir, "cout_" + os.path.basename(nl_anat_img))
    command="fnirt --ref={} --in={} --aff={} --iout={} --cout={} --subsamp=2,2,2,1".format(template, a_img, l_anat_omat, nl_anat_img, cout_anat_img)
    os.system(command)
    
    # make binary mask from non-lin warped image
    print("       + Creating binary mask from non-linearly warped image...")
    bin_nl_anat_img=os.path.join(anat_dir, "bin_" + os.path.basename(nl_anat_img))
    command="fslmaths {} -bin {}".format(nl_anat_img, bin_nl_anat_img)
    os.system(command)
    
    # apply std warp to func data
    print("       + Applying standardized warp to functional data...")
    nl_func_img=os.path.join(func_dir, "nl_"+os.path.basename(f_img))
    command="applywarp --ref={} --in={} --out={} --warp={} --premat={}".format(template, f_img, nl_func_img, cout_anat_img, l_func_omat)
    os.system(command)

    # create tempalte mask
    print("       + Creating binary template mask...")
    mask_path=os.path.join(anat_dir, "mask_" + os.path.basename(template))
    command="fslmaths {} -bin {}".format(template, mask_path)
    os.system(command)

    return(nl_anat_img, nl_func_img, cout_anat_img, l_anat_omat) # anat, func, warp, premat

def applywarp(in_img, out_img, ref_img, warp_img, premat):
    os.system("fnirt --ref={} --in={} --aff={} --iout={} --cout={} --subsamp=2,2,2,1".format(ref_img, in_img, premat, out_img, warp_img))
    return(out_img)

def segment(img, out_dir):
    command="fast -n 3 -t 1 -o '{}' '{}'".format(os.path.join(out_dir,"seg"), img)
    os.system(command)

    return(os.path.join(out_dir, "seg_pve_0.nii.gz"))

def bin_mask(mask, thr=0.5):
    output = os.path.join(os.path.dirname(mask), "bin_"+os.path.basename(mask))
    command="fslmaths {} -thr {} -bin {}".format(mask, thr, output)
    os.system(command)
    
    return(output)

def roi_tcourse(img, mask, save_path):
    # compute the mean time course for ROI
    command="fslmeants -i '{}' -m '{}' -o '{}'".format(img, mask, save_path)
    os.system(command)
    
    return(save_path)
    
def spat_smooth(img, mm, out_dir): 
    sigma=gauss_mm2sigma(mm)
    s_img=os.path.join(out_dir, "s_"+os.path.basename(img))
    
    command="fslmaths {} -kernel gauss {} -fmean {}".format(img, sigma, s_img)
    os.system(command)
    
    return(s_img)

def combine_nuis(nuis1, nuis2, output):
    if nuis1 is None:
        nuis2_pd=pd.read_csv(nuis2, sep="\s+", header=None)
        nuis2_pd.to_csv(output, sep="\t", header=None, index=False)
        return(output)
    elif nuis2 is None:
        nuis1_pd=pd.read_csv(nuis1, sep="\s+", header=None)
        nuis1_pd.to_csv(output, sep="\t", header=None, index=False)
        return(output)

    nuis1_pd=pd.read_csv(nuis1, sep="\s+", header=None)
    nuis2_pd=pd.read_csv(nuis2, sep="\s+", header=None)

    nuis_cbind = pd.concat([nuis1_pd, nuis2_pd], axis=1)
    nuis_cbind.to_csv(output, sep="\t", header=None, index=False)

    return(output)
    
def nuis_reg(img, _1d, out_dir, pref="nuis", poly="1"):
    clean_img=os.path.join(out_dir, pref + "_" + rm_ext(img) + ".nii.gz")
    command="3dDeconvolve -input {} -ortvec  {} {} -polort {} -errts {}".format(img, _1d, pref, poly, clean_img)
    os.system(command)

    return(clean_img)

def file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def meantsBOLD(img, outdir, nomoco):
    dvars_txt = os.path.join(outdir, "dvars.1D")
    dvars_png = os.path.join(outdir, "dvars.png")

    out_compound = os.path.join(outdir, "dvars_outCols.1D")
    out_outliers = os.path.join(outdir, "dvars_outliers.1D")

    command="fsl_motion_outliers -i {} -o {} -s {} -p {} --dvars".format(img, out_compound, dvars_txt, dvars_png)
    if nomoco:
        command+=" --nomoco"

    os.system(command)

    sleep(10)
    if not os.path.exists(out_compound):
        nlines=file_len(dvars_txt)
        dvars_compound=np.zeros([nlines,1], dtype=int)
        dvars_compound=pd.DataFrame(dvars_compound)
        dvars_compound.to_csv(out_compound, sep="\t", index=False, header=False)

    dvars_out=pd.read_csv(out_compound, delim_whitespace=True, header=None)
    dvars_out=dvars_out.sum(axis=1)
    dvars_out=dvars_out.astype("int")
    dvars_out.to_csv(out_outliers, sep="\t", index=False, header=False)

    return(out_outliers)

def fd_out(img, voxel_size, outdir):
    thresh = voxel_size/2
    fd_txt = os.path.join(outdir, "fd.1D")
    fd_png = os.path.join(outdir, "fd.png")

    out_compound = os.path.join(outdir, "fd_outCols.1D")
    out_outliers = os.path.join(outdir, "fd_outliers.1D")

    command = "fsl_motion_outliers -i {} -o {} -s {} -p {} --fd --thresh={}".format(img, out_compound, fd_txt, fd_png, thresh)
    os.system(command)

    sleep(5)
    if not os.path.exists(out_compound):
        nlines=file_len(fd_txt)
        fd_compound=np.zeros([nlines,1], dtype=int)
        fd_compound=pd.DataFrame(fd_compound)
        fd_compound.to_csv(out_compound, sep="\t", index=False, header=False)

    fd_out=pd.read_csv(out_compound, delim_whitespace=True, header=None)
    fd_out=fd_out.sum(axis=1)
    fd_out=fd_out.astype("int")

    fd_out.to_csv(out_outliers, sep="\t", index=False, header=False)

    return(out_outliers)

def mk_outliers(dvars, fd, out_dir, method="UNION"):
    if fd is not None:
        fd_mat = pd.read_csv(fd, delimiter="\t", header=None)
    if dvars is not None:
        dvars_mat = pd.read_csv(dvars, delimiter="\t", header=None)
    
    if method=="UNION":
        outliers_mat = (fd_mat + dvars_mat).astype("bool")
        outliers_mat = outliers_mat.astype("int")
    elif method=="INTERSECT":
        outliers_mat = dvars_mat[0]*fd_mat[0]
        outliers_mat = outliers_mat.dropna()
    elif method=="FD":
        outliers_mat = fd_mat
    elif method=="DVARS":
        outliers_mat = dvars_mat
    
    outlier_path = os.path.join(out_dir, "scrub_outliers.txt")
    outliers_mat.to_csv(outlier_path, sep="\t", index=False, header=False)

    return(outlier_path)

def interp_time(img, out, int_ind):
    if len(img.shape)!=4:
        raise Exception("Nifti image must have 4 dimensions.")
    
    # make blank image
    int_dim = list(img.shape[0:3])
    int_dim.append(int_ind[1]-int_ind[0]-1)
    int_img = np.empty(int_dim)

    int_ind = np.setdiff1d(int_ind, [0, img.shape[3]]).tolist()
        
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for slice in range(img.shape[2]):
                if len(int_ind)==1:
                    int_val = img[row, col, slice, int_ind[0]]
                else:
                    int_val = interp_pts(img[row, col, slice, int_ind[0]], img[row, col, slice, int_ind[1]], int_ind[1]-int_ind[0]-1)
                int_img[row, col, slice, :] = int_val

    return(int_img)

def interp_pts(start, end, nvals, round=True):
    step_len =(end-start)/(nvals+1)
    interp_set=[start+(step_len*i) for i in range(1,nvals+1)]

    if round:
        interp_set=round_dec(interp_set, [start, end])

    return(interp_set)

def round_dec(vals, ref):
    n_places = max(n_dec(i) for i in ref)
    return [round(i, n_places) for i in vals]

def n_dec(val):
    return abs(decimal.Decimal(str(val)).as_tuple().exponent)

def surrounding_timepoints(t_range, ind):   
    sub_ind = np.empty((0,2), dtype=int, order='C')
    
    for ii in ind:
        t_start = ii-1
        t_end   = ii+1
        
        while t_start in ind:
            t_start -= 1
        while t_end in ind:
            t_end += 1
        
        if t_start < min(t_range):
            t_start = "NaN"
        if t_end < min(t_range):
            t_end = "NaN"
        
        sub_ind = np.vstack([sub_ind, (t_start, t_end)])
        sub_ind = np.unique(sub_ind, axis=0)
    return(sub_ind)

def scrubbing(nifti, outliers, out_dir, interpolate=False):
    img = nib.load(nifti)
    mat = img.get_fdata()

    outliers=pd.read_csv(outliers, delimiter="\t", header=None)
    outliers=outliers.values.tolist()
    outliers=which([o[0] for o in outliers], 1)

    print("       + Removing outlier timepoints...")
    mat[:,:,:,outliers]=float("NaN")

    if interpolate:
        print("       + Linearly-interpolating outliers...")
        sub_ind = surrounding_timepoints(range(mat.shape[3]), outliers)

        for oo in range(len(sub_ind)):
            tmp = interp_time(mat, outliers[oo], sub_ind[oo,:])
            mat[:,:,:,(sub_ind[oo,0]+1):sub_ind[oo,1]] = tmp
    
    # TODO: move bellow to a new function
    new_img = nib.Nifti1Image(mat, img.affine, header=img.header)
    print("       + Saving scrubbed timeseries...")
    scrub_path=os.path.join(out_dir, "scrub_"+rm_ext(os.path.basename(nifti))+".nii.gz")
    nib.save(new_img, scrub_path)
    
    return(scrub_path)

# Wrappers

def wrapper_lvl2(input_file, config_file):
    config = interpret_config(config_file)
    input  = interpret_input(input_file)
    nrow   = len(input.index)
    
    for row in range(nrow):
        subj_pp = ppo.Preproc_subj(input.loc[row,:], config, step_order)
        
        with open(os.path.join(subj_pp.out, "steps.p"), "wb") as fp:
            pickle.dump(subj_pp.steps, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(subj_pp.out, "config.p"), "wb") as fp:
            pickle.dump(subj_pp.config, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("")
        
    return