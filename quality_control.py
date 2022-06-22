import os
import matplotlib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt 
from pathlib import Path
from scipy.stats import pearsonr
from nilearn.image import iter_img
from nilearn import datasets, plotting
from nilearn.masking import apply_mask
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_stat_map
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

x = "Hi, I'm inside mod.py!"

if __name__ != "__main__":
    print("Using non-interactive matplotlib backend: Agg")
    matplotlib.use("Agg")  # non-interactive backend

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

def load_labels(csv_path):
    """
    Returns a list of provided in a csv file.
    :param csv_path: path to csv file (str)
    :return: list of labels
    """
    labels = np.genfromtxt(csv_path, delimiter=',', dtype="str", encoding=None)
    if labels.ndim != 1:
        raise Exception("labels must be a 1-D file (i.e., single-column)")
    return list(labels)

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

def t_series__mask(nii, mask=None):
    """
    Functions returns the mean time course within the specified binary nifti mask ROI.
    
    If no mask is provided, all 0 voxels are excluded and the raw mean time course for 
    the entire nifti file is returned.
    """
    if mask is not None:
        glob = apply_mask(nii, mask)
        return np.mean(glob, axis=1)
        
    nii[nii==0] = "nan"
    return np.nanmean(nii, axis=(0,1,2))

def t_series__atlas(nifti, atlas):
    """
    Function returns the time series for each ROI in the specified brain atlas.
    """
    
    atlas_filename = atlas["maps"]
    
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True)
    return masker.fit_transform(nifti)
    
def fc_mat(nifti, atlas):
    print("     four - one")
    roi_t = t_series__atlas(nifti, atlas)
    print("     four - two")
    correlation_measure = ConnectivityMeasure(kind="correlation")
    print("     four - three")
    return correlation_measure.fit_transform([roi_t])[0], roi_t

def euc_dist(a,b):
    """
    Returns the euclidean distiance between two points in n-dimensional space.
    """
    return np.linalg.norm(a-b)

def fcmat__plot(mat, atlas_labels, savepath):
    """
    Plot and save the FC matrix between ROIs.
    """
    
    fig = plt.figure(figsize=(10,10))
    np.fill_diagonal(mat, 0)
    plotting.plot_matrix(mat, labels=atlas_labels, colorbar=True, vmax=0.8, vmin=-0.8, figure=fig)
    fig.savefig(savepath)
    return

def connectome__plot(mat, region_coords, savepath):
    """
    Plot and save the connectome plot between ROIs.
    """
    fig = plt.figure(figsize=(10,10))
    plotting.plot_connectome(mat, region_coords, edge_threshold="80%", colorbar=True, figure=fig)
    fig.savefig(savepath)
    return
    
def rXdist__plot(r, dist, savepath):
    """
    Plot the correlation between each pair of ROIs as a function of the 
    euclidean distance between them.
    """
    
    fig = plt.figure(figsize=(20,10))
    ax  = fig.add_axes([0.05,0.05,.9,.9])
    ax.scatter(dist, r, color="black")
    ax.axhline(y=0, color="r", linestyle="dashed")
    ax.set_xlabel("Euclidean Distance Between ROIs (vox)")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("FC as a Function of Euclidean Distance Between ROIs")
    fig.savefig(savepath)
    return

def elem_uppertriangle(X):
    if X.ndim != 2:
        raise Exception("X must be a 2D numpy array.")
    dims = X.shape
    return int(dims[0]*(dims[1]-1)/2)

def rXdist(X, coords):
    """
    Compute the correlation and euclidean distance between each pair of ROIs.
    """
    n_uppertriangle = elem_uppertriangle(X)
    
    r_tab    = np.zeros([n_uppertriangle])   # correlation (r) b/w rois A & B
    a_tab    = np.zeros([n_uppertriangle,3]) # coordinates of roi A (rows)
    b_tab    = np.zeros([n_uppertriangle,3]) # coorindates of roi B (columns)
    dist_tab = np.zeros([n_uppertriangle])   # Euclidean dist b/w rois A & B
    
    dims = X.shape
    count=0
    for rr in range(dims[0]):
        for cc in range(dims[1]):
            if rr >= cc:
                continue
            r_tab   [count]  = X[rr,cc]
            a_tab   [count,] = list(coords[rr])
            b_tab   [count,] = list(coords[cc])
            dist_tab[count]  = abs(euc_dist(a_tab[count,], b_tab[count,]))
            
            count+=1
    return r_tab, a_tab, b_tab, dist_tab

def rXdist__diff(X2, X1, coords):
    r_tab1, a_tab1, b_tab1, dist_tab1 = rXdist(X1, coords)
    r_tab2,      _,      _,         _ = rXdist(X2, coords)
    
    r_tab_diff = r_tab2 - r_tab1
    return r_tab_diff, a_tab1, b_tab1, dist_tab1

####### Motion parameter correlation

def signalXmot__plot(data, labels=None, xlabel=None, ylabel=None, title=None):
    """
    Create and save a bar plot for QC metrics.
    """
    if labels is None:
        labels = [str(x+1) for x in range(len(data))]
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,.9,.8])
    ax.bar(labels, data)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axhline(y=0, color="black")
    return fig

def import_motion(filepath, names=None):
    """
    Import the motion parameters.
    """
    # TODO: set names=0 to use the first line of the text file
    if names is None:
        return pd.read_csv(filepath, header=None, sep=r"\s+")
    return pd.read_csv(filepath, names=names, sep=r"\s+")

def nifti_getmat(nifti):
    """
    Import nifti file and returns the 3- or 4-d matrix data.
    :param nifti: full path to nifti file
    :return: nfiti matrix data
    """
    img = nib.load(nifti)
    return img.get_fdata()

def cor_motxsignal(mot_params, nifti, mask=None, motparam_names=None):
    """
    Plot the correlation between motion parameters and neural signal.
    """
    
    mot_params   = mot_params.to_numpy()
    glob_tcourse = t_series__mask(nifti, mask)
    
    cor_mat = np.zeros(mot_params.shape[1])
    for param in range(mot_params.shape[1]):
        cor_mat[param], _ = pearsonr(glob_tcourse, mot_params[:,param])
    
    return cor_mat

def ica(img, output, n_ic=4, mask=None, name=None, memory=None):
    if name is None:
        name = rm_ext(os.path.basename(img))
    ica_img = os.path.join(output, "IC_"+name+".nii.gz")

    canica=CanICA(n_components=n_ic, smoothing_fwhm=0, memory=memory, mask=mask, random_state=0, n_jobs=1, n_init=4)
    canica.fit(img)
    canica.components_img_.to_filename(ica_img)
    
    ica_fig_ls = []

    for i, cur_img in enumerate(iter_img(canica.components_img_)):
        ica_jpg = os.path.join(output, "IC"+str(i)+"_"+name+".jpg")
        plot_stat_map(cur_img, display_mode="ortho", title="IC %d" % i, colorbar=True, output_file=ica_jpg)
        ica_fig_ls.append(ica_jpg)

# support functions

def qclvl1__mkdir(path):
    path_plot     = Path(os.path.join(path, "plots"))
    path_plot_ica = Path(os.path.join(path, "plots", "ICA"))
    path_metric   = Path(os.path.join(path, "metrics"))
    
    path_plot.mkdir(parents=True, exist_ok=True)
    path_metric.mkdir(parents=True, exist_ok=True)
    path_plot_ica.mkdir(parents=True, exist_ok=True)
    return path_metric, path_plot, path_plot_ica
    
def rm_ext(filename):
    filename=os.path.basename(filename)
    newpath =os.path.splitext(filename)[0]
    
    if "." in newpath:
        newpath=rm_ext(newpath)
    return(newpath)  

# main wrapper

def qclvl1(nifti_pp, nifti_ref, mot_params, qc_dir, n_ic=4, mask=None, ica_memory=None, atlas=None, atlas_labels=None, atlas_rois=None):
    # create QC directories
    dir_metrics, dir_plots, dir_ica = qclvl1__mkdir(qc_dir)
    print("one")

    # load nifti files
    nifti_pp  = nib.load(nifti_pp)
    nifti_ref = nib.load(nifti_ref)
    print("two")
    
    # load the atlas data
    atlas_data = load_atlas(atlas=atlas, labels=atlas_labels, rois=atlas_rois)
    print("three")
    
    # load motion
    mot_estim = import_motion(mot_params)
    print("four")
    
    # between-roi correlation matrices
    rmat_pp, roiT_pp   = fc_mat(nifti_pp , atlas_data)
    rmat_ref, roiT_ref = fc_mat(nifti_ref, atlas_data)
    print("five")
        
    # get data to generate connectome and r-by-distance plots
    rtab_diff, _, _, disttab = rXdist__diff(rmat_pp, rmat_ref, coords=atlas_data.region_coords)
    print("six")
    
    # correlation between motion and global signal
    signalXmot_pp  = cor_motxsignal(mot_estim, nifti_pp , mask=mask)
    signalXmot_ref = cor_motxsignal(mot_estim, nifti_ref, mask=mask)
    print("seven")
    
    # plotting
    
    ## plot connectome
    connectome__plot(rmat_pp, atlas_data.region_coords, os.path.join(dir_plots, "roi_connectome.jpg"))
    print("eight")
    
    ## plot FC matrix
    fcmat__plot(rmat_pp, atlas_data.labels, os.path.join(dir_plots, "fc_matrix_preprocessed.jpg"))
    fcmat__plot(rmat_ref, atlas_data.labels, os.path.join(dir_plots, "fc_matrix_reference.jpg"))
    print("nine")
    
    ## plot r-by-distance
    rXdist__plot(rtab_diff, disttab, os.path.join(dir_plots, "QC_rXdist.jpg"))
    print("ten")
    
    ## plot correlation with motion parameters
    # ax1 = fig.add_subplot(1,2,1) # preprocessed image
    signalXmot__plot(signalXmot_pp)
    print("eleven")
    plt.savefig(os.path.join(dir_plots, "motionCorrelation_preprocessed.jpg"))
    print("twelve")
    # ax2 = plt.subplot(1,2,2) # reference image
    signalXmot__plot(signalXmot_ref)
    print("thirteen")
    plt.savefig(os.path.join(dir_plots, "motionCorrelation_reference.jpg"))
    print("fourteen")
    
    ## ICA
    ica(nifti_pp , dir_ica, n_ic, nib.load(mask), name="preprocessed", memory=ica_memory)
    print("fifteen")
    ica(nifti_ref, dir_ica, n_ic, nib.load(mask), name="reference", memory=ica_memory)
    print("sixteen")
    
    # save the data to csv files
    np.savetxt(os.path.join(dir_metrics, "rXdist.csv"), np.transpose(np.vstack((disttab, rtab_diff))))
    print("seventeen")
    
    np.savetxt(os.path.join(dir_metrics, "roi_FC_preprocessed.csv"), rmat_pp)
    np.savetxt(os.path.join(dir_metrics, "roi_FC_reference.csv"   ), rmat_ref)
    print("eighteen")
    
    np.savetxt(os.path.join(dir_metrics, "roi_tcourse_preprocessed.csv"), roiT_pp)
    np.savetxt(os.path.join(dir_metrics, "roi_tcourse_reference.csv"   ), roiT_ref)
    print("ninteteen")
    
    np.savetxt(os.path.join(dir_metrics, "motionCorrelation_preprocessed.csv"), signalXmot_pp)
    np.savetxt(os.path.join(dir_metrics, "motionCorrelation_reference.csv"   ), signalXmot_ref)
    print("twenty")
    
    return "FINISHED QUALITY CONTROL"