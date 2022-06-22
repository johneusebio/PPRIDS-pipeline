def qclvl1(nifti_pp, nifti_ref, mot_params, qc_dir, n_ic=4, mask=None, atlas=None, atlas_labels=None, atlas_rois=None):
    # create QC directories
    dir_metrics, dir_plots, dir_ica = qclvl1__mkdir(qc_dir)
    
    # load nifti files
    nifti_pp  = nib.load(nifti_pp)
    nifti_ref = nib.load(nifti_ref)
    
    # load the atlas data
    atlas_data = load_atlas(atlas=atlas, labels=atlas_labels, rois=atlas_rois)
    
    # load motion
    mot_estim = import_motion(mot_params)
    
    # between-roi correlation matrices
    rmat_pp, roiT_pp   = fc_mat(nifti_pp , atlas_data)
    rmat_ref, roiT_ref = fc_mat(nifti_ref, atlas_data)
        
    # get data to generate connectome and r-by-distance plots
    rtab_diff, _, _, disttab = rXdist__diff(rmat_pp, rmat_ref, coords=atlas_data.region_coords)
    
    # correlation between motion and global signal
    signalXmot_pp  = cor_motxsignal(mot_estim, nifti_pp , mask=mask)
    signalXmot_ref = cor_motxsignal(mot_estim, nifti_ref, mask=mask)
    
    # plotting
    
    ## plot connectome
    connectome__plot(rmat_pp, atlas_data.region_coords, os.path.join(dir_plots, "roi_connectome.jpg"))
    
    ## plot FC matrix
    fcmat__plot(rmat_pp, atlas_data.labels, os.path.join(dir_plots, "fc_matrix_preprocessed.jpg"))
    fcmat__plot(rmat_ref, atlas_data.labels, os.path.join(dir_plots, "fc_matrix_reference.jpg"))
    
    ## plot r-by-distance
    rXdist__plot(rtab_diff, disttab, os.path.join(dir_plots, "QC_rXdist.jpg"))
    
    ## plot correlation with motion parameters
    # ax1 = fig.add_subplot(1,2,1) # preprocessed image
    signalXmot__plot(signalXmot_pp)
    plt.savefig(os.path.join(dir_plots, "motionCorrelation_preprocessed.jpg"))
    # ax2 = plt.subplot(1,2,2) # reference image
    signalXmot__plot(signalXmot_ref)
    plt.savefig(os.path.join(dir_plots, "motionCorrelation_reference.jpg"))
    
    ## ICA
    ica(nifti_pp , dir_ica, n_ic, mask, name="preprocessed")
    ica(nifti_ref, dir_ica, n_ic, mask, name="reference")
    
    # save the data to csv files
    np.savetxt(os.path.join(dir_metrics, "rXdist.csv"), np.transpose(np.vstack((disttab, rtab_diff))))
    
    np.savetxt(os.path.join(dir_metrics, "roi_FC_preprocessed.csv"), rmat_pp)
    np.savetxt(os.path.join(dir_metrics, "roi_FC_reference.csv"   ), rmat_ref)
    
    np.savetxt(os.path.join(dir_metrics, "roi_tcourse_preprocessed.csv"), roiT_pp)
    np.savetxt(os.path.join(dir_metrics, "roi_tcourse_reference.csv"   ), roiT_ref)
    
    np.savetxt(os.path.join(dir_metrics, "motionCorrelation_preprocessed.csv"), signalXmot_pp)
    np.savetxt(os.path.join(dir_metrics, "motionCorrelation_reference.csv"   ), signalXmot_ref)
    
    return