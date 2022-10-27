def ica(img, output, n_ic=4, mask=None, name=None):
    if name is None:
        name = rm_ext(os.path.basename(img))
    ica_img = os.path.join(output, "IC_"+name+".nii.gz")

    canica=CanICA(n_components=n_ic, memory="nilearn_cache", memory_level=2, mask=mask, random_state=0, n_jobs=1, n_init=4)
    canica.fit(img)
    canica.components_img_.to_filename(ica_img)
    
    ica_fig_ls = []

    for i, cur_img in enumerate(iter_img(canica.components_img_)):
        ica_jpg = os.path.join(output, "IC"+str(i)+"_"+name+".jpg")
        plot_stat_map(cur_img, display_mode="ortho", title="IC %d" % i, colorbar=True, output_file=ica_jpg)
        ica_fig_ls.append(ica_jpg)