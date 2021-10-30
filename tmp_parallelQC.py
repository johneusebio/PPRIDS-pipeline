import concurrent.futures
import os

import quality_control as qc
import preprocessing as pp


def run_qc(subj):
    topdir = "/mnt/d/doctoral_thesis/SART"
    nifti_pp = os.path.join(topdir, subj, "func", "s_scrub_nuis_nl_m_t_func.nii.gz")
    nifti_ref = os.path.join(topdir, subj, "func", "nl_m_t_func.nii.gz")
    mot_params = os.path.join(topdir, subj, "motion", "1d_t_func.1D")
    qc_dir = os.path.join(topdir, subj, "quality_control")
    mask = os.path.join(topdir, subj, "anat", "mask_MNI152_T1_2mm_brain.nii.gz")
    memory_obj=subj

    qc.qclvl1(nifti_pp, nifti_ref, mot_params, qc_dir, mask=mask, ica_memory=memory_obj)

subjs = ["2357ZL", "2359ZL", "2360ZL", "2364ZL", "2365ZL", "2371ZL", "2377ZL", "2431ZL", "2486ZL", "2505ZL", "2523ZL", "2524ZL", "2528ZL", "2538ZL", "2540ZL", "2542ZL", "2544ZL", "2548ZL", "2553ZL", "2554ZL", "2555ZL", "2556ZL", "2561ZL", "2568ZL", "2570ZL", "2573ZL", "2577ZL", "2581ZL", "2582ZL", "2583ZL", "2584ZL", "2588ZL", "2590ZL", "2591ZL", "2592ZL", "2593ZL", "2594ZL", "2595ZL", "2596ZL", "2597ZL", "2599ZL", "2600ZL", "2602ZL", "2603ZL", "2604ZL", "2605ZL", "2607ZL", "2608ZL", "2609ZL", "2610ZL"]

complete_order = {}

with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_qc, subj): subj for subj in subjs}
    
    for i, f in enumerate(concurrent.futures.as_completed(futures), start=0):
        subj = futures[f]  # deal with async nature of submit
        print(f"subj idx: {subj}")
        print(subj)

        # count how many tasks are done (or just initialize a counter at the top to avoid looping)
        states = [i._state for i in futures]
        print(states)
        print(i)
        idx = states.count("FINISHED")
        complete_order[subj] = idx - 1
        print(f"completed order: {complete_order}")

        try:
            data = f.result()
        except Exception as exc:
            print("%r generated an exception: %s" % (subj, exc))
        else:
            print("%r page is %d bytes" % (subj, len(data)))

        print("")
