# preprocessing

preprocessing is an fMRI preprocessing pipeline specialized for resting state data, though it can be used for task-related data. This pipeline was designed with modularity in mind and, as such, users are given a number of options at each step and can choose the order in which steps are executed. Though it should be noted that some steps are dependent on other steps, and so not all possible orders will execute.

# Usage

Neuroimaging data can be preprocessed using the `wrapper_lvl2` function.

```python
wrapper_lvl2(input_file="/path/to/input.txt", config_file="/path/tp/config.txt")
```

If you wish to preprocess each dataset in parallel, you can simply use the `wrapper_lvl2_parallel` function the same way. Additionally, you may specify the `max_workers` - i.e., the maximum number of processing cores you wish to utilize simultaneously for preprocessing. If left unspecified, it will default to `min(32, os.cpu_count() + 4)`, as per the latest concurrent.futures version. Usage is illustrated below:

```python
wrapper_lvl2_parallel(input_file="/path/to/input.txt", config_file="/path/tp/config.txt", max_workers=None)
```

The `input_file` contains information directing the pipeline to the functional and anatomical datasets to be analyzed, and the output directory where you would like to save the preprocessed data. The `input_file` must be formatted as follows, with each row representing a different dataset being analyzed

```
FUNC=[/path/to/functional/image.nii], ANAT=[/path/to/anatomical/MPRage.nii], OUTPUT=[/path/to/output/directory/]
FUNC=[/path/to/functional/image.nii], ANAT=[/path/to/anatomical/MPRage.nii], OUTPUT=[/path/to/output/directory/]
FUNC=[/path/to/functional/image.nii], ANAT=[/path/to/anatomical/MPRage.nii], OUTPUT=[/path/to/output/directory/]
```

The `config_file` contains the settings to be used while preprocessing the provided datasets. These settings will be applied identically across all datasets.

```
SKULLSTRIP=[1]
SMOOTH=[6]
```

The config file does not need to contain configurations for all available pipeline steps. If a step is ommitted, the pipeline will simply apply the default settings, stored in the `default_config` variable as a dictionary. Any step in the pipeline can be disabled by simply setting its config value to `0`. The default configuration settings are: 

```
SKULLSTRIP=[1]
SLICETIME=[1]
MOTCOR=[1]
NORM=[1]
SMOOTH=[6]
MOTREG=[1]
GSR=[1]
NUISANCE=[3]
TEMPLATE=[/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz]
SCRUB=[UNION]
```

# Preprocessing Steps

The order in which preprocessing steps are carried out is stored in the `step_order` dictionary variable. The user may modify this if they wish to alter the order in which preprocessing steps are applied.

```python
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
```

## BASELINE

This step copies and renames the anatomical and functional data from the source and into the output directory, as well as extracting the TR from the functional data. 

## SKULLSTRIP

FSL's Brain Extraction Tool (BET) is used to extract the brain from the rest of the anatomical scan.

```shell
bet <input> <output> -R
```

The `-R` flag denotes that this is running a more "robust" brain center estimation. See the [BET Documentation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) for details.

## SLICETIME

Slicetime correction of functional data is performed using AFNI's 3dTshift tool. This will use the slicewise shifting information in the dataset header and the TR value extracted from the header at BASELINE. 

## MOTCOR

Motion correction is implemented using AFNI's 3dvolreg tool. This is also the step at which motion parameters are estimated. As such, this step is necessary for motion regression (see [NUISANCE](#nuisance) for details).

## NORM

Anatomical and functional data are warped to fit standardized space, as defined by `TEMPLATE` in the config. This step makes use of FSL's FLIRT and FNIRT tools to spatially normalize the data.

## NUISANCE

This step will regress out nuissance regressors from fMRI voxel timecourse data. The nuissance regressors can include the 6 motion parameters as well as the global signal, defined as the mean timecourse of the cerebrospinal fluid. Nuissance regression is carried out using AFNI's 3dDeconvolve tool. The specified `NUISSANCE` config value corresponds to the degree of the polynomial corresponding to the null hypothesis (see [3dDeconvolve documentation](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html) for details).

To nuissance regressor options in the `config`, simply set MOTREG (motion regression) and `GSR` (global signal regression) to 1:

```
MOTREG=[1]
GSR=[1]
```

Please note that `MOTREG` will only run if `MOTCOR` ([motion regression](#motcor)) is also enabled, as that is the step where motion parameters are estimated. Additionally, since `MOTREG` is dependent on `MOTCOR`, then `MOTCOR` **must** occur before `MOTREG`. 

## SCRUB

This step removes ("scrubs") outlier timepoints from the fMRI timecourse and lineraly interpolates them from surrounding non-scrubbed timepoints. Outliers are determined with two criteria: DVARS and/or FD.

- DVARS: RMS intensity difference of volume N to volume N+1 (see Power et al, NeuroImage, 59(3), 2012)
- FD: frame displacement (average of rotation and translation parameter differences - using weighted scaling, as in Power et al.)

Users can specify the criteria used to define outliers in the `SCRUB` entri in config. Available options are:

- `DVARS`: Only scrub timepoints that are flagged as outliers using DVARS.
- `FD`: Only scrub timepoints that are flagged as outliers using FD.
- `UNION`: Scrub timepoints that are flagged as outliers using DVARS and/or FD.
- `INTERSECT`: Only scrub timepoints that are flagged as outliers using **both** DVARS and FD, while ignoring those that are only flagged under one criterion.

## SMOOTH

Spatial smoothing and mean filtering is applied to the functional data using FSL's `fslmaths` tool. The size of the spatial smoothing kernel is defined in the config such that `SMOOTH=[6]` will smooth the functional data using a 6mm FWHM gaussian kernel.