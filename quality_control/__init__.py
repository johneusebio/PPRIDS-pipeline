import os
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