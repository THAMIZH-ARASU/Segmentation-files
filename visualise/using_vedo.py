from vedo import Volume, show
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the NIfTI file
nifti_img = nib.load('liver/labelsTr/liver_129.nii')

# Get the image data as a NumPy array
data = nifti_img.get_fdata()

# Apply Gaussian smoothing
smoothed_data = gaussian_filter(data, sigma=1)
volume = Volume(smoothed_data).c("red").alpha([0.0, 0.5, 1.0]) 

# Show volumes
show(volume)
