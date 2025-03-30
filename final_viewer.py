from vedo import Volume, show
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

# Load the NIfTI file
nifti_img = nib.load('prediction.nii')
print("Shape: ", nifti_img.shape)
print("Num of slices: ", nifti_img.shape[2])

# Get the image data as a NumPy array
data = nifti_img.get_fdata()

# Apply Gaussian smoothing first
smoothed_data = gaussian_filter(data, sigma=1.6)

# Resample the data to have more slices in the z direction
# This will interpolate new slices between existing ones
zoom_factors = [1, 1, 200/75]  # Keep x,y the same, adjust z to have 200 slices
resampled_data = zoom(smoothed_data, zoom_factors, order=1)  # order=1 for linear interpolation

print("Resampled shape:", resampled_data.shape)

# Create the volume with the resampled data
volume = Volume(resampled_data)
volume.cmap("red").alpha([0.0, 0.5, 1.0])
#volume.mode(1)  # Volumetric rendering

# Show the volume
show(volume)
