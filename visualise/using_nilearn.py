from nilearn import plotting
import nibabel as nib

# Load saved NIfTI file
nii_image = nib.load("liver_1.nii")

# 3D visualization
plotting.view_img(nii_image, bg_img=None, opacity=0.5)