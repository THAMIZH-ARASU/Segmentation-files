import vtk
import numpy as np
import nibabel as nib

# Load the NIfTI file
    
nifti_img = nib.load('liver_0.nii')
# Get the image data as a NumPy array
data = nifti_img.get_fdata()
# Create a vtk image data object
image_data = vtk.vtkImageData()
# Set the dimensions and spacing
image_data.SetDimensions(data.shape[1], data.shape[0], data.shape[2])
image_data.SetSpacing(1, 1, 1)
image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

# Populate the vtk image data with the numpy array
for z in range(data.shape[2]):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            image_data.SetScalarComponentFromDouble(x, y, z, 0, data[y, x, z])

# Create a volume property
volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOn()
volume_property.SetInterpolationTypeToLinear()

# Create a color transfer function
color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(np.min(data), 0.0, 0.0, 0.0)
color_transfer_function.AddRGBPoint(np.max(data), 1.0, 1.0, 1.0)

# Create an opacity transfer function
opacity_transfer_function = vtk.vtkPiecewiseFunction()
opacity_transfer_function.AddPoint(np.min(data), 0.0)
opacity_transfer_function.AddPoint(np.max(data), 0.8)

volume_property.SetColor(color_transfer_function)
volume_property.SetScalarOpacity(opacity_transfer_function)

# Create a volume mapper
volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
volume_mapper.SetInputData(image_data)
volume_mapper.SetBlendModeToComposite()

# Create a volume
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Create a renderer and render window
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create an interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Add the volume to the renderer
renderer.AddVolume(volume)
renderer.SetBackground(0, 0, 0)
renderer.ResetCamera()

# Start the interaction
render_window.SetSize(600, 600)
render_window.Render()
render_window_interactor.Start()