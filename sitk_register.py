import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    #plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.imshow(fixed_npa[:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    #plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.imshow(moving_npa[:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha( alpha, fixed, moving, moving_resampled):
    fixed_npa = sitk.GetArrayViewFromImage(fixed)
    moving_npa= sitk.GetArrayViewFromImage(moving)
    
    plt.subplots(1,3,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(2,2,1)
    #plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.imshow(fixed_npa,cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(2,2,2)
    #plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.imshow(moving_npa,cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    

    plt.subplot(2,2,3)
    img0 = (1.0 - alpha)*moving + alpha*moving_resampled
    plt.imshow(sitk.GetArrayViewFromImage(img0));
    plt.title('registered moving vs original moving')
    plt.axis('off')

    plt.subplot(2,2,4)
    img1 = (1.0 - alpha)*fixed + alpha*moving_resampled
    plt.imshow(sitk.GetArrayViewFromImage(img1));
    plt.title('registered moving vs fixed')
    plt.axis('off')

    #plt.savefig(fn)
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))  


def register(fn1, fn2, output_img_fn, output_transform_fn):
    fixed_image_sitk =  sitk.ReadImage(fn1, sitk.sitkFloat32)
    moving_image_sitk = sitk.ReadImage(fn2, sitk.sitkFloat32) 
    
    fixed_image = sitk.GetArrayFromImage(fixed_image_sitk)
    moving_image = sitk.GetArrayFromImage(moving_image_sitk)

    del fixed_image_sitk, moving_image_sitk

    dim0 = moving_image.shape[0]
    if fixed_image.shape[0] > moving_image.shape[0] : dim0 = fixed_image.shape[0]

    dim1 = moving_image.shape[1]
    if fixed_image.shape[1] > moving_image.shape[1] : dim1 = fixed_image.shape[1]
    fixed_img_padded = np.zeros([dim0, dim1], dtype=float)
    moving_img_padded = np.zeros([dim0, dim1], dtype=float)

    fixed_img_padded[0:fixed_image.shape[0],0:fixed_image.shape[1]] = fixed_image[:,:]
    moving_img_padded[0:moving_image.shape[0],0:moving_image.shape[1]] = moving_image[:,:]

    fixed_image_padded_sitk = sitk.GetImageFromArray(fixed_img_padded) 
    moving_image_padded_sitk = sitk.GetImageFromArray(moving_img_padded) 

    #interact(display_images, fixed_image_z=(0,fixed_image_padded_sitk.GetSize()[1]-1), moving_image_z=(0,moving_image_padded_sitk.GetSize()[1]-1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image_padded_sitk)), moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image_padded_sitk)));

    #print(fixed_image_padded_sitk.GetHeight(), fixed_image_padded_sitk.GetWidth() )
    #print(moving_image_padded_sitk.GetHeight(), moving_image_padded_sitk.GetWidth() )
    initial_transform = sitk.CenteredTransformInitializer(fixed_image_padded_sitk, moving_image_padded_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, initial_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())

    #interact(display_images_with_alpha, image_z=(0,fixed_image_padded_sitk.GetSize()[1]), alpha=(0.0,1.0), fixed = fixed(fixed_image_padded_sitk), moving=fixed(moving_resampled));

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    #registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    #registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    #registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image_padded_sitk, sitk.sitkFloat32), sitk.Cast(moving_image_padded_sitk, sitk.sitkFloat32))

    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, final_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())

    interact(display_images_with_alpha, alpha=0.5, fixed = fixed(fixed_image_padded_sitk), moving=fixed(moving_image_padded_sitk), moving_resampled=fixed(moving_resampled) );
    sitk.WriteImage(moving_resampled, output_img_fn) #os.path.join(output_dir, 'RIRE_training_001_mr_T1_resampled.mha'))
    sitk.WriteTransform(final_transform, output_transform_fn) # os.path.join(output_dir, 'RIRE_training_001_CT_2_mr_T1.tfm'))

