import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
#matplotlib inline

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha( alpha, fixed, moving, moving_resampled, fn):
    fixed_npa = sitk.GetArrayViewFromImage(fixed)
    moving_npa= sitk.GetArrayViewFromImage(moving)
    moving_resampled_npa= sitk.GetArrayViewFromImage(moving_resampled)
    
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
    img0 = (1.0 - alpha)*moving_npa + alpha*moving_resampled_npa
    plt.imshow(img0);
    plt.title('registered moving vs original moving')
    plt.axis('off')

    plt.subplot(2,2,4)
    img1 = (1.0 - alpha)*fixed_npa + alpha*moving_resampled_npa
    plt.imshow(img1);
    plt.title('registered moving vs fixed')
    plt.axis('off')
    plt.savefig(fn)
    #plt.show()
    
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

def get_pad_dim(moving_image, fixed_image):
    dim0 = moving_image.shape[0]
    if fixed_image.shape[0] > moving_image.shape[0] : dim0 = fixed_image.shape[0]

    dim1 = moving_image.shape[1]
    if fixed_image.shape[1] > moving_image.shape[1] : dim1 = fixed_image.shape[1]
    return dim0, dim1

def pad_images(moving_image, fixed_image):
    dim0, dim1 = get_pad_dim(moving_image, fixed_image)

    fixed_img_padded = np.zeros([dim0, dim1], dtype=float)
    moving_img_padded = np.zeros([dim0, dim1], dtype=float)

    fixed_img_padded[0:fixed_image.shape[0],0:fixed_image.shape[1]] = fixed_image[:,:]
    moving_img_padded[0:moving_image.shape[0],0:moving_image.shape[1]] = moving_image[:,:]

    fixed_image_padded_sitk = sitk.GetImageFromArray(fixed_img_padded) 
    moving_image_padded_sitk = sitk.GetImageFromArray(moving_img_padded) 
    return moving_image_padded_sitk, fixed_image_padded_sitk


def register(fn1, fn2, output_img_fn, output_qc_fn, output_transform_fn, transform_list):
    fixed_image =  sitk.GetArrayFromImage(sitk.ReadImage(fn1, sitk.sitkFloat32))
    moving_image = sitk.GetArrayFromImage(sitk.ReadImage(fn2, sitk.sitkFloat32)) 
    
    moving_image_padded_sitk, fixed_image_padded_sitk =    pad_images(moving_image, fixed_image)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image_padded_sitk, moving_image_padded_sitk, sitk.Euler2DTransform(), True)
    
    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, initial_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR) #RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=1000, convergenceMinimumValue=1e-10, convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [1, 1, 1, 1, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[16, 8, 4, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image_padded_sitk, sitk.sitkFloat32), sitk.Cast(moving_image_padded_sitk, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, final_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())
    display_images_with_alpha(alpha=0.5, fixed = fixed_image_padded_sitk, moving=moving_image_padded_sitk, moving_resampled=moving_resampled, fn=output_qc_fn)

    if transform_list != [] :
        n = len(transform_list)+1
        print("N", n)
        composite = sitk.Transform(2, sitk.sitkComposite)
        for t in transform_list : 
            prior_transform = sitk.ReadTransform(t)
            composite.AddTransform(prior_transform)
        composite.AddTransform(final_transform)
        output_transform=composite
    else : 
        output_transform=final_transform
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    output_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, output_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())


    imageio.imsave(output_img_fn, sitk.GetArrayViewFromImage(output_resampled) )
    sitk.WriteTransform(final_transform, output_transform_fn) # os.path.join(output_dir, 'RIRE_training_001_CT_2_mr_T1.tfm'))
    return final_transform

