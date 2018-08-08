import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from utils.utils import *
import cv2
#matplotlib inline

def display_images_with_alpha( alpha, fixed, moving, moving_resampled, fn, order_fixed, order_moving, fixed_tier, moving_tier):
    fixed_npa = fixed #imageio.imread(fixed)
    moving_npa= moving #imageio.imread(moving)
    moving_resampled_npa= moving_resampled #imageio.imread(moving_resampled)
    
    plt.subplots(1,3,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(2,2,1)
    plt.title( str(fixed_tier)+ ' '+ str(order_fixed))
    plt.imshow(fixed_npa,cmap=plt.cm.Greys_r);
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(2,2,2)
    plt.title( str(moving_tier)+ ' '+ str(order_moving))
    plt.imshow(moving_npa,cmap=plt.cm.Greys_r);
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
    plt.clf()



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

#def sitk_register(fixed, moving, transform_fn):
def register(fixed_fn, moving_fn, transform_fn, resolutions, max_iterations, max_length):
    fixedImage = sitk.ReadImage(fixed_fn)
    movingImage = sitk.ReadImage(moving_fn)

    initial_transform = sitk.CenteredTransformInitializer(fixedImage, movingImage, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS )
    
    registration_method = sitk.ImageRegistrationMethod()
    ## Similarity metric settings.
    #registration_method.SetMetricAsANTSNeighborhoodCorrelation(5)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    #registration_method.SetMetricAsJointHistogramMutualInformation(100)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(1)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsPowell(numberOfIterations=max_iterations, maximumLineIterations=2000, stepTolerance=1e-5, valueTolerance=1e-5, stepLength=2)
    #registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=300, initialRadius=100)
    #registration_method.SeatOptimizerAsConjugateGradientLineSearch(learningRate=0.1, numberOfIterations=300)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.      
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = resolutions ) 
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas= resolutions ) #[1] * len(resolutions) )
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    registration_method.SetNumberOfThreads(8)

    final_transform = registration_method.Execute(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(movingImage, sitk.sitkFloat32))
    #moving_resampled = sitk.Resample(movingImage, fixedImage, final_transform, sitk.sitkLinear, 0.0, movingImage.GetPixelID())
    sitk.WriteTransform(final_transform, transform_fn)
    del registration_method
    del initial_transform
    del final_transform
    del fixedImage
    del movingImage

def resample(moving_fn, fixed_fn, transform_fn_list, rsl_fn=""):
    movingImage = sitk.ReadImage(moving_fn)
    fixedImage = sitk.ReadImage(fixed_fn)

    composite = sitk.Transform(2, sitk.sitkComposite )
    for fn in transform_fn_list :
        transform = sitk.ReadTransform(fn)
        composite.AddTransform(transform)

    interpolator = sitk.sitkCosineWindowedSinc
    rslImage = sitk.Resample(movingImage, composite, interpolator, 0.)
    rsl = np.copy(sitk.GetArrayViewFromImage(rslImage))
    if rsl_fn != "" : imageio.imsave(rsl_fn, rsl)
    return rsl 


'''def register(fn1, fn2, output_img_fn, output_qc_fn, output_transform_fn, transform_list):
    fixed_image =  sitk.GetArrayFromImage(sitk.ReadImage(fn1, sitk.sitkFloat32))
    moving_image = sitk.GetArrayFromImage(sitk.ReadImage(fn2, sitk.sitkFloat32))

    moving_image_padded_sitk, fixed_image_padded_sitk =    pad_images(moving_image, fixed_image)
    #The following is a bit of a rigmarole to set the background of the images to 0 (based on the median value).
    #This improves the estimation of Center of Mass in the CenteredTransformInitializer function

    initial_transform = sitk.CenteredTransformInitializer(fixed_image_padded_sitk, moving_image_padded_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY )

    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, initial_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())
    

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)#RANDOM)  #
    registration_method.SetMetricSamplingPercentage(0.5)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=1000, convergenceMinimumValue=1e-10, convergenceWindowSize=10)
    
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.      
    sigmas=[16, 8, 4, 1]
    shrinks = [4,2,1,1] #* len(sigmas)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors =shrinks ) 
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image_padded_sitk, sitk.sitkFloat32), sitk.Cast(moving_image_padded_sitk, sitk.sitkFloat32))
    
    moving_resampled = sitk.Resample(moving_image_padded_sitk, fixed_image_padded_sitk, final_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())
    #moving_resampled = sitk.Resample(temp_moving_sitk, temp_fixed_sitk, final_transform, sitk.sitkLinear, 0.0, moving_image_padded_sitk.GetPixelID())
    
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

    print("Saving output image to ", output_img_fn)
    imageio.imsave(output_img_fn, sitk.GetArrayViewFromImage(output_resampled) )
    sitk.WriteTransform(final_transform, output_transform_fn) # os.path.join(output_dir, 'RIRE_training_001_CT_2_mr_T1.tfm'))
    return final_transform
'''
