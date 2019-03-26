import SimpleITK as sitk
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from utils.utils import *
#matplotlib inline

def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []

def update_iterations(registration_method):
    global metric_values, multires_iterations
    metric=registration_method.GetMetricValue()
    metric_values.append(metric)                                       
    return 0

def update_multires_iterations():
    global metric_values, multires_iterations
    iteration=len(metric_values)
    multires_iterations.append(iteration)

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(out_fn) : #registration_method):
    global metric_values, multires_iterations
    
    # Plot the similarity metric values

    plt.plot(range(len(metric_values)), metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.savefig(out_fn)
    plt.close()
    return 0

def receptor_show(img_fn, img2_fn, title=None, margin=0.05, dpi=80, direction="caudal_to_rostral"):
    img = sitk.ReadImage(img_fn)
    img2 = sitk.ReadImage(img2_fn)
    nda = sitk.GetArrayViewFromImage(img)
    nda2 = sitk.GetArrayViewFromImage(img2)
    spacing = img.GetSpacing()
    spacing2 = img.GetSpacing()
       
    #nda = np.flip(nda.T, 1)
    #if direction == "rostral_to_caudal":
    #    nda = np.flip(nda, 0)
 
    ysize = nda.shape[0]
    xsize = nda.shape[1]

    ysize2 = nda2.shape[0]
    xsize2 = nda2.shape[1]
      
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    #fig = plt.figure(title, figsize=figsize, dpi=dpi)
    #ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], 0, ysize*spacing[0])
    plt.subplot(2,1,1)
    t = plt.imshow(nda, extent=extent, interpolation='hamming', origin='lower')#cmap='gray',
    plt.subplot(2,1,2)
    t = plt.imshow(nda2, extent=extent, interpolation='hamming', origin='lower')#cmap='gray',
    
    if(title):
        plt.title(title)
    plt.show()



def display_images_with_alpha( alpha, fixed, moving, moving_resampled, fn, order_fixed, order_moving, fixed_tier, moving_tier, metric=0):
    fixed_npa = (fixed - fixed.min() ) / (fixed.max() - fixed.min())  #imageio.imread(fixed)
    moving_npa = (moving - moving.min() ) / (moving.max() - moving.min())  #imageio.imread(moving)
    moving_resampled_npa =(moving_resampled-moving_resampled.min()) / (moving_resampled.max() - moving_resampled.min())  #imageio.imread(moving_resampled)
    #moving_npa= moving #imageio.imread(moving)
    #moving_resampled_npa= moving_resampled #imageio.imread(moving_resampled)
    extent = 0, moving_npa.shape[1], 0, moving_npa.shape[0]
    
    plt.title( 'moving:'+str(moving_tier)+ ' fixed'+ str(order_moving))
    plt.imshow(fixed_npa, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
    plt.imshow(moving_resampled_npa, cmap=plt.cm.hot, alpha=0.35, interpolation='bilinear', extent=extent)
    plt.title('rsl moving vs fixed')
    plt.axis('off')


    plt.axis('off')
    plt.tight_layout()

    plt.savefig(fn, dpi=200,bbox_inches="tight" )
    plt.clf()
    return 0 
    
    if metric != 0 :
        plt.title("Metric= "+ str(metric) )

    plt.subplots(1,3,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(2,3,1)
    plt.title( 'fixed:'+str(fixed_tier)+ ' '+ str(order_fixed))
    plt.imshow(fixed_npa,cmap=plt.cm.gray);
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(2,3,2)
    plt.title( 'moving:'+str(moving_tier)+ ' '+ str(order_moving))
    plt.imshow(moving_npa,cmap=plt.cm.bone);
    plt.axis('off')
   
    plt.subplot(2,3,3)
    plt.title( 'rsl moving:'+str(moving_tier)+ ' '+ str(order_moving))
    plt.imshow(moving_resampled_npa,cmap=plt.cm.bone);
    plt.axis('off')


    plt.subplot(2,3,4)
    img1 = (1.0 - alpha)*fixed_npa + alpha*moving_resampled_npa
    plt.imshow(fixed_npa, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
    plt.imshow(moving_npa, cmap=plt.cm.hot, alpha=0.35, interpolation='bilinear', extent=extent)
    plt.title('original moving vs fixed')

    plt.subplot(2,3,5)
    plt.imshow(moving_npa, cmap=plt.cm.bone, interpolation='bilinear', extent=extent)
    plt.imshow(moving_resampled_npa, cmap=plt.cm.hot, alpha=0.35, interpolation='bilinear', extent=extent)
    plt.title('rsl moving vs original moving')
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.imshow(fixed_npa, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
    plt.imshow(moving_resampled_npa, cmap=plt.cm.hot, alpha=0.35, interpolation='bilinear', extent=extent)
    plt.title('rsl moving vs fixed')
    plt.axis('off')


    plt.axis('off')
    plt.tight_layout()

    plt.savefig(fn, dpi=200,bbox_inches="tight" )
    #plt.show()
    plt.clf()

from skimage.exposure import  equalize_hist
def preprocess_img(img):

    ar = sitk.GetArrayViewFromImage(img)
    ar = equalize_hist(ar) 
    img = sitk.GetImageFromArray(ar)

    return img



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

def create_mask(img):
    from skimage.filters import threshold_li as threshold
    ar = sitk.GetArrayViewFromImage(img)
    thr = threshold(ar)
    #print(ar.min(), ar.mean(), ar.max(), thr)
    idx = ar > thr
    mask=np.zeros(ar.shape)
    mask[idx] = 1
    #print("Number of pixels in mask: ",np.sum(mask))
    mask_img = sitk.GetImageFromArray(mask)
    return mask_img

def register(fixed_fn, moving_fn, transform_fn, resolutions, max_iterations, transform_type="Euler2DTransform", preprocess=False, numberOfHistogramBins=100, fixed_spacing=[0.2, 0.2], moving_spacing=[0.2, 0.2],stepLength=2, valueTolerance=1e-5, stepTolerance=1e-5,  mask=True, invert_transform=False):
    fixedImage = sitk.ReadImage(fixed_fn,  sitk.sitkFloat32)
    movingImage = sitk.ReadImage(moving_fn,  sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    if mask : 
        fixedImageMask = create_mask(fixedImage)
        registration_method.SetMetricFixedMask(fixedImageMask)
        movingImageMask = create_mask(movingImage)
        registration_method.SetMetricMovingMask(movingImageMask)

    if preprocess :
        fixedImage = preprocess_img( fixedImage )
        movingImage = preprocess_img(movingImage)
  
    fixedImage.SetSpacing(fixed_spacing)
    movingImage.SetSpacing(moving_spacing)

    if transform_type == "AffineTransform" :
        transform = sitk.AffineTransform(2) 
    else : 
        transform = sitk.Euler2DTransform() 
    
    initial_transform=sitk.CenteredTransformInitializer(fixedImage, movingImage,transform,sitk.CenteredTransformInitializerFilter.GEOMETRY ) #MOMENTS )
    ## Similarity metric settings.
    #registration_method.SetMetricAsANTSNeighborhoodCorrelation(5)
    registration_method.SetMetricAsCorrelation()
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=numberOfHistogramBins)
    #registration_method.SetMetricAsJointHistogramMutualInformation(100)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(1)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    #registration_method.SetOptimizerAsPowell(numberOfIterations=max_iterations, maximumLineIterations=2000, stepTolerance=stepTolerance, valueTolerance=valueTolerance, stepLength=stepLength)
    #registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=300, initialRadius=100)
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.01, numberOfIterations=600, convergenceMinimumValue=1e-10, convergenceWindowSize=10 )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.     
    resolution_sigmas = resolutions #[ i / 2.355 for i in resolutions ]
    #resolution_sigmas = [ i / 2.355 for i in resolutions ]
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = resolutions ) 
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas= resolution_sigmas ) #[1] * len(resolutions) )
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    registration_method.SetNumberOfThreads(8)

    final_transform = registration_method.Execute(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(movingImage, sitk.sitkFloat32))
    #print(final_transform)
    #print( registration_method.GetOptimizerStopConditionDescription())
    final_similarity_value = registration_method.MetricEvaluate(fixedImage, movingImage) 
    iterations = registration_method.GetOptimizerIteration()
    print("Final metric =", final_similarity_value  )
    #print("Iterations =", iterations)
    
    #moving_resampled = sitk.Resample(movingImage, fixedImage, final_transform, sitk.sitkLinear, 0.0, movingImage.GetPixelID())
    if invert_transform : 
        final_transform = final_transform.GetInverse()
    sitk.WriteTransform(final_transform, transform_fn)
    del registration_method
    del initial_transform
    del final_transform
    del fixedImage
    del movingImage
    return final_similarity_value, iterations


def register3D(fixed_fn, moving_fn, transform_fn, resolutions, max_iterations, transform_type="Euler2DTransform", preprocess=False, numberOfHistogramBins=100, fixed_spacing=[0.2, 0.2], moving_spacing=[0.2, 0.2],stepLength=2, valueTolerance=1e-5, learningRate=0.1, convergenceWindowSize=10, stepTolerance=1e-5, mask=True, invert_transform=False, optimizer="powell"):
    if optimizer == "powell" :
        plot_fn = "coreg_opt-powell_hist-"+str(numberOfHistogramBins)+"_stpLen-"+str(stepLength)+".png"
    elif optimizer :
        plot_fn = "coreg_opt-gradDesc_hist-"+str(numberOfHistogramBins)+"_rate-"+str(learningRate)+"_wind-"+str(convergenceWindowSize)+".png"
    registration_method = sitk.ImageRegistrationMethod()
    #if preprocess :
    #    fixedImage = preprocess_img(fixedImage)
    #    movingImage = preprocess_img(movingImage)
    moving_reader = sitk.ImageFileReader()
    #moving_reader.SetImageIO("MINCImageIO")
    moving_reader.SetFileName(moving_fn)
    movingImage = moving_reader.Execute();

    fixed_reader = sitk.ImageFileReader()
    #fixed_reader.SetImageIO("MINCImageIO")
    fixed_reader.SetFileName(fixed_fn)
    fixedImage = fixed_reader.Execute();

    fixedImage.SetSpacing(fixed_spacing)
    movingImage.SetSpacing(moving_spacing)

    if transform_type == "AffineTransform" :
        transform = sitk.AffineTransform(3) 
    else : 
        transform = sitk.Euler2DTransform() 
    
    initial_transform=sitk.CenteredTransformInitializer(fixedImage, movingImage,transform,sitk.CenteredTransformInitializerFilter.MOMENTS )
    ## Similarity metric settings.
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(5)
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=numberOfHistogramBins)
    #registration_method.SetMetricAsJointHistogramMutualInformation(100)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    #
    # Optimizer settings.
    #
    if optimizer == "powell" :
        registration_method.SetOptimizerAsPowell(numberOfIterations=max_iterations, maximumLineIterations=2000, stepTolerance=stepTolerance, valueTolerance=valueTolerance, stepLength=stepLength)
    elif optimizer == "gradient_descent" :
        registration_method.SetOptimizerAsGradientDescent(learningRate=learningRate, numberOfIterations=max_iterations, convergenceMinimumValue=1e-10, convergenceWindowSize=10 )
    #registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=300, initialRadius=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    #
    # Setup for the multi-resolution framework.     
    #
    
    #registration_method.SetMaximumKernelWidth(64)
    resolution_sigmas = resolutions #[ i / 2.355 for i in resolutions ]
    #resolution_sigmas = [ i / 2.355 for i in resolutions ]
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = resolutions ) 
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas= resolution_sigmas ) #[1] * len(resolutions) )
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetNumberOfThreads(8)

    #
    # Set Callbacks
    #
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda : update_iterations(registration_method)) 
    registration_method.AddCommand(sitk.sitkEndEvent, lambda : plot_values(plot_fn) )

    final_transform = registration_method.Execute(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(movingImage, sitk.sitkFloat32))
    #print(final_transform)
    #print( registration_method.GetOptimizerStopConditionDescription())
    final_similarity_value = registration_method.MetricEvaluate(fixedImage, movingImage) 
    iterations = registration_method.GetOptimizerIteration()
    print("Final metric =", final_similarity_value  )
    #print("Iterations =", iterations)
    
    #moving_resampled = sitk.Resample(movingImage, fixedImage, final_transform, sitk.sitkLinear, 0.0, movingImage.GetPixelID())
    if invert_transform : 
        final_transform = final_transform.GetInverse()
    sitk.WriteTransform(final_transform, transform_fn)
    del registration_method
    del initial_transform
    del final_transform
    del fixedImage
    del movingImage
    return final_similarity_value, iterations




def resample(moving_fn, transform_fn_list, rsl_fn="", ndim=2):
    movingImage = sitk.ReadImage(moving_fn)
    composite = sitk.Transform(ndim, sitk.sitkComposite )
    for fn in transform_fn_list :
        transform = sitk.ReadTransform(fn)
        composite.AddTransform(transform)

    interpolator = sitk.sitkCosineWindowedSinc
    rslImage = sitk.Resample(movingImage, composite, interpolator, 0.)
    #rslImage=movingImage
    rsl = np.copy(sitk.GetArrayViewFromImage(rslImage))
    print("Max:", rsl.max(), "Min:", rsl.min() )
    if ndim == 2 :
        if rsl_fn != "" : imageio.imsave(rsl_fn, rsl)
    else :
        writer = sitk.ImageFileWriter()
        writer.SetFileName(rsl_fn)
        writer.Execute(rslImage)

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
