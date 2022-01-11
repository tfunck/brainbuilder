import utils.ants_nibabel as nib
#import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import label
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy, kurtosis, skew
from scipy.interpolate import griddata

fn="RG#HG#MR1S6#R#flum#5716#21#L#L.nii.gz"
mask_fn = "RG#HG#MR1S6#R#flum#5716#21#L_seg.nii.gz"

def pseudo_classify_autoradiograph(autoradiograph_fn, mask_fn):

    out_fn = re.sub('.nii','_cls.nii', autoradiograph_fn) 
    #load autoradiograph
    img = nib.load(autoradiograph_fn)
    vol = img.get_fdata()

    #load mask
    mask_img = nib.load(mask_fn)
    mask_vol = mask_img.get_fdata()

    #blur to 200um
    vol = gaussian_filter(vol, (0.2/0.02)/np.pi)

    xdim, zdim = vol.shape

    # set size of sliding kernel 
    mm=3
    kernel_dim = int(1/0.02) * mm
    kernel_pad = int( (kernel_dim+1) / 2 )

    #smallest area to keep in classified image
    smallest_area=5 / (0.02*0.02)

    features=[]
    locations=[]
    step=kernel_dim*2
    # slide kernel over autoradiograph and calculate image metrics
    for x in range(0,xdim,step):
        for z in range(0,zdim,step):
            if mask_vol[x,z] > 0 :
                i0=max(0,x-kernel_pad)
                j0=max(0,z-kernel_pad)
                
                i1=min(xdim,x+kernel_pad)
                j1=min(zdim,z+kernel_pad)

                section = vol[i0:i1, j0:j1 ]
                mask_section=mask_vol[i0:i1, j0:j1] 

                #plt.imshow(section*mask_section); plt.show()
                section = section[mask_section>0]

                m=np.mean(section)
                s=np.std(section)
                k=kurtosis(section.reshape(-1,1))[0]
                sk=skew(section.reshape(-1,1))[0]
                e=entropy( section.reshape(-1,1) )[0]
                vector = [x,z,m,s,k,sk,e]
                features.append(vector)
                locations.append( [ x, z ] )

    #normalize columns of feature metrics
    features = (np.array(features) - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Use a gaussian mixture model to classify the features into n class labels
    n=3
    labels = GaussianMixture(n_components=n).fit_predict(features) +1

    # Define a mesh grid over which we can interpolate label values
    xx, zz = np.meshgrid(range(xdim), range(zdim))

    # Apply nearest neighbour interpolation
    out = griddata(locations, labels, (xx, zz), method='nearest')
    out = out.T
    out[ mask_vol == 0 ] = 0
    out = label(out)


    #Get rid of labeled regions that are less than the minimum label size
    labels_unique = np.unique(labels)[1:]
    for l in labels_unique :
        if np.sum(out==l) < smallest_area :
            out[ out == l ] = 0

    # save classified image as nifti
    nib.Nifti1Image(out.astype(np.int32), img.affine).to_filename(out_fn)



