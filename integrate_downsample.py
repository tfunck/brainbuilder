from utils.utils import shell
from scipy.integrate import simps
import nibabel as nib
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp2d, griddata

def integrate_downsample_tif(in_image, step, affine, out_image):
    img = imageio.imread(in_image)
    xstep = affine[0,0] 
    zstep = affine[2,2] 
    xmax=xstep * img.shape[0]
    zmax=zstep * img.shape[1]
    
    xlo = np.arange(0, xmax, step) # img.shape[0]) 
    zlo = np.arange(0, zmax, step) #img.shape[1]) 
    
    xlo_int = range(0,len(xlo))
    zlo_int = range(0,len(zlo))

    dwn_img=np.zeros([len(xlo),len(zlo)])

    xhi = np.arange(0, xmax, xstep)
    zhi = np.arange(0, zmax, zstep)
   
    for x0 , x_int in zip(xlo, xlo_int) :
        x1 = x0 
        xi0 = int(np.round(x0 / xstep))
        xi1=xi0
        while x1 < x0 + step -xstep and xi1 < img.shape[0] : 
            x1 += xstep         
            xi1 += 1

        for z0, z_int in zip(zlo, zlo_int ):

            zi0 = int(np.round(z0 / zstep))
            z1 = z0 
            zi1 = zi0 
           
            while z1 < z0 + step-zstep  and zi1 < img.shape[1] : 
                z1 += zstep
                zi1 += 1
            dwn_img[x_int,z_int] += np.sum(img[ xi0:xi1,  zi0:zi1 ])

    imageio.imsave(out_image, dwn_img)
    #print(np.sum(img), np.sum(dwn_img))
    
    #plt.subplot(2,1,1); plt.imshow(img)
    #plt.subplot(2,1,2); plt.imshow(dwn_img)
    #plt.show()
    

