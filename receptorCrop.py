
import scipy.misc
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
from glob import glob
from sklearn.feature_extraction import image
from sys import exit, argv
import os
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import normalized_mutual_info_score
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from skimage.morphology import remove_small_holes, reconstruction
from skimage.restoration import inpaint
import pandas as pd
import argparse
from slices2mnc import slices2mnc
from utils.mouse_click import click
from utils.utils import *
from skimage import filters
from cropp_img import cropp_img

### Required libraries :
### pip3 install --upgrade --user imageio scikit-image opencv-python


def crop(img, img_dwn, bounding_box_dwn) :
    if np.sum(bounding_box_dwn) > 0 : 
        #Create downsampled cropped image
        cropped_dwn = img_dwn * bounding_box_dwn 
        #cropped_dwn[(bounding_box_dwn==0) | (mask==1)]=np.median( img_dwn[(bounding_box_dwn == 0) | (mask==1) ])
        #plt.subplot(2,2,1)
        #plt.imshow(bounding_box_dwn)
        #plt.show()
        
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(bounding_box_dwn)
        y0=min(ymin)
        y1=max(ymax)
        x0=min(xmin) 
        x1=max(xmax)
        cropped_dwn = cropped_dwn[y0:y1,x0:x1]
        
        #Create cropped image at full resolution
        bounding_box=scipy.misc.imresize(bounding_box_dwn,size=(img.shape[0],img.shape[1]),interp="nearest")
        #plt.subplot(2,2,1)
        #plt.imshow(bounding_box_dwn)
        #plt.subplot(2,2,2)
        #plt.imshow(cropped_dwn)
        #plt.subplot(2,2,3)
        #plt.imshow(bounding_box)
        #plt.show()
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(bounding_box)
        y0=min(ymin)
        y1=max(ymax)
        x0=min(xmin) 
        x1=max(xmax)
        cropped = img * bounding_box
        cropped = cropped[y0:y1,x0:x1]
    else :
        cropped = img
        cropped_dwn = img_dwn
        bounding_box=np.zeros(cropped.shape)
    return cropped, cropped_dwn, bounding_box


def save_qc(img_dwn, img_smoothed, im_thr, cc, bb, cropped_dwn,out_fn,manual_check):
    dim0=img_dwn.shape[0]
    dim1=img_dwn.shape[1]
    qc=np.zeros([ dim0, dim1*2 ])
    plt.clf()
    plt.Figure()
    plt.title("Image Downsample")
    plt.subplot(2,3,1)
    plt.imshow(img_dwn / img_dwn.max())
    
    print("QC :", out_fn)
    plt.subplot(2,3,2)
    plt.title("Smoothed")
    plt.imshow(img_smoothed)

    plt.subplot(2,3,3)
    plt.title("Segmentation")
    plt.imshow(im_thr)

    plt.subplot(2,3,4)
    plt.title("Labels")
    plt.imshow(cc)

    plt.subplot(2,3,5)
    plt.title("Bounding Box")
    plt.imshow(bb)

    if np.sum(cropped_dwn) != 0 :
        plt.subplot(2,3,6)
        plt.title("Cropped")
        plt.imshow(cropped_dwn/cropped_dwn.max())
    
    plt.savefig(out_fn, figsize=(20, 20) )
    
    if manual_check :
        plt.show()
        # Create figure and axes
        # Display the image
        #plt.imshow(qc,cmap='gray' )
        #plt.show()
        while True :
            response = input("Does image pass QC? (y/n)")
            if response == "y" : return 0
            elif response == "n" : return 1
            print("Click and drag with left mouse to draw rectangle.")
            print("Warning: can crash if you don't start rectangle from top left corner!")
            print("If the 'r' key is pressed, reset the cropping region.")
            print("If the 'c' key is pressed, break from the loop.")
    else : plt.clf()

    return 0

def crop_gui(subject_output_base, img_dwn, img, qc_fn):
    refPts = click(subject_output_base+"img_dwn.jpg")
    try :
        x0=refPts[0][0]
        y0=refPts[0][1]
        x1=refPts[1][0]
        y1=refPts[1][1]
        manual_crop_dwn = np.zeros(img_dwn.shape)
        manual_crop_dwn[y0:y1,x0:x1] = 1
        #cropped = manual_crop * img
        cropped, cropped_dwn, bounding_box = crop(img, img_dwn, manual_crop_dwn )
        
        scipy.misc.imsave(qc_fn, cropped_dwn)
    except TypeError :
        return [], []
    return cropped, cropped_dwn, bounding_box


def crop_source_files(source_files, lines_files, output_dir, downsample_step=0.5, clobber=False, manual_check=False, manual_only=False, no_frame=False) :
    qc_status=0
    for f in source_files :
        #Set output filename
        fsplit = os.path.splitext(os.path.basename(f))
        qc_dir = output_dir + os.sep + "qc" + os.sep 
        subject_output_dir = output_dir + os.sep + "detailed" + os.sep + fsplit[0] + os.sep
        subject_output_base = subject_output_dir+fsplit[0]+'_'
        fout = output_dir + os.sep + fsplit[0]+'_cropped'+fsplit[1]
        fout_mnc = output_dir + os.sep + fsplit[0]+'_cropped'+'.mnc'
        qc_fn=qc_dir+os.sep+fsplit[0]+'_cropped_qc.png'
        bounding_box_fn=subject_output_base + 'bounding_box.png' 
        if (not os.path.exists(fout) or not os.path.exists(qc_fn)) or clobber :
            line_fn = [ i for i in line_files if fsplit[0] in i  ] 
            if line_fn != [] : line_fn = line_fn[0]
            else : exit(0)
            print("Input:", f)
            print("Line:", line_fn)
            if not os.path.exists(subject_output_dir) : os.makedirs(subject_output_dir)
            if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

            #Load image and convert from rgb to grayscale
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = rgb2gray(img)
            
            iline = imageio.imread(line_fn)
            iline=binary_dilation(iline,iterations=3).astype(int)

            low_contrast=False
            low_contrast_ligands=["oxot", "epib", "ampa", "uk14", "mk80" ]
            for ligand in low_contrast_ligands :
                if ligand in fsplit[0] :
                    low_contrast=True
                    break

            #Downsampled the image to make processing faster
            img_dwn = downsample(img, subject_output_base+"img_dwn.jpg", downsample_step)
            iline_dwn = downsample(iline, "", downsample_step)

            if not manual_only :
                bounding_box_dwn, img_smoothed,  im_thr, cc = cropp_img(img_dwn,iline_dwn,no_frame,low_contrast=low_contrast)
                #Crop image
                cropped, cropped_dwn, bounding_box = crop(img, img_dwn, bounding_box_dwn)

                #Quality Control
                qc_status = save_qc(img_dwn, img_smoothed, im_thr, cc, bounding_box_dwn,  cropped_dwn, qc_fn, manual_check)

            if qc_status != 0 or manual_only : 
                #Automated cropping failed to pass QC, use manual QC
                cropped, cropped_dwn, bounding_box = crop_gui(subject_output_base,img_dwn,img,qc_fn)
            
            if cropped != [] :
                print("Cropped:", fout,"\n")
                scipy.misc.imsave(fout, cropped)
                scipy.misc.imsave(bounding_box_fn, bounding_box)
                #if os.path.exists(fout_mnc) : 
                #    os.remove(fout_mnc)
                #    slices2mnc(fout, fout_mnc)
            else : 
                print("No cropped image to save")

        #if not os.path.exists(fout_mnc) or clobber :
        #        slices2mnc(fout, fout_mnc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir',  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--step', dest='downsample_step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--manual', dest='manual_check', action='store_true', default=False, help='Do QC and manually crop region if automated method fails')
    parser.add_argument('--manual-only', dest='manual_only', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--no-frame', dest='no_frame', action='store_true', default=False, help='No frames in line images.')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : os.makedirs(args.output_dir)
    source_files = glob(args.source_dir+os.sep+"final"+os.sep+"*"+args.ext)
    line_files = glob(args.source_dir+os.sep+"lines"+os.sep+"*"+args.ext)
    if source_files == [] : print("Warning: Could not find source files in ", args.source_dir+os.sep+"*"+args.ext )
    crop_source_files(source_files, line_files, args.output_dir, downsample_step=args.downsample_step, clobber=args.clobber, manual_check=args.manual_check, manual_only=args.manual_only, no_frame=args.no_frame)

