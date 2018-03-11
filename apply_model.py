from glob import glob
import os
from utils.utils import downsample
from train_model import dice
import imageio
import numpy as np
from scipy.misc import imresize 
import matplotlib.pyplot as plt


def get_max_model(train_output_dir):
    model_list_str = train_output_dir+os.sep+"model_checkpoint-*hdf5"
    model_list = glob(model_list_str)
    if model_list == [] :
        print("Could not find any saved models in:", model_list_str )
        exit(1)
    models = [ os.path.basename(f).split('.')[-2] for f in model_list]
    max_model_idx = models.index(max(models))
    max_model = model_list[max_model_idx]
    return max_model

def get_raw_files(raw_source_dir):
    raw_files_str = raw_source_dir +  os.sep + "*.tif"
    raw_files = glob(raw_files_str)
    if raw_files == [] :
        print("Could not find any files in: ", raw_files_str)
        exit(1)
    return(raw_files)

def downsample_raw(raw_files, output_dir, step, clobber) :
    downsample_files=[]

    dwn_dir=output_dir + os.sep + 'downsampled'
    if not os.path.exists(dwn_dir) : os.makedirs(dwn_dir)

    for f in raw_files :
        f2_split = os.path.splitext(os.path.basename(f))
        f2=dwn_dir +os.sep+f2_split[0]+'.png'
        if not os.path.exists(f2) or clobber :
            img = imageio.imread(f)
            if len(img.shape) == 3 : img = np.mean(img, axis=2)
            img = downsample(img, step=step, interp='cubic')
            imageio.imsave(f2,img)
        downsample_files += [f2]

    return downsample_files

def get_lines(downsample_files,raw_files, max_model,output_dir, clobber) :

    print("Using model located in:", max_model) 
   
    line_dir=output_dir + os.sep + 'lines'
    if not os.path.exists(line_dir) : os.makedirs(line_dir)
    line_files=[]
    img0 = imageio.imread(raw_files[0])
    if len(img0.shape) == 3 : img0 = np.mean(img0, axis=2)
    y0=img0.shape[0]
    x0=img0.shape[1]

    for f in downsample_files :
        line_fn_split = os.path.splitext(os.path.basename(f))
        line_fn=line_dir +os.sep+line_fn_split[0]+'.png'
        if not os.path.exists(line_fn) or clobber: 
            try : #Load keras if has not been loaded yet 
                keras
            except NameError :
                from keras.models import load_model
                import keras.metrics
                keras.metrics.dice = dice
                model = load_model(max_model )

            img=imageio.imread(f)
            ydim=img.shape[0]
            xdim=img.shape[1]
            img=img.reshape([1,ydim,xdim,1])
            X = model.predict(img, batch_size=1)
            idx = X > np.max(X) * 0.5
            X[ idx ]  = 1
            X[ ~idx ] = 0

            X=X.reshape(ydim,xdim)
            X2 = imresize(X, (y0,x0), interp='nearest')
            X2=X2.reshape(y0,x0)
            
            imageio.imsave(line_fn, X2)
        line_files += [line_fn]

    return line_files


def fill(iraw, iline) :

    m=np.max(iline)
    xx, yy = np.meshgrid(range(iraw.shape[1]), range(iraw.shape[0]))
    xx=xx.reshape(-1)
    yy=yy.reshape(-1)
    iraw_vtr = iraw.reshape(-1)
    iline_vtr = iline.reshape(-1)

    idx0=np.zeros(xx.shape).astype(bool)
    idx1=np.zeros(xx.shape).astype(bool)
    idx2=np.zeros(xx.shape).astype(bool)
    idx3=np.zeros(xx.shape).astype(bool)
    span=8
    while np.sum(iline) > 0 : 


        print(np.sum(iline))
        y0 = yy - span
        y0[y0 < 0] = 0

        x0 = xx - span
        x0[x0 < 0] = 0
        
        y1=yy+span
        y1[y1 >= iraw.shape[0]] = iraw.shape[0]-1
        
        x1=xx+span
        x1[x1 >= iraw.shape[1]] = iraw.shape[1]-1
        
        idx0[:]=False
        idx1[:]=False
        idx2[:]=False
        idx3[:]=False
        inside=iline[yy,xx] == m
        idx0[(iline[ y0, xx ]  < m ) & inside  ] =True
        idx1[(iline[ y1, xx ]  < m ) & inside ] =True
        idx2[(iline[ yy, x0 ]  < m ) & inside ] =True
        idx3[(iline[ yy, x1 ]  < m ) & inside ] =True

        n = idx0.astype(int) +  idx1.astype(int) + idx2.astype(int) + idx3.astype(int)
        
        i = n == 0
        
        iline[~i.reshape(iraw.shape)]=0
        
        n[~i]= n[~i].astype(float)
        n[i]=1
        iraw[yy[~i],xx[~i]]=0
        temp=np.copy(iraw)

        temp[yy[idx0],xx[idx0]] += iraw[y0[idx0], xx[idx0]]   
        temp[yy[idx1],xx[idx1]] += iraw[y1[idx1], xx[idx1]]   
        temp[yy[idx2],xx[idx2]] += iraw[yy[idx2], x0[idx2]]   
        temp[yy[idx3],xx[idx3]] += iraw[yy[idx3], x1[idx3]] 

        temp /= n.reshape(iraw.shape)
        iraw=temp
        span *= 2

    return iraw


from scipy.ndimage.morphology import binary_dilation, binary_erosion
def remove_lines(line_files, raw_files, raw_output_dir, clobber) :
    final_dir=raw_output_dir + os.sep + 'final'
    if not os.path.exists(final_dir) : os.makedirs(final_dir)

    for raw in raw_files :
        base = os.path.splitext(os.path.basename(raw))[0]
        fout = final_dir + os.sep + base + '.png'
        if not os.path.exists(fout) or clobber : 
            lines= [ f  for f in line_files if base in f ]
            if lines != [] : lines=lines[0]
            else : 
                print("failed at remove_lines"); 
                exit(1)

            iraw = imageio.imread(raw)
            if len(iraw.shape) == 3 : iraw = np.mean(iraw, axis=2)

            iline = imageio.imread(lines)
            iline=binary_dilation(iline,iterations=3).astype(int)
            iraw=fill(iraw, iline)
            imageio.imsave(fout, iraw)

    return 0

def apply_model(train_output_dir, raw_source_dir, raw_output_dir, step, clobber):



    max_model=get_max_model(train_output_dir)
    raw_files=get_raw_files(raw_source_dir)  
    print("Got raw file names.")
    downsample_files = downsample_raw(raw_files, raw_output_dir, step, clobber)
    print("Got downsampled files.")
    line_files = get_lines(downsample_files, raw_files,max_model, raw_output_dir, clobber)
    print("Loaded line files.")
    remove_lines(line_files, raw_files, raw_output_dir, clobber)
    print("Removed lines from raw files.")


    return 0
