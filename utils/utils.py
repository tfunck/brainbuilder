import numpy as np
import bisect
from scipy.ndimage.filters import gaussian_filter
import scipy
import os
import pandas as pd
from os.path import basename
import re

def downsample(img, subject_fn="", step=0.1, interp='cubic'):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * 0.02 
    l1 = img.shape[1] * 0.02
    #Calculate the length for the downsampled image
    dim0=int(np.ceil(l0 / step))
    dim1=int(np.ceil(l1 / step))
    #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
    sd0 = step / 2.634 
    sd1 = step / 2.634 
    #Gaussian filter
    img_blr = gaussian_filter(img, sigma=[sd0, sd1])
    #Downsample
    img_dwn = scipy.misc.imresize(img_blr,size=(dim0, dim1),interp=interp )
    if subject_fn != "" : 
        print("Downsampled:", subject_fn +'img_dwn.jpg')
        scipy.misc.imsave(subject_fn +'img_dwn.jpg', img_dwn)
    return(img_dwn)

#def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def rgb2gray(rgb): return np.mean(rgb, axis=2)

def find_min_max(seg):

    fmin = lambda a : min(np.array(range(len(a)))[a == 1])
    fmax = lambda a : max(np.array(range(len(a)))[a == 1])
    xmax = [ fmax(seg[i,:])  for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0   ]
    xmin = [ fmin(seg[i,:])  for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0   ]
    y_i = [ i for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0  ]

    ymax = [ fmax(seg[:,i])   for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    ymin = [ fmin(seg[:,i])   for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    x_i =  [ i for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    return ymin,ymax,xmin,xmax,y_i,x_i

def extrema(labels) :
    xx, yy = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    xxVals = (xx[labels > 0])
    yyVals = (yy[labels > 0])
    y0 = min(yyVals)
    y1 = max(yyVals) 
    x0 = min(xxVals)
    x1 = max(xxVals)
    return([x0,y0,x1,y1])


def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        #hist = np.zeros(256, dtype=np.int)
        #for r in range(src.shape[0]):
        #    for c in range(src.shape[1]):
        #        print(src[r,c])
        #        hist[src[r,c]] += 1
        # Cumulative histogram
        hist, widths = np.histogram( src.reshape(-1,1), 256 )
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    den=1
    if vin[1] != vin[0] : den =  (vin[1] - vin[0])
    scale = (vout[1] - vout[0]) / den
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

#def split_filename(df):
#    dfout=df.copy()
#    dfout.index = dfout.index.map(lambda x : re.sub(r"([0-9])s([0-9])", r"\1#\2", x, flags=re.IGNORECASE))
#    ar=list(map(lambda x: re.split('#|\.|\/',basename(x)), dfout.index.values))
#    df0=pd.DataFrame(ar, columns=["a","b","mri","slab","hemisphere","ligand","sheat","repeat","ext"])
#    df0.index=dfout.index
#    dfout=pd.concat([df0, dfout], axis=1)
#    return dfout

def sort_slices(df, slice_order) :
    df["order"]=[0]*df.shape[0]
    for fn, df0 in df.groupby("filename") :
        print(fn)
        for fn2, df2 in slice_order.groupby("name"):
            #print(df0.mri.values[0], df0.ligand.values[0],  df0.sheet.values[0],  df0.repeat.values[0] )
            if not df0.mri.values[0] in df2["name"].values[0] : continue
            if not df0.ligand.values[0] in df2["name"].values[0] : continue
            if not df0.sheet.values[0] in df2["name"].values[0] : continue
            if not df0.repeat.values[0] in df2["name"].values[0] : continue
            df["order"].loc[ df.filename == fn] = df2["number"].values[0]
            #print(fn , fn2  )
            #print(df2["number"].values[0])
            #print(df["order"].loc[ df.filename == fn])

    df.sort_values(by=["order"], inplace=True)
    print(df)
    return df

def set_csv(source_files, output_dir, include_str, slice_order_fn="", out_fn="receptor_slices.csv", clobber=True):
    include_list = include_str.split(',')
    df_list=[] 
    if os.path.exists(slice_order_fn) : slice_order = pd.read_csv(slice_order_fn)
    if not os.path.exists(out_fn) or clobber:
        cols=["filename", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        df=pd.DataFrame(columns=cols)
        for f0 in source_files:

            f=re.sub(r"([0-9])s([0-9])", r"\1#\2", f0, flags=re.IGNORECASE)
            ar=re.split('#|\.|\/|\_',basename(f))
            ar=[[f0]+ar]

            df0=pd.DataFrame(ar, columns=cols)
            df_list.append(df0)
            #f_split = basename(f).split("#")
            #if "s" in f_split[2] : sep="s"
            #else : sep="S"

            #mr = f_split[2].split(sep)[0]
            #slab = f_split[2].split(sep)[1]

            #hemi = f_split[3]
            #tracer = f_split[4]
            ligand = df0.ligand[0]
            if not include_str == '': 
                if not ligand in include_list : continue

            #n = f_split[6].split("_")[0]
            #df = df.append( pd.DataFrame([[f0, mr, slab, hemi, tracer,n]], columns=cols))
        df=pd.concat(df_list)
        if os.path.exists(slice_order_fn)  :
            df = sort_slices(df, slice_order )
        else : df.sort_values(by=["mri","slab","ligand","sheet","repeat"], inplace=True)

        if output_dir != "" : df.to_csv(output_dir+os.sep+out_fn)
    else :
        df = pd.read_csv(output_dir+os.sep+out_fn)

    return(df)

