import numpy as np
import bisect
from scipy.ndimage.filters import gaussian_filter
import scipy
import os
import pandas as pd
from os.path import basename
import re
from subprocess import call, Popen, PIPE, STDOUT
import contextlib

newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout'):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == '' and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == '' and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = ''.join(out)
            print(out)
            yield out

def shell(cmd, verbose=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout=""
    if verbose == True:
        print(cmd)


    process=Popen( cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        #if verbose :
        print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print ("Error:")
        print ("Command:", cmd)
        print ("Stderr:", stdout)
    return stdout, stderr, errorcode

def get_z_x_max(source_files):
    zmax=xmax=0
    for i in range(len(source_files)) : 
        f=source_files[i]
        fn = output_dir + os.sep + os.path.basename(f)
        if not os.path.exists(fn) :
            img = imageio.imread(f)
            img = downsample(img, fn,  step=0.2)
        else : 
            img = imageio.imread(fn)
        z=img.shape[1]
        x=img.shape[0]
        if x > xmax : xmax=x 
        if z > zmax : zmax=z 
        source_files[i] = fn

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
        print("Downsampled:", subject_fn )
        scipy.misc.imsave(subject_fn, img_dwn)
    return(img_dwn)

#def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def rgb2gray(rgb): return np.mean(rgb, axis=2)

def find_min_max(seg):
    m=np.max(seg)
    fmin = lambda a : min(np.array(range(len(a)))[a == m])
    fmax = lambda a : max(np.array(range(len(a)))[a == m])
    xmax = [ fmax(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0 ]
    xmin = [ fmin(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0 ]
    y_i = [ i for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0  ]

    ymax = [ fmax(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    ymin = [ fmin(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    x_i =  [ i for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]
    
    if xmin == [] : xmin = [ 0 ]
    if xmax == [] : xmax = [ seg.shape[1] ]
    if  y_i == [] : y_i = [0]

    if ymin == [] : ymin = [ 0 ]
    if ymax == [] : ymax = [ seg.shape[0] ]
    if  x_i == [] : x_i = [0]
    
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

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        #hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
        #hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[ np.round(src[r,c]).astype(int)] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


def split_filename(fn):
    dfout.index = dfout.index.map(lambda x : re.sub(r"([0-9])s([0-9])", r"\1#\2", x, flags=re.IGNORECASE))
    ar=list(map(lambda x: re.split('#|\.|\/',basename(x)), dfout.index.values))
    df0=pd.DataFrame(ar, columns=["mri","slab","hemisphere","ligand","sheat","repeat"])
    df0.index=dfout.index
    dfout=pd.concat([df0, dfout], axis=1)
    return dfout

def sort_slices(df, slice_order) :
    df["order"]=[-1]*df.shape[0]
    for fn2, df2 in slice_order.groupby("name"):
        fn2b = re.sub(r"([0-9])s([0-9])", r"\1#\2", fn2, flags=re.IGNORECASE)
        fn2c = re.split('#|\.|\/|\_',fn2b)
        if len(fn2c) < 2 : continue
        mri=fn2c[2] 
        slab=fn2c[3]
        hemisphere=fn2c[4]
        ligand=fn2c[5]
        sheet=fn2c[6]
        repeat=fn2c[7]
        df["order"].loc[ (mri == df.mri) & (slab == df.slab) & (hemisphere == df.hemisphere) & (ligand == df.ligand) & (sheet == df.sheet) & (repeat == df.repeat ) ] = df2["global_order"].values[0]
    df.sort_values(by=["order"], inplace=True)
    return df



def set_slice_order(slice_order_fn="") :
    slice_order_list=[]
    for i in range(1,3) : #FIXME : Should not be hardcoded
        slice_order_fn = "MR1_R_slab_"+str(i)+"_section_numbers.csv"
        if os.path.exists(slice_order_fn) : 
            df0=pd.read_csv(slice_order_fn)
            df0["slab"]= [i] * df0.shape[0]
            slice_order_list.append(df0)
    slice_order = pd.concat(slice_order_list)
    
    slice_order["global_order"] = slice_order["number"]
    slice_order_unique = np.sort(slice_order["slab"].unique())
    for i in slice_order_unique[1:] :
        prev_slab = i - 1
        prev_slab_max = slice_order["number"].loc[ slice_order["slab"] == prev_slab ].max() 
        slice_order["global_order"].loc[ slice_order["slab"] == i ] += prev_slab_max

    return slice_order

def set_slice_name(source_files, cols, include_str, exclude_str):
    include_list = include_str.split(',')
    exclude_list = exclude_str.split(',')
    df=pd.DataFrame(columns=cols)
    df_list=[] 
    for f0 in source_files:
        f=re.sub(r"([0-9])s([0-9])", r"\1#\2", f0, flags=re.IGNORECASE)
        ar=re.split('#|\.|\/|\_',basename(f))
        
        ar=[[f0]+ar]
        df0=pd.DataFrame(ar, columns=cols)
        ligand = df0.ligand[0]
        if not include_str == '': 
            if not ligand in include_list : continue
        if not exclude_str == '': 
            if ligand in exclude_list : 
                continue
        df_list.append(df0)
        #n = f_split[6].split("_")[0]
        #df = df.append( pd.DataFrame([[f0, mr, slab, hemi, tracer,n]], columns=cols))
    df=pd.concat(df_list)
    return df

def set_slab_border(df) :
    df["border"] = [0] * df.shape[0]
    for slab, df0 in df.groupby("slab") :
        min_order = df0["order"].min() + df0["order"].max() * 0.02
        max_order = df0["order"].max() * 0.98
        df0["border"].loc[ (df0["order"] < min_order ) | ( df0["order"] > max_order)  ] = 1
        df.loc[ df["slab"] == slab ] = df0

    return df
    
def set_csv(source_files, output_dir, include_str, exclude_str, slice_order_fn="", out_fn="receptor_slices.csv", clobber=False):

    if not os.path.exists(output_dir+os.sep+out_fn ) or clobber:
        #Load csv files with order numbers
        cols=["filename", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        slice_order = set_slice_order()
        df = set_slice_name(source_files, cols, include_str, exclude_str)
        df = sort_slices(df, slice_order )
        #df = set_slab_border(df)
        #if os.path.exists(slice_order_fn)  :
        #else : df.sort_values(by=["mri","slab","ligand","sheet","repeat"], inplace=True)

        if output_dir != "" : df.to_csv(output_dir+os.sep+out_fn)


    else :
        df = pd.read_csv(output_dir+os.sep+out_fn)

    return(df)

