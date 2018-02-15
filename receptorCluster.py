from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import numpy as np
import scipy.misc
import scipy.stats
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
from glob import glob
from sys import exit, argv
import os
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import argparse
from mpl_toolkits.mplot3d import Axes3D

def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def curvature(img):
    x, y = np.gradient(img)
    cg =  np.sqrt(x**2 + y**2)
    img=cg
    d0 = np.gradient(img,edge_order=2, axis=0)
    d00 = np.gradient(d0,edge_order=2, axis=0)
    d1 = np.gradient(img,edge_order=2, axis=1)
    d11 = np.gradient(d1,edge_order=2, axis=1)
    d10 = np.gradient(d1,edge_order=2, axis=0)

    num = (d00*d11**2 + d11*d00**2 - 2*d10*d1*d0) 
    den = (d0**2 + d1**2)**(3/2)
    den[den==0]=1
    num[den==0]=0
    k =np.abs(num/ den)
    return(k)

def downsample(img, step=0.1):
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
    img_dwn = scipy.misc.imresize(img_blr,size=(dim0, dim1),interp='cubic' )
    return(img_dwn)

from sklearn.decomposition import PCA

def get_image_stats(source_files) :
    pd_list = []
    col=["file","entropy","curvature","kurtosis","skew","mean","median", "20p","sd"]
    i=0
    for f in source_files :
        print(f)
        img = rgb2gray(imageio.imread(f))
        img = downsample(img)
        val, bin_width  = np.histogram(img.reshape(-1,1), 255)
        e = scipy.stats.entropy(val)
        K = np.mean(curvature(img))
        kur = scipy.stats.kurtosis(val)
        ske = scipy.stats.skew(val)
        m0 = np.mean(img)
        m1 = np.median(img)
        s = np.std(img)
        p20, = np.percentile(img, [20])
        pd_list.append(pd.DataFrame([(f,e, K, kur, ske, m0, m1,p20, s)], columns=col))
        i+=1
    df=pd.concat(pd_list)
    df.set_index("file", inplace=True)
    return(df)

def split_filename(df):
    ar=list(map(lambda x: x.split("#"), df.index.values))
    df0=pd.DataFrame(ar, columns=["a","b","mri","hemisphere","tracer","slice","number"])
    df0.index=df.index
    df=pd.concat([df0, df], axis=1)
    return df

def clustering_stats(df, output_dir):
    df_list=[]
    for names, df0 in df.groupby(["tracer","label"]):
        tracer=names[0]
        label=names[1]
        per = 1. * df0.shape[0] / df.shape[0]
        df_list.append(pd.DataFrame([(label, tracer, per)], columns=["label","tracer","percent"]))
    clusterStats = pd.concat(df_list)
    clusterStats = clusterStats.pivot(index="label", columns='tracer', values='percent')
    clusterStats.fillna(value=0,inplace=True)
    clusterStats["LabelTotal"] = clusterStats.sum(axis=1)
    row = pd.DataFrame(clusterStats.sum(axis=0)).T
    row.index = ['TracerTotal']
    clusterStats = clusterStats.append(row)
    clusterStats.to_csv(output_dir+os.sep+"clusterStats.csv")
    return clusterStats

def get_pca(df, output_dir, N=3) :
    pca = PCA(n_components=N, whiten=True)
    pca.fit(df)
    df_pca = pca.transform(df)
    for i in np.arange(N,dtype=int):
        name = "PC"+str(i)
        df[name]=df_pca[:, i]
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_pca[:,0],df_pca[:,1],df_pca[:,2])
    plt.savefig(output_dir + 'raw_pca.png')
    return df
    
def cluster(df, nMeans=3):
    db = KMeans(nMeans).fit_predict(df.loc[:,["PC0","PC1","PC2"]])
    df["label"] = db
    return df

def distribute_files(df, output_dir):
    for label in df.label.unique() :
        dir_name=output_dir+os.sep+"labels"+os.sep+str(label)
        if not os.path.exists(dir_name) :
            os.makedirs(dir_name)
    for name, df0 in df.iterrows() : 
        source=os.getcwd()+os.sep+name
        label_dir=os.getcwd()+os.sep+output_dir+os.sep+"labels"+os.sep
        target=label_dir+str(int(df0["label"]))+os.sep+os.path.basename(source)
        if not os.path.islink(target) :
            os.symlink(source, target)

def main(source_files, output_dir, nMeans=3, clobber=False) :
    df_fn = output_dir + os.sep + "image_stats.csv"
    
    if not os.path.exists(df_fn) or clobber :
        df = get_image_stats(source_files)
        df.to_csv(df_fn)
    else :
        df=pd.read_csv(df_fn,index_col="file")
    

    df = df.apply(scipy.stats.zscore)
    df=get_pca(df, output_dir)
    df = split_filename(df)

    df=cluster(df, nMeans)
    distribute_files(df,output_dir)
    clustering_stats(df,output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir',  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--step', dest='downsample_step', default="1", type=int, help='File extension for input files (default=.tif)')
    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--nmeans', dest='nMeans', default=3, type=int, help='Number of means for clustering')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : os.makedirs(args.output_dir)
    source_files = glob(args.source_dir+"**"+os.sep+"*"+args.ext)
    if source_files != []:
        main(source_files, args.output_dir, nMeans=args.nMeans, clobber=args.clobber )
    else : print("Warning: could not find any files")
