from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import numpy as np
import scipy.misc
import scipy.stats
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
from glob import glob
from sys import exit, argv
import os
from os.path import basename
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import argparse
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from skimage.filters import frangi, hessian
from skimage import feature
from cropp_img import get_connected_regions
from sklearn.decomposition import PCA
import shutil
from utils.lines import get_lines
from skimage.filters import threshold_otsu, threshold_yen, threshold_li
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def aggregate_df(df, output_dir) :
    df = pd.melt(df, id_vars=["a","b","mri","slab","hemisphere","tracer","globalNum","sliceNum","number"] ,  value_vars=["entropy","curvature","kurtosis","skew","mean","median", "20p","sd"] )
    df_std = df.groupby(["a","b","mri","slab","hemisphere","tracer","variable"])["value"].std()
    df_mean = df.groupby(["a","b","mri","slab","hemisphere","tracer","variable"])["value"].std()

    df_mean.name = "mean"
    df_std.name = "std"
    df2 = pd.DataFrame({"std":df_std, "mean":df_mean})
    df2.reset_index(inplace=True)
    return(df2)

def plot_stats(df, output_dir) :
    plt.clf()
    grid = sns.FacetGrid(df, col="variable", row="hemisphere", hue="tracer", size=1.5)
    grid.map(plt.plot, "slab", "mean")
    plt.savefig(output_dir + os.sep + "plot_pca.jpg")    
    plt.clf()
    return 0
    

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

p20 = lambda img : np.percentile(img, [20])
cc_sum = lambda img :np.sum(get_connected_regions(img)[0])

global measures
measures = ["entropy","kurtosis","skew","mean","median", "p20", "cc_size","otsu","yen","li","averageLines"]
flist={"entropy":scipy.stats.entropy,"kurtosis":scipy.stats.kurtosis,"skew":scipy.stats.skew,"mean":np.mean,"median":np.median, "p20":p20, "cc_size":cc_sum,"otsu":threshold_otsu,"yen":threshold_yen,"li":threshold_li, "averageLines":get_lines}
histogram_methods=["entropy","kurtosis","skew"]
def get_image_stats(source_files, output_dir, clobber=False) :
    pd_list = []
    col=["file", "file_dwn"] + measures
    i=0
    png_dir = os.getcwd() + os.sep + output_dir + os.sep + "png"
    if not os.path.exists(png_dir): os.makedirs(png_dir)
    n=len(source_files)
    source_files = np.array(source_files)
    for f in source_files :
        f_base = os.path.basename(f)
        fsplit = os.path.splitext(f_base)
        f_dwn = png_dir +os.sep+fsplit[0]+"_dwn.png" 

        if not os.path.exists(f_dwn) or clobber:
            img = downsample(rgb2gray(imageio.imread(f)))
            scipy.misc.imsave(f_dwn, img)
        else :
            img = rgb2gray(imageio.imread(f_dwn))
        
        val, bin_width  = np.histogram(img.reshape(-1,1), 255)
        otsu = threshold_otsu(img)
        yen = threshold_yen(img) 
        li = threshold_li(img)
        cc_size = cc_sum(img)
        e = scipy.stats.entropy(val)
        kur = scipy.stats.kurtosis(val)
        ske = scipy.stats.skew(val)
        m0 = np.mean(img)
        m1 = np.median(img)
        p20, = p20(img)
        pd_list.append(pd.DataFrame([(f,f_dwn,e, kur, ske, m0, m1,p20, cc_size, otsu, yen, li)], columns=col))
        i+=1
    df=pd.concat(pd_list)
    df.set_index("file", inplace=True)
    return(df)

def update_df(df, fn):
    for name, f  in  flist.items() :
        if not name in df.columns :
            print("Adding :", name)
            mlist=[]
            for filename in df.file_dwn:
                img = imageio.imread(filename)
                val, bin_width  = np.histogram(img.reshape(-1,1), 255)
                if name in histogram_methods :
                    mlist.append(f(val))
                else :
                    mlist.append(f(img))
            df[name]=mlist
    df.to_csv(fn)
    return df        


def split_filename(df):
    dfout=df.copy()
    dfout.index = dfout.index.map(lambda x : re.sub(r"([0-9])s([0-9])", r"\1#\2", x, flags=re.IGNORECASE))
    ar=list(map(lambda x: re.split('#|\.|\/',basename(x)), dfout.index.values))
    df0=pd.DataFrame(ar, columns=["a","b","mri","slab","hemisphere","tracer","globalNum","sliceNum","ext"])
    df0.index=dfout.index
    dfout=pd.concat([df0, dfout], axis=1)
    return dfout

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
    pca.fit(df[["skew", "kurtosis", "cc_size", "averageLines"]])
    print("PCA explained variance:", np.sum(pca.explained_variance_ratio_))
    df_pca = pca.transform(df[["skew", "kurtosis", "cc_size", "averageLines"]])

    for i in np.arange(N,dtype=int):
        name = "PC"+str(i)
        df[name]=df_pca[:, i]
    data = pca.inverse_transform( np.eye(N) )
    data = np.mean(data, axis=0)
    temp=pd.DataFrame({"measure":["skew", "kurtosis", "cc_size", "averageLines"],"value":data})
    grid = sns.swarmplot(x="measure", y="value", data=temp)
    plt.savefig(output_dir +os.sep +"feature_contribution.png")
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_pca[:,0],df_pca[:,1],df_pca[:,2])
    plt.savefig(output_dir + os.sep+ 'raw_pca.png')
    plt.clf()
    return df

from sklearn.neighbors import NearestNeighbors
def cluster(df, output_dir,  method="DBSCAN"):
    #X=df.loc[:,["PC0","PC1","PC2"]]
    print(df)
    #X=df.loc[:,["entropy", "skew","median", "kurtosis", "averageLines"]]
    X=df.loc[:,["entropy", "skew", "kurtosis", "cc_size"]]

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    dist = np.sort(distances[:,-1])
    dist = np.delete(dist, dist.argmax())
    dist = gaussian_filter(dist, sigma=len(dist)/250 )
    ddist = np.gradient(dist)
    d2dist = np.gradient(ddist)
    plt.plot(range(len(dist)), dist)
    plt.plot(range(len(dist)), ddist)
    plt.plot(range(len(dist)), d2dist)
    plt.savefig(output_dir+os.sep+'nn_plot.png')
    plt.clf()
    eps = 0.5 # dist[d2dist.argmax()]
    print("EPS:", eps)
    method="KMeans"
    if method == "KMeans" :
        db = KMeans(2).fit_predict(df.loc[:,["averageLines"]])
        df["label"] = db
    
        #for i in range(2):
        #    db = DBSCAN(eps=eps, min_samples=10).fit( X[df["label"]==i ] )
        #    df.loc[ df["label"]==i, "label"] = i*10 + db.labels_
        #    print("Labels", np.unique(db.labels_))
    return df

def distribute_files(df, output_dir):
    for label in df.label.unique() :
        dir_name=output_dir+os.sep+"labels"+os.sep+str(label)
        if not os.path.exists(dir_name) :
            os.makedirs(dir_name)
    
    label_dir=os.getcwd()+os.sep+output_dir+os.sep+"labels"+os.sep
    if os.path.exists(label_dir) : shutil.rmtree(label_dir)
    os.makedirs(label_dir)

    for label, df1  in df.groupby(["label"]) :
        
        target_dir = label_dir+str(label)+os.sep
        if not os.path.exists(target_dir) : os.makedirs(target_dir)

        for name, df0 in df1.iterrows() :
            if not  os.path.exists(name) : continue
            source = os.getcwd() + os.sep + name # df0.index
            fsplit = os.path.splitext(os.path.basename(source))
            target= target_dir + os.sep + fsplit[0] + ".tif"
            if not os.path.exists(target) :
                os.symlink( source, target )



def slab_tracer_anova(df) :
    df.rename( index=str, columns={"20p":"twentyPercentile"}, inplace=True )

    for measure in measures:
        if measure == "20p" : measure = "twentyPercentile"
        formula = measure + ' ~ C(tracer) * C(slab)  '
        model = ols(formula, df).fit()
        aov_table = anova_lm(model, typ=2)


def main(source_files, output_dir, clobber=False) :
    df_fn = output_dir + os.sep + "image_stats.csv"
    
    if not os.path.exists(df_fn) or clobber :
        df = get_image_stats(source_files, output_dir, clobber=clobber)
        df.to_csv(df_fn)
    else :
        df=pd.read_csv(df_fn,index_col="file")
        df=update_df(df, df_fn)
    #df.loc[:,"averageLines"][ pd.isnull(df["averageLines"])  ]=0
    #df.to_csv(df_fn)
    #Calculate zscore for all the image metrics
    dfz = df.copy()
    dfz[measures] = df[measures].apply(scipy.stats.zscore)
    #Apply PCA
    df_pca = get_pca(dfz, output_dir)
    #Apply clustering to DataFrame
    df_labeled=cluster(dfz, output_dir)
    #Turn filenames into columns (MRI, slab, etc.)
    df2 = split_filename(df_labeled)
    #Calculate distribution of labels
    clustering_stats(df2,output_dir) 
    #Distribute files by label group
    distribute_files(df_labeled,output_dir)

    #Aggregate by mean and std for MRI, hemisphere and slab
    dfAgr = aggregate_df(df2, output_dir)
    #Calculate effect of slab and tracer on measures
    #slab_tracer_anova(df2) #FIXME : Doesn't work yet
    #Plot distribution of image statistics
    plot_stats(dfAgr, output_dir) 

    return(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--step', dest='downsample_step', default="1", type=int, help='File extension for input files (default=.tif)')
    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : os.makedirs(args.output_dir)
    
    source_files=[]
    if args.source_dir != None :
        source_files = glob(args.source_dir+"**"+os.sep+"*"+args.ext)
        if source_files == [] : print("Warning: could not find any files")

    main(source_files, args.output_dir, clobber=args.clobber )

