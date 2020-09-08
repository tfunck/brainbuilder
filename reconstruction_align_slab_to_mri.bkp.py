from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
import json
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import shutil
import re
from glob import glob
from ANTs import ANTs
from utils.utils import shell, splitext
from nibabel.processing import resample_from_to, resample_to_output
from utils.utils import splitext  
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat
from numpy.linalg import det
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('TKAgg')


def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)
def v2w(i, step, start) :
    return start + i * step

def findSRVcoordinates(vol) :
    '''
    Finds the min and max spatial coordinate of the srv image
    '''
    profile = np.sum(vol, axis=(0,2))

    if np.sum(profile) == 0 :
        print("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile>0)[0][0]
    srvMax =  np.argwhere(profile>0)[-1][0]
    del vol
    return srvMin, srvMax

def get_slab_start_end(i, y, slab_ymax, srv_ymax, ratio=0.05):
    #Get slab start and end position in voxel coordinates along y-axis
    
    #                                                srv_ymax
    # srv :   |------------------------------------------|
    #         |----------------| 
    #  y1=y + slab_ymax
    #                                                slab_ymax
    # slab:                             |----------------|
    #

    offset =  ratio * slab_ymax
    start = y 
    end = y + slab_ymax
    start =max( int(start - offset), 0)
    end = min(int(end + offset ), srv_ymax)
    mid = (end+start)/2.
    return start, mid, end 

def registration(outDir,slab, start,end,srv,tfm_prefix, moving_fn,fixed_fn, moving_rsl_prefix, moving_rsl_fn_inverse, srv_img,  args) :
    metric=args.metric
    sampling=args.sampling
    clobber=args.clobber

    if not metric in ['CC', 'MI', 'Mattes', 'Demons', 'GC'] : 
        print("Error: Metric",metric,"not found")
        exit(1)

    ### Extract fixed image from MRI GM Classification SRV and adjust start location along y axis
    fixed = srv[:, start:end, :]

    affine =np.copy( srv_img.affine)
    affine[1,3] = affine[1,3] + affine[1,1] * start
    if not os.path.exists(fixed_fn) or clobber > 0 :
        print("Writing fixed ", fixed_fn)
        print(start, end, affine[1,:])
        nib.Nifti1Image( fixed, affine  ).to_filename(fixed_fn)
    
    #Works
    iterations = ['50x25','50x25'] 
    shrink_factors = ['2x1','2x1']
    smoothing_sigmas = ['1x0','1x0']
    metrics = ['GC', 'GC' ]
    tfm_types = ['Rigid','Affine']
    rate = ['0.1','0.05']
    
    moving_rsl_fn = outDir+os.sep+'cls_moving_rsl_'+str(slab)+'_'+str(start)+'_'+str(end) +'_level-1_GC_Affine.nii.gz'

    tfm_syn = outDir+os.sep+'affine_'+str(slab)+'_'+str(start)+'_'+str(end) +'_level-1_GC_Affine_Composite.h5'
    if not os.path.exists(tfm_syn)  or args.clobber :
        print("Finding:", tfm_syn, moving_rsl_fn)
        tfm_syn, moving_rsl_fn = ANTs( tfm_prefix, fixed_fn, moving_fn, moving_rsl_prefix, iterations=iterations, tfm_type=tfm_types, rate=rate, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas, metrics=metrics, verbose=1, clobber=clobber,  exit_on_failure=True, generate_masks=False)
    else : 
        print("Skipping")

    if not os.path.exists(moving_rsl_fn) or args.clobber :
        shell(f'antsApplyTransforms -d 3 -i {moving_fn} -r {fixed_fn} -t {tfm_syn} -o {moving_rsl_fn}')

    del fixed
    del srv
    return tfm_syn, moving_rsl_fn

def calc_distance_metric(fixed_fn, moving_rsl_fn, args):
    metric_val=0
    if  os.path.exists(fixed_fn) and os.path.exists(moving_rsl_fn) :
        if args.metric == "GC" :
            try :
                print('fixed_fn',fixed_fn) 
                print('moving_rsl_fn', moving_rsl_fn)
                fixed  = ants.from_numpy( nib.load(fixed_fn).get_fdata())
                moving = ants.from_numpy( nib.load(moving_rsl_fn).get_fdata())
                ants_metric = ants.create_ants_metric(fixed,moving, metric_type='Correlation')
                metric_val = ants_metric.get_value()
                del ants_metric
                del fixed
                del moving
            except RuntimeError :
                pass
    
    return  metric_val

def load_srv_image(srv_fn, outDir, step, clobber=0) :
    split = splitext(os.path.basename(srv_fn))
    srv_img_fn = outDir + os.sep +  split[0] + '_'+str(step)+split[1]
    print('SRV downsampled :', srv_img_fn)
    if not os.path.exists(srv_img_fn)  or clobber > 2 :
        srvImg = nib.load(srv_fn)

        srv_img = resample_to_output(srvImg, [step,step,step], order=1)
        srv_img.to_filename(srv_img_fn)
    else : 
        srv_img = nib.load(srv_img_fn)

    return srv_img

def get_slab_image(cls_fn, moving_fn, outDir, step, clobber=0) :
    if not os.path.exists(moving_fn) or clobber > 2:
        slabImg = nib.load(cls_fn)
        vol=slabImg.get_fdata()
        slabRsl = resample_to_output(nib.Nifti2Image(vol,slabImg.affine), [step,step,step], order=2)
        print('Resampling', cls_fn, 'to dimensions', slabRsl.affine )
        del slabImg
        del vol
        slab = slabRsl.get_data()
        nib.Nifti1Image( slab, slabRsl.affine ).to_filename(moving_fn)
    else : 
        slabRsl = nib.load(moving_fn)
    print('Using moving image:', moving_fn)
    return slabRsl

def get_cls_fn(i,cls_base_fn,desc) :
    cls_fn = cls_base_fn.format(*desc,i,*desc,i)
    if not os.path.exists(cls_fn) :
        print('Error : could not find cls slab for ', cls_fn)
        exit(1)
    print('Read',cls_fn)
    return cls_fn

def calculate_prior_distribution(cls_base_fn, srvMin, srvMax, nSlabs, srv_ymax,slab_offset_ratio, outDir, step,slab_shift_width,desc=(0,0), clobber=0):
    distanceList=[]
    slabShape=[]
    srvWidth = srvMax - srvMin
    slabPositionPriorList=[]
    slabPositionPrior={}
    for i in range(1,nSlabs+1) : 
        moving_fn = outDir+os.sep+'orig_'+str(i)+'.nii.gz'
        cls_fn = get_cls_fn(i, cls_base_fn, desc)
        slabRsl = get_slab_image(cls_fn, moving_fn, outDir, step, clobber=2)    
        slab = slabRsl.get_data()
        slabShape.append( slab.shape[1] )
   
    #total slab width
    totalWidth = np.sum(slabShape) 

    # width differece = difference between total width of slabs and that of the mri
    widthDiff = (srvWidth-totalWidth)/(nSlabs-1)
    prob_arrays=[]
    for i in range(nSlabs) :
        s = i+1
        prevWidth=0
        if s > 1 : prevWidth = np.sum(slabShape[(i-1)::-1])
        
        #
        #   6    5    4    3    2    1
        # |----|----|----|----|----|----|
        #                              srvMax
        
        slabPositionPriorList += [ np.rint(srvMax - i*widthDiff - slabShape[i]/2.0 - prevWidth).astype(int)  ]
    return slabPositionPriorList

def update_df(slab_df, best_df, df, out, srv_fn, metric="nmi") :
    print(slab_df)
    idx = slab_df["MetricValue"].loc[ slab_df["Metric"]==metric ].idxmax()
    best_df =  best_df.append( slab_df.iloc[idx,] )
    df = df.append(slab_df)
    df.to_csv(out.csv_fn)
    return slab_df, best_df, df

def update_srv(i,best_df, srv, srv_img, outDir, metric) :
    srv_affine = srv_img.affine

    moving_rsl_fn = best_df["moving_rsl"].loc[ best_df["slab"]==i ].values[0]
    moving_rsl_img = nib.load(moving_rsl_fn)
    moving_rsl_img = resample_from_to(moving_rsl_img, srv_img)
    moving_rsl_vol = moving_rsl_img.get_fdata()
    srv[ moving_rsl_vol >0] =0
    nib.Nifti1Image( srv,srv_affine).to_filename(f'{outDir}/updated_srv_{i}.nii.gz') 
    return srv

def save_best_alignments(i, best_df, srv_fn, out):
    #find moving file
    moving_rsl_fn = best_df["moving_rsl"].loc[ best_df["slab"] == i , ].values[0]

    #find fixed file
    fixed_fn =  best_df["fixed"].loc[ best_df["slab"] == i ,   ].values[0]

    movingRslImg = nib.load(moving_rsl_fn)

    srvImgLoRes = nib.load(fixed_fn)
    srvImgHiRes = nib.load(srv_fn)

    yLoStep  = srvImgLoRes.affine[1,1]
    yLoStart = srvImgLoRes.affine[1,3]

    yHiStep  = srvImgHiRes.affine[1,1]
    yHiStart = srvImgHiRes.affine[1,3]

    y0 = best_df["y"].loc[ best_df["slab"] == i , ].values[0]
    y1 = best_df["y_end"].loc[ best_df["slab"] == i , ].values[0]

    y0High = np.rint( (  yLoStep * y0 ) / yHiStep).astype(int)
    y1High = np.rint( (  yLoStep * y1 ) / yHiStep).astype(int)
    srv = srvImgHiRes.get_data() 

    outVol = srv[ : , y0High : y1High , : ]

    out_fn = out.outDirFinal + os.sep + os.path.basename( fixed_fn  )
    out_moving_rsl_fn = out.outDirFinal + os.sep + os.path.basename( moving_rsl_fn  )
    tfm_fn = best_df['tfm'].loc[ best_df['slab'] == i ].values[0]
    tfm_final_fn = out.outDirFinal + os.sep + os.path.basename( tfm_fn  )

    shutil.copy( moving_rsl_fn, out_moving_rsl_fn) 
    
    shutil.copy( tfm_fn , tfm_final_fn) 

    aff = srvImgHiRes.affine
    aff[1,3] = srvImgHiRes.affine[1,3] + srvImgLoRes.affine[1,1] * y0
    nib.Nifti1Image(outVol, aff ).to_filename( out_fn )

    best_df["fixed"].loc[ best_df["slab"] == i ,] = out_fn
    del outVol
    del srv
    return best_df

def align_single_slab(srv_fn, cls_fn, slab, args, srv, srv_img, slabPositionPriorList, df, best_df, out) :
    moving_fn = args.outDir+os.sep+'orig_'+str(slab)+'.nii.gz'
    
    slabRsl = get_slab_image(cls_fn, moving_fn, args.outDir, args.step, clobber=args.clobber)    
    slabVol = slabRsl.get_data()
    srvMin, srvMax = findSRVcoordinates(srv)
    slab_df=pd.DataFrame({})
    df_fn='{}/{}.csv'.format(args.outDir, slab)

    for y in range(srvMin, srvMax, args.slab_shift_width) : 
        start, mid, end = get_slab_start_end(slab, y, slabVol.shape[1], srv_img.shape[1], args.slab_offset_ratio)
        if start < srvMin - args.slab_offset_ratio :
            continue

        if end > srvMax + args.slab_offset_ratio :
            continue
        
        if np.abs(mid - slabPositionPriorList[slab-1]) > slabVol.shape[1]/2 :
            continue

        fixed_fn = out.outDir+os.sep+'srv_fixed_'+str(slab)+'_'+str( start )+'_'+str( end )+'.nii.gz'
        moving_rsl_prefix = out.outDir+os.sep+'cls_moving_rsl_'+str(slab)+'_'+str(start)+'_'+str(end) #+'.nii.gz'

        moving_rsl_fn_inverse = out.outDir+os.sep+'srv_fixed_rsl_'+str(slab)+'_'+str(start)+'_'+str(end)+'_level-1_GC_Affine_Composite.nii.gz'
        
        tfm_prefix = out.outDir + os.sep+'affine_'+str(slab)+'_'+str(start)+"_"+str(end)+"_"
        tfm_fn, moving_rsl_fn = registration(out.outDir, slab, start,end,srv, tfm_prefix, moving_fn, fixed_fn, moving_rsl_prefix, moving_rsl_fn_inverse, srv_img, args )
        metric_val = calc_distance_metric(fixed_fn, moving_rsl_fn, args)

        row_dict = {
                "slab":[slab], 
                "y":[start],
                "y_end":[end], 
                "MetricValue": [-metric_val],
                "Metric":[args.metric],
                "tfm":[tfm_fn],
                "fixed":[fixed_fn], 
                "moving_rsl":[moving_rsl_fn],
                "RegLevels":[len(args.iterations)],
                "Offset":[args.slab_offset_ratio],
                "RegSchedule":[ '_'.join(args.iterations) ]
                }
        
        row = pd.DataFrame(row_dict)
        slab_df = slab_df.append(row, ignore_index=True)
    
    slab_df, best_df, df =  update_df(slab_df, best_df, df, out, srv_fn, metric=args.metric)

    srv_new = update_srv(slab, best_df, srv, srv_img, out.outDir, args.metric) 
    del srv
    del srv_img
    del slabVol
    return df, best_df, srv_new



def save_plot(csv_fn, out_fn,  out, args) :
            
    df = pd.read_csv(csv_fn)
    
    rate_str = '-'.join([str(i) for i in args.rate ])
    itr_str = '-'.join(args.iterations)
    
    #df = pd.melt(df, id_vars=["slab", "y","y_end"], value_vars=[args.metric,"p",args.metric+"_p" ]) 
    g = sns.FacetGrid(df, row="slab",col="Metric", hue="slab",sharey=False,sharex=True)
    g = g.map(plt.plot, "y", "MetricValue")
    [plt.setp(ax.texts, text="") for ax in g.axes.flat] # remove the original texts
                                                    # important to add this before setting titles
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.fig.suptitle(" ".join(["metric :", args.metric, "rate :", rate_str, "itr :", itr_str,"offset :", str(args.slab_offset_ratio)]), y=1, fontsize=8)
    print("Writing plot to", out_fn)
    plt.savefig(out_fn)

def adjust_slab_list(ll) :
    ll.sort()
    oo=[]
    for i in range(len(ll)):
        j = int(i/2)
        if i % 2 :
            oo.append(ll[ -(j+1)]  )
        else :
            oo.append(ll[j])
    return oo

class OutputFilenames():
    def __init__(self, args) :
        self.outDir = args.outDir

        self.outDirFinal = args.outDir +  os.sep + 'final' + os.sep

        self.csv_fn = self.outDir + os.sep + 'slab_position.csv'
        self.best_csv_fn =  self.outDirFinal+os.sep+'best_slab_position.csv'

        if not os.path.exists(self.outDir) :
            os.makedirs(self.outDir)

        if not os.path.exists(self.outDirFinal) :
            os.makedirs(self.outDirFinal)

    def _gen_output_label(self, args) :
        rate_str = '-'.join([str(i) for i in args.rate ])
        itr_str = '-'.join(args.iterations)
        return "nbins-"+str(args.nbins)+"_metric-"+args.metric+"_rate-"+rate_str+"_itr-"+itr_str+"_offset-"+str(args.slab_offset_ratio)

def align_slabs( args, cls_base_fn="/2_segment/{}_{}_{}/brain-{}_hemi-{}_slab-{}_seg.nii.gz", srv_fn = "srv/mri1_gm_bg_srv.nii.gz", scale_factors_fn="data/scale_factors.json" ):
    cls_base_fn = args.inDir + os.sep + cls_base_fn

    #Get list of segmented (seg) slab volumes
    args.slab_fn_list = glob(cls_base_fn.format(args.brain,args.hemi,'*',args.brain,args.hemi,'*' ) )
    args.slab_fn_list.sort()
    print(args.slab_fn_list)
     
    #Get slab number of each slab
    args.slab_n_list = [ int(x.split('_')[2].split('-')[1]) for fn in args.slab_fn_list for x in fn.split('/') if 'slab-' in x  ] 
    print("Found slabs:", args.slab_fn_list)

    #Change order of slabs so that it jumps from smallest to largest slab number. 
    args.slab_n_list = adjust_slab_list( args.slab_n_list )
    print('Slab processing order:', args.slab_n_list)
    
    # Set optimization step rate
    args.rate = [0.1, 0.1, 0.1]
    for i in range(len(args.user_rate)) :
        args.rate[i] = args.user_rate[i]

    out = OutputFilenames(args)
    nSlabs = len(args.slab_fn_list)
    largestSlabN = int(max(args.slab_n_list))

    desc = (args.brain, args.hemi)

    if not os.path.exists(args.outDir) :
        os.makedirs(args.outDir)

    if not os.path.exists(out.best_csv_fn) or args.clobber > 0 : 
        print('Generate :', out.best_csv_fn)
        #Load SRV MRI GM mask
        srv_img = load_srv_image(srv_fn,args.outDir,args.step, args.clobber)
        srv = srv_img.get_data()

        #Calculate prior distribution to weight metrics 
        
        srvMin, srvMax = findSRVcoordinates(srv)
         
        slabPositionPriorList = calculate_prior_distribution(cls_base_fn, srvMin, srvMax, largestSlabN, srv.shape[1], args.slab_offset_ratio, args.outDir, args.step, args.slab_shift_width, desc=desc, clobber=args.clobber ) 

        df = pd.DataFrame({"slab":[], "y":[],"y_end":[], "nmi": [],  "tfm":[], "MetricValue":[], "Metric":[], "RegLevels":[], "Offset":[], "RegSchedule":[]  })
        best_df = pd.DataFrame({"slab":[], "y":[],"y_end":[], "nmi": [],  "tfm":[],"MetricValue":[], "Metric":[], "RegLevels":[], "Offset":[], "RegSchedule":[] })

        for slab in args.slab_n_list : 
            cls_fn=args.slab_fn_list[i-1] 
            print('Aligning:', cls_fn, slab)
            df, best_df, srv = align_single_slab(srv_fn, cls_fn, slab, args, srv, srv_img, slabPositionPriorList, df, best_df, out)
            best_df = save_best_alignments(slab, best_df, srv_fn,  out)
        del srv
        best_df.to_csv(out.best_csv_fn)
    else : 
        print("File already exists: ", out.best_csv_fn)
    
    plot_fn = args.outDir+os.sep+"slab_position.png"
    if not os.path.exists(out.csv_fn) or not os.path.exists(plot_fn) :
        save_plot(out.csv_fn, plot_fn, out, args)

    best_df = pd.read_csv(out.best_csv_fn)

    #Delete temporary (non-final) files
    for f in glob(args.outDir + os.sep + "*nii.gz")  :
        os.remove(f)

    return best_df

class AlignSlabsArgs():
    def __init__(self, slab_fn_list,inDir, outDir, metric='GC', label='', user_rate=[0.05,0.05,0.05], step=1, slab_shift_width=5, slab_offset_ratio=0.05, verbose=0, sampling=1, nbins=64, iterations=['500x250x100','500x250x50'], clobber=0) : 
        self.slab_fn_list=slab_fn_list
        self.inDir = inDir
        self.outDir = outDir 
        self.metric = metric 
        self.label = label
        self.user_rate = user_rate
        self.step = step 
        self.slab_shift_width = slab_shift_width 
        self.slab_offset_ratio = slab_offset_ratio
        self.verbose = verbose 
        self.sampling = sampling 
        self.nbins = nbins 
        self.iterations = iterations
        self.clobber = 0
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find slab position.')
    parser.add_argument('--slabs','-s', dest='slab_fn_list', type=int, nargs='+', help='List of slabs to process')
    parser.add_argument('--output-dir','-o', dest='outDir', default='reconstruction_output', help='Output directory')
    parser.add_argument('--in-dir','-i', dest='inDir', default='reconstruction_output', help='Input directory')
    parser.add_argument('--metric','-m', dest='metric', default='GC', help='Metric for registration')
    parser.add_argument('--brain','-b', dest='brain', default='MR1', help='Metric for registration')
    parser.add_argument('--hemi', dest='hemi', default='R', help='Metric for registration')
    parser.add_argument('--label','-l', dest='label', default='', type=str, help='')
    parser.add_argument('--rate','-r', dest='user_rate', default=[0.1, 0.1, 0.1],nargs='+', help='Step rate for optimization during registration')
    parser.add_argument('--step','-t', dest='step', default=1, type=float,help='Common resolution to which MRI and receptor GM masks are downsampled')
    parser.add_argument('--slab-shift-width','-w', dest='slab_shift_width',type=int, default=5, help='Amount of voxels by which window for fixed slab is shifted.')
    parser.add_argument('--slab-offset-ratio','-f', dest='slab_offset_ratio', default=0.15, type=float, help='Percentage to be added to width of slab')
    parser.add_argument('--verbose','-v', dest='verbose', default=0, type=int, help='ANTs verbosity level')
    parser.add_argument('--sampling','-a', dest='sampling', default='1',type=str, help='Sampling rate for registration')
    parser.add_argument('--nbins','-n', dest='nbins', default='32',type=str, help='Sampling rate for registration')
    parser.add_argument('--iterations','-I', dest='iterations', nargs='+', default=['500x100x50x10','500x100x50x10'], help='Number of iterations for each registration level')
    parser.add_argument('--clobber', '-c', dest='clobber', type=int, default=0, help='Clobber results')
    args = parser.parse_args()

    align_slabs(args, srv_fn = "srv/mri1_gm_bg_srv.nii.gz", scale_factors_fn="data/scale_factors.json" )






