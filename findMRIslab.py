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
from ANTs import ANTs
from utils.utils import shell, splitext
from nibabel.processing import resample_to_output
from utils.utils import splitext  
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat
from numpy.linalg import det
from scipy.interpolate import interp1d


def get_determinant(f) :
    M = loadmat(f)
    mat = np.array(M['AffineTransform_float_3_3']).reshape(4,3)
    mat = np.hstack([mat, np.array([0,0,0,1]).reshape(4,1)  ])
    return det(mat)

def findSRVcoordinates(vol) :
    '''
    Finds the min and max spatial coordinate of the srv image
    '''
    profile = np.sum(vol, axis=(0,2))
    profileRange = np.arange(vol.shape[1])[ profile > 0 ]
    
    if len(profileRange) == 0 :
        print("Error : empty srv file")
        exit(1)

    srvMin = profileRange[0]  
    srvMax = profileRange[-1] 
    del vol
    return srvMin, srvMax

def get_slab_start_end(i, y, slab_ymax, srv_ymax, ratio=0.1):
    #Get slab start and end position in voxel coordinates along y-axis
    y0 = max(y, 0)
    y1 = min(y + slab_ymax, srv_ymax)

    offset = ratio * slab_ymax
    if int(i) in [1,2,3] :
        #print('Slab', i, 'section',srv.shape[1],'y', y,'y0', y0, 'y1', y1)
        start = srv_ymax - y1
        end = srv_ymax - y0
        start =max( int(start - offset), 0)
        end = min(int(end + offset ), srv_ymax)
    else :
        end = y1
        start = y0
        start = min(int(start - offset), srv_ymax)
        end = max(int(end+offset), 0)
    mid = (end+start)/2.
    return start, mid, end 

def registration(outDir,slab, start,end,srv,tfm_prefix, moving_fn,fixed_fn, moving_rsl_prefix, moving_rsl_fn_inverse, srvRsl, p, args) :
    metric=args.metric
    sampling=args.sampling
    clobber=args.clobber

    if not metric in ['CC', 'MI', 'Mattes', 'Demons', 'GC'] : 
        print("Error: Metric",metric,"not found")
        exit(1)

    ### Extract fixed image from MRI GM Classification SRV and adjust start location along y axis
    fixed = srv[:, start:end, :]

    affine =np.copy( srvRsl.affine)
    affine[1,3] = affine[1,3] + affine[1,1] * start
    if not os.path.exists(fixed_fn) or clobber > 0 :
        print("Writing fixed ", fixed_fn)
        print(start, end, affine[1,:])
        nib.Nifti1Image( fixed, affine  ).to_filename(fixed_fn)
    
    #Works
    iterations =['1500x1000', '1500x1500']
    shrink_factors = ['2x1', '2x1'] #, '2'] #, '3']
    smoothing_sigmas = ['1.0x0.0', '1.0x0.0'] #, '1.0'] #, '1.5']
    metrics = ['GC', 'GC', 'GC' ]
    tfm_types=['Rigid','Affine','SyN']
    
    moving_rsl_fn = outDir+os.sep+'cls_moving_rsl_'+str(slab)+'_'+str(start)+'_'+str(end) +'_level-1_GC_Affine.nii.gz'

    tfm_syn = outDir+os.sep+'affine_'+str(slab)+'_'+str(start)+'_'+str(end) +'_level-1_GC_Affine_Composite.h5'
    if (not os.path.exists(tfm_syn) and not os.path.exists(moving_rsl_fn)) or args.clobber :
        print("Finding:", tfm_syn, moving_rsl_fn)
        tfm_syn, moving_rsl_fn = ANTs( tfm_prefix, fixed_fn, moving_fn, moving_rsl_prefix, iterations=iterations, tfm_type=tfm_types, shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas, metrics=metrics, verbose=0, clobber=clobber,  exit_on_failure=True, generate_masks=False)
    else : 
        print("Skipping")
    metric_val=0
    if  os.path.exists(fixed_fn) and os.path.exists(moving_rsl_fn) :
        if args.metric == "Mattes" :
            try :
                metric_val=ants.create_ants_metric(ants.image_read(fixed_fn), ants.image_read(moving_rsl_fn), metric_type='MattesMutualInformation').get_value()
            except RuntimeError :
                pass

        if args.metric == "GC" :
            try :
                metric_val=ants.create_ants_metric(ants.image_read(fixed_fn), ants.image_read(moving_rsl_fn), metric_type='Correlation').get_value()
            except RuntimeError :
                pass
    
    del fixed
    return tfm_syn, moving_rsl_fn, metric_val

def load_srv_image(srv_fn, outDir, step, clobber=0) :
    split = splitext(os.path.basename(srv_fn))
    srvRsl_fn = outDir + os.sep +  split[0] + '_'+str(step)+split[1]

    if not os.path.exists(srvRsl_fn)  or clobber > 2 :
        srvImg = nib.load(srv_fn)
        srvRsl = resample_to_output(srvImg, [step,step,step], order=1)
        srvRsl.to_filename(srvRsl_fn)
    else : 
        srvRsl = nib.load(srvRsl_fn)

    return srvRsl

def get_slab_image(cls_fn, moving_fn, outDir, step, clobber=0) :
    if not os.path.exists(moving_fn) or clobber > 2:
        slabImg = nib.load(cls_fn)
        vol = gaussian_filter1d(slabImg.get_data(), 3, axis=(1) )
        slabRsl = resample_to_output(nib.Nifti2Image(vol,slabImg.affine), [step,step,step], order=0)
        del slabImg
        slab = slabRsl.get_data()
        nib.Nifti1Image( slab, slabRsl.affine ).to_filename(moving_fn)
    else : 
        slabRsl = nib.load(moving_fn)
    return slabRsl

def get_cls_fn(i,cls_base_fn) :
    cls_fn=re.sub('<slab>',str(i), cls_base_fn)
    if not os.path.exists(cls_fn) :
        print('Error : could not find cls slab for ', cls_fn)
        exit(1)
    print('Read',cls_fn)
    return cls_fn

def calculate_prior_distribution(cls_base_fn, srvMin, srvMax, nSlabs, srv_ymax,slab_offset_ratio, outDir, step,slab_shift_width, clobber=0):
    distanceList=[]
    slabShape=[]
    srvWidth = srvMax - srvMin
    slabPositionPriorList=[]
    slabPositionPrior={}
    for i in range(1,nSlabs+1) : 
        moving_fn = outDir+os.sep+'orig_'+str(i)+'.nii.gz'
        cls_fn = get_cls_fn(i, cls_base_fn)
        slabRsl = get_slab_image(cls_fn, moving_fn, outDir, step, clobber=2)    
        slab = slabRsl.get_data()
        slabShape.append( slab.shape[1] )
    
    totalWidth = np.sum(slabShape) 
    widthDiff = (srvWidth-totalWidth)/(nSlabs-1)
    prob_arrays=[]
    for i in range(nSlabs) :
        s = i+1
        prevWidth=0
        if s > 1 : prevWidth = np.sum(slabShape[(i-1)::-1])
        slabPositionPriorList += [ np.rint(srvMax - i*widthDiff - slabShape[i]/2.0 - prevWidth).astype(int)  ]

        impulse_array = np.zeros([srv_ymax])
        impulse_array[slabPositionPriorList[-1]] = 1.
        blurred_impulse_array = gaussian_filter(impulse_array, slab.shape[1]*1.5, mode='constant',cval=0 )
        prob_arrays.append( blurred_impulse_array )
        plt.plot(blurred_impulse_array)

    plt.savefig(outDir+'/prior.png')

    print('Slab Position Prior', srvMin, slabPositionPriorList[i], srvMax)
    slab_position_prior_prob={}
    for i in range(nSlabs) : 
        s=i+1
        slabPositionPrior[s]= slabPositionPriorList[i]
        slab_position_prior_prob[s]={}
        slab_position_prior_prob[int(s)]['max'] = np.max(prob_arrays[i])
        for y in range(srvMin, srvMax, slab_shift_width) : 
            start, mid, end = get_slab_start_end(s, y, slabShape[i], srv_ymax,slab_offset_ratio)
            slab_position_prior_prob[int(s)]['function'] = interp1d(range(srv_ymax), prob_arrays[i], kind='cubic')

    return slab_position_prior_prob

def update_df(slab_df, best_df, df, out, srv_fn, metric="nmi_p") :
    #print(slab_df)
    #print(best_df)
    idx = slab_df["MetricValue"].loc[ slab_df["Metric"]==metric ].idxmax()
    best_df =  best_df.append( slab_df.iloc[idx,] )
    df = df.append(slab_df)
    df.to_csv(out.csv_fn)
    return slab_df, best_df, df

def update_srv(i,slab_df, srv, affine, outDir, metric) :
    idx = slab_df["MetricValue"].loc[ slab_df["Metric"]==metric ].idxmax()
    slab_y0=slab_df["y"].iloc[  idx,  ]
    slab_y1=slab_df["y_end"].iloc[ idx, ]

    temp=np.copy(srv)
    temp[:, slab_y0:slab_y1 , :] = temp[:, slab_y0:slab_y1 , :] * int(i)
    
    
    if int(i) in [1,2,3] : 
        srv[:, slab_y0: , :] = 0
    else :
        srv[:, 0 : slab_y1, :] = 0
    
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
    print(srvImgHiRes.shape)
    print(y0, y1)
    print( y0High , y1High)
    srv = srvImgHiRes.get_data() 

    outVol = srv[ : , y0High : y1High , : ]

    out_fn = out.outDirFinal + os.sep + os.path.basename( fixed_fn  )
    out_moving_rsl_fn = out.outDirFinal + os.sep + os.path.basename( moving_rsl_fn  )
    shutil.copy( moving_rsl_fn, out_moving_rsl_fn) 

    aff = srvImgHiRes.affine
    aff[1,3] = srvImgHiRes.affine[1,3] + srvImgLoRes.affine[1,1] * y0
    print(aff)
    print(srv_fn)
    print(out_fn)
    print(outVol.shape)
    nib.Nifti1Image(outVol, aff ).to_filename( out_fn )

    best_df["fixed"].loc[ best_df["slab"] == i ,] = out_fn

    return best_df

def align_single_slab(srv_fn, cls_fn, i, args, srv, srvRsl, slab_position_prior_prob, df, best_df, out) :
    moving_fn = args.outDir+os.sep+'orig_'+str(i)+'.nii.gz'
    
    slabRsl = get_slab_image(cls_fn, moving_fn, args.outDir, args.step, clobber=args.clobber)    
    slab = slabRsl.get_data()
    srvMin, srvMax = findSRVcoordinates(srv)
    print("Slab", i, srvMin, srvMax)
    print("Clobber",args.clobber)
    slab_df=pd.DataFrame({})
    for y in range(srvMin, srvMax, args.slab_shift_width) : 
        start, mid, end = get_slab_start_end(i, y, slab.shape[1], srvRsl.shape[1],args.slab_offset_ratio)
        print(y, start, mid, end)
        fixed_fn = out.outDir+os.sep+'srv_fixed_'+str(i)+'_'+str( start )+'_'+str( end )+'.nii.gz'
        moving_rsl_prefix = out.outDir+os.sep+'cls_moving_rsl_'+str(i)+'_'+str(start)+'_'+str(end) #+'.nii.gz'

        moving_rsl_fn_inverse = out.outDir+os.sep+'srv_fixed_rsl_'+str(i)+'_'+str(start)+'_'+str(end)+'_level-1_GC_Affine_Composite.nii.gz'
        
        tfm_prefix = out.outDir + os.sep+'affine_'+str(i)+'_'+str(start)+"_"+str(end)+"_"

        p = slab_position_prior_prob[int(i)]['function'](mid) / slab_position_prior_prob[int(i)]['max']
        print("\t",p)
        if p  < 0.90 :
            print("Break : low probability threshold reached :", p )
            break
        tfm_fn, moving_rsl_fn, metric_val = registration(out.outDir, i, start,end,srv, tfm_prefix, moving_fn, fixed_fn, moving_rsl_prefix, moving_rsl_fn_inverse, srvRsl, p, args )

        row_dict = {
                "slab":[i]*3, 
                "y":[start]*3,
                "y_end":[end]*3, 
                "MetricValue": [-metric_val,p,-metric_val*p],
                "Metric":[args.metric, "p", args.metric+"_p"],
                "tfm":[tfm_fn]*3,
                "fixed":[fixed_fn]*3, 
                "moving_rsl":[moving_rsl_fn]*3,
                "RegLevels":[len(args.iterations)]*3,
                "Offset":[args.slab_offset_ratio]*3,
                "RegSchedule":[ '_'.join(args.iterations) ]*3
                }
        
        row = pd.DataFrame(row_dict)
        slab_df = slab_df.append(row, ignore_index=True)
        print("Slab:", i, "start :", start,"end:", end, "Mattes", -metric_val*p, "p :", p )
    
    slab_df, best_df, df =  update_df(slab_df, best_df, df, out, srv_fn, metric=args.metric+'_p')

    srv = update_srv(i,slab_df, srv, srvRsl.affine, out.outDir, args.metric+'_p') 

    return df, best_df, srv



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

def align_slabs( args, cls_base_fn="MR1/R_slab_<slab>/classify/vol_cls_<slab>_250um.nii.gz", srv_fn = "srv/mri1_gm_bg_srv.nii.gz", scale_factors_fn="data/scale_factors.json" ):
    args.slab_fn_list = adjust_slab_list( args.slab_fn_list )
    print(args.slab_fn_list)
    # Set optimization step rate
    args.rate = [0.1, 0.1, 0.1]
    for i in range(len(args.user_rate)) :
        args.rate[i] = args.user_rate[i]

    out = OutputFilenames(args)
    nSlabs = len(args.slab_fn_list)
    largestSlabN = int(max(args.slab_fn_list))

    cls_base_fn = args.inDir + os.sep + cls_base_fn

    if not os.path.exists(args.outDir) :
        os.makedirs(args.outDir)

    if not os.path.exists(out.best_csv_fn) or args.clobber > 0 : 
        print('Generate :', out.best_csv_fn)
        #Load SRV MRI GM mask
        srvRsl = load_srv_image(srv_fn,args.outDir,args.step, args.clobber)
        srv = srvRsl.get_data()

        #Calculate prior distribution to weight metrics 
        srvMin, srvMax = findSRVcoordinates(srv)
         
        slab_position_prior_prob = calculate_prior_distribution(cls_base_fn, srvMin, srvMax, largestSlabN, srv.shape[1], args.slab_offset_ratio, args.outDir, args.step, args.slab_shift_width, clobber=args.clobber ) 

        df = pd.DataFrame({"slab":[], "y":[],"y_end":[], "nmi": [],  "tfm":[], "MetricValue":[], "Metric":[], "RegLevels":[], "Offset":[], "RegSchedule":[]  })
        best_df = pd.DataFrame({"slab":[], "y":[],"y_end":[], "nmi": [],  "tfm":[],"MetricValue":[], "Metric":[], "RegLevels":[], "Offset":[], "RegSchedule":[] })

        for i in args.slab_fn_list : 
            cls_fn=get_cls_fn(i, cls_base_fn)
            df, best_df, srv = align_single_slab(srv_fn, cls_fn, i, args, srv, srvRsl, slab_position_prior_prob, df, best_df, out)
            best_df = save_best_alignments(i, best_df, srv_fn,  out)

        best_df.to_csv(out.best_csv_fn)
    else : 
        print("File already exists: ", out.best_csv_fn)
    
    plot_fn = args.outDir+os.sep+"slab_position.png"
    if not os.path.exists(out.csv_fn) or not os.path.exists(plot_fn) :
        save_plot(out.csv_fn, plot_fn, out, args)

    best_df = pd.read_csv(out.best_csv_fn)

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
    parser.add_argument('--output-dir','-o', dest='outDir', default='receptor_to_mri', help='Output directory')
    parser.add_argument('--metric','-m', dest='metric', default='GC', help='Metric for registration')
    parser.add_argument('--label','-l', dest='label', default='', type=str, help='')
    parser.add_argument('--rate','-r', dest='user_rate', default=[0.1, 0.1, 0.1],nargs='+', help='Step rate for optimization during registration')
    parser.add_argument('--step','-t', dest='step', default=1, help='Common resolution to which MRI and receptor GM masks are downsampled')
    parser.add_argument('--slab-shift-width','-w', dest='slab_shift_width', default=5, help='Amount of voxels by which window for fixed slab is shifted.')
    parser.add_argument('--slab-offset-ratio','-f', dest='slab_offset_ratio', default=0.15, type=float, help='Percentage to be added to width of slab')
    parser.add_argument('--verbose','-v', dest='verbose', default=0, type=int, help='ANTs verbosity level')
    parser.add_argument('--sampling','-a', dest='sampling', default='1',type=str, help='Sampling rate for registration')
    parser.add_argument('--nbins','-n', dest='nbins', default='32',type=str, help='Sampling rate for registration')
    
    parser.add_argument('--iterations','-i', dest='iterations', nargs='+', default=['500x250x100','500x250x100'], help='Number of iterations for each registration level')
    parser.add_argument('--clobber', '-c', dest='clobber', type=int, default=0, help='Clobber results')
    args = parser.parse_args()

    align_slabs(args, srv_fn = "srv/mri1_gm_bg_srv.nii.gz", scale_factors_fn="data/scale_factors.json" )






