import bisect
import contextlib
import re
import scipy
import os
import pandas as pd
import imageio
import utils.ants_nibabel as nib
import PIL
import matplotlib
import time
import ants
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from re import sub
import nibabel
from utils.fit_transform_to_paired_points import fit_transform_to_paired_points
from ants import get_center_of_mass
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage.filters import gaussian_filter 
from os.path import basename
from subprocess import call, Popen, PIPE, STDOUT
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from skimage.transform import resize
from scipy.ndimage import label, center_of_mass

def get_edges_from_faces(faces):
    #for convenience create vector for each set of faces 
    f_i = faces[:,0]
    f_j = faces[:,1]
    f_k = faces[:,2]
    
    #combine node pairs together to form edges
    f_ij = np.column_stack([f_i,f_j])
    f_jk = np.column_stack([f_j,f_k])
    f_ki = np.column_stack([f_k,f_i])

    #concatenate the edges into one big array
    edges_all = np.concatenate([f_ij,f_jk, f_ki],axis=0).astype(np.uint32)

    #there are a lot of redundant edges that we can remove
    #first sort the edges within rows because the ordering of the nodes doesn't matter
    edges_all_sorted_0 = np.sort(edges_all,axis=1)
    #create a vector to keep track of vertex number
    edges_all_range= np.arange(edges_all.shape[0]).astype(int)
    #
    edges_all_sorted = np.column_stack([edges_all_sorted_0, edges_all_range ])
    
    #sort the rows so that duplicate edges are adjacent to one another 
    edges_range_sorted = pd.DataFrame( edges_all_sorted  ).sort_values([0,1]).values
    edges_sorted = edges_range_sorted[:,0:2]

    #convert sorted indices to indices that correspond to face numbers
    #DEBUG commented out following line because it isnt' used:
    #sorted_indices = edges_range_sorted[:,2] % faces.shape[0]

    # the edges are reshuffled once by sorting them by row and then by extracting unique edges
    # we need to keep track of these transformations so that we can relate the shuffled edges to the 
    # faces they belong to.
    edges, edges_idx, counts = np.unique(edges_sorted , axis=0, return_index=True, return_counts=True)
    edges = edges.astype(np.uint32)

    #print('2.', np.sum( np.sum(edges_all==(26, 22251),axis=1)==2 ) ) 
    
    assert np.sum(counts!=2) == 0,'Error: more than two faces per edge {}'.format( edges_sorted[edges_idx[counts!=2]])     
    #edge_range = np.arange(edges_all.shape[0]).astype(int) % faces.shape[0]
    return edges

def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert, faces_fn, ref_gii_fn, ext, mni_fn):
    print("transforming", in_gii_fn)

    origin = [0,0,0]
    if ext in ['.pial', '.white'] : 
        _, _, volume_info = load_mesh(ref_gii_fn)
        origin=volume_info['cras'] 
    else : volume_info = ref_gii_fn

    coords = h5.File(in_gii_fn)['data'][:]
    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    if np.sum(tfm.fixed_parameters) != 0 : flip=-1
    
    in_file = open(in_gii_fn, 'r')
    
    out_path, out_ext = os.path.splitext(out_gii_fn)
    coord_fn = out_path + '_ants_reformat.csv'
    temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv'
    coords = np.concatenate([coords, np.zeros([coords.shape[0],2])], axis=1 )
    print('flip', flip)
    print('origin',origin)

    if os.path.splitext(faces_fn)[1] == '.h5' :
        faces_h5=h5.File(faces_fn,'r')
        faces = faces_h5['data'][:]
    else :
        _, faces, _ = load_mesh(faces_fn)

    #the for loops are here because it makes it easier to trouble shoot to check how the vertices need to be flipped to be correctly transformed by ants
    #for flipx in [-1]: #[1,-1] :
    #    for flipy in [-1]: #[1,-1]:
    #        for flipz in [1]: #[1,-1]:
    coords[0] = flip*(coords[0] -origin[0])
    coords[1] = flip*(coords[1] +origin[1])
    coords[2] = (coords[2] +origin[2])
    
    df = pd.DataFrame(coords,columns=['x','y','z','t','label'])
    df.to_csv(coord_fn, columns=['x','y','z','t','label'], header=True, index=False)
    

    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert}]  -o {temp_out_fn}',verbose=True)
    df = pd.read_csv(temp_out_fn,index_col=False)
    df['x'] = flip*(df['x'] - origin[0])
    df['y'] = flip*(df['y'] - origin[1])
    df['z'] = (df['z'] - origin[2])

    new_coords = df[['x','y','z']].values

    f_h5 = h5.File(out_gii_fn, 'w')
    f_h5.create_dataset('data', data=new_coords) 
    f_h5.close()

    save_mesh(out_path, new_coords, faces, volume_info=volume_info)
    os.remove(temp_out_fn)
    #obj_fn = out_path+ f'_{flipx}{flipy}{flipz}'+ '.obj'

    #save_obj(obj_fn,coords, faces)
    #print(obj_fn)

def get_section_intervals(vol):
    section_sums = np.sum(vol, axis=(0,2))
    valid_sections = section_sums > np.min(section_sums)
    plt.subplot(2,1,1); plt.plot(section_sums)
    plt.subplot(2,1,2); plt.plot(valid_sections); 
    plt.savefig(f'val_sections_{np.sum(valid_sections)}.png'); plt.clf(); plt.cla()
    labeled_sections, nlabels = label(valid_sections)
    assert nlabels >= 2, 'Error: there must be a gap between thickened sections. Use higher resolution volumes.'

    intervals = [ (np.where(labeled_sections==i)[0][0], np.where(labeled_sections==i)[0][-1]) for i in range(1, nlabels+1) ]
    assert len(intervals) > 0 , 'Error: no valid intervals found for volume.'  
    return intervals
    
def resample_to_autoradiograph_sections(brain, hemi, slab, resolution,input_fn, ref_fn, tfm_inv_fn, iso_output_fn, output_fn):
    '''
    About:
        Apply 3d transformation and resample volume into the same coordinate space as 3d receptor volume.           This produces a volume with 0.02mm dimension size along the y axis.

    Inputs:
        brain:      current subject brain id
        hemi:       current hemisphere (R or L)
        slab:       current slab number
        resolution:     current resolution level
        input_fn:     gm super-resolution volume (srv) extracted from donor brain
        ref_fn:     brain mask of segmented autoradiographs
        tfm_inv_fn:   3d transformation from mni to receptor coordinate space
        srv_space_rec_fn:      
        
    Outpus:
        None
    '''
    shell(f'antsApplyTransforms -n HammingWindowedSinc -v 1 -d 3 -i {input_fn} -r {ref_fn} -t {tfm_inv_fn} -o /tmp/tmp.nii.gz',True)
    img = nib.load('/tmp/tmp.nii.gz')
    vol = img.get_fdata()
    assert np.sum(vol) > 0, f'Error: empty volume {iso_output_fn}'
     
    img = resample_to_output(nibabel.Nifti1Image(vol,img.affine), [float(resolution)]*3, order=0)
    vol = img.get_fdata()
    nib.Nifti1Image(vol, img.affine ).to_filename(iso_output_fn)

    img = resample_to_output(nibabel.Nifti1Image(vol,img.affine), [float(resolution),0.02, float(resolution)], order=0)
    vol = img.get_fdata()
    nib.Nifti1Image(vol, img.affine ).to_filename(output_fn)

    os.remove('/tmp/tmp.nii.gz')


def fix_affine(fn):
    try :
        ants.image_read(fn)
    except RuntimeError :
        img = nib.load(fn)
        dim = vol.shape
        aff = img.affine

        origin = list(affine[ [0,1],[3,3] ])
        spacing = list( affine[ [0,1],[0,1] ])
        ants_image = ants.from_numpy(img.get_fdata(), origin=origin, spacing=spacing)
        ants_image.to_filename(fn)
        try :
            ants.image_read(fn)
        except RuntimeError :
            print("Error could not fix direction cosines for"+fn)
            exit(1)

def read_points(fn, ndim = 3):
    start_read=False
    points0 = []
    points1 = []
    fn_list = []
    with open(fn,'r') as F:
        for line in F.readlines() :
            if start_read : 
                string_split = line.split(' ')
                points0 += [[ float(i) for i in string_split[1:(1+ndim)] ]]
                points1 += [[ float(i) for i in string_split[4:(4+ndim)] ]]

            if 'Points' in line : start_read=True
            
            if '%Volume:' in line : 
                fn = line.split(' ')[1]
                fn_list.append( os.path.basename(fn.rstrip()))
   
    return np.array(points0), np.array(points1), fn_list[0], fn_list[1]

def safe_ants_image_read(fn, tol=0.001, clobber=False):
    img = nib.load(fn)
    affine = img.affine
    origin = list(affine[ [0,1,2],[3,3,3] ])
    spacing = list( affine[ [0,1,2],[0,1,2] ])
    ants_image = ants.from_numpy(img.get_fdata(), origin=origin, spacing=spacing)
    return ants_imag


def points2tfm(points_fn, affine_fn, fixed_fn, moving_fn, ndim=3, transform_type="Affine", invert=False, clobber=False):

    #comFixed=[] 
    #comMoving=[] 
    #comMoving[-1] *= 1 
    if not os.path.exists(affine_fn) or clobber :

        fixed_img = ants.image_read(fixed_fn)
        moving_img = ants.image_read(moving_fn)


        comFixed = list( get_center_of_mass(fixed_img) ) 
        comMoving = list( get_center_of_mass(moving_img) )

        print(comFixed)
        print(comMoving)

        fixed_dirs = fixed_img.direction[[0,1,2],[0,1,2]]
        moving_dirs = moving_img.direction[[0,1,2],[0,1,2]]
        # f=rec_points / m=mni_points
        fixed_points, moving_points, fixed_fn, moving_fn = read_points(points_fn, ndim=ndim)

        print('\t: Calculate affine matrix from points')
        if invert:
            # f=mni_points / m=rec_points
            temp_points = np.copy(moving_points)
            moving_points = np.copy(fixed_points)
            fixed_points = temp_points

        fixed_points = fixed_dirs * fixed_points
        moving_points = moving_dirs * moving_points
        print(np.mean(fixed_points,axis=0))
        print(np.mean(moving_points,axis=0))

        fixed_points = fixed_points - np.mean(fixed_points,axis=0) + comFixed
        moving_points = moving_points - np.mean(moving_points,axis=0) + comMoving

        landmark_tfm = fit_transform_to_paired_points(moving_points, fixed_points, transform_type=transform_type , centerX=comFixed, centerY=comMoving)

        ants.write_transform(landmark_tfm, affine_fn)
        
        df=pd.DataFrame( {'x':moving_points[:,0],'y':moving_points[:,1],'z':moving_points[:,2]} )
        
        rsl_points = ants.apply_transforms_to_points(3, df, affine_fn, whichtoinvert=[True] )
        
        error = np.sum(np.sqrt(np.sum(np.power((rsl_points - fixed_points), 2), axis=1)))
        print('Error', error / rsl_points.shape[0])

    return affine_fn

def newer_than(input_list, output_list) :
    '''
    check if the files in input list are newer than target_filename
    '''
    for output_filename in output_list :
        t0 = os.path.getctime(output_filename)

        for input_filename in input_list :
            t1 = os.path.getctime(input_filename)

            if t1 < t0 : 
                print('Input file is newer than output file.')
                print('\tInput file:')
                print('\t',t1, time.ctime(os.path.getctime(input_filename)))
                print('\tOutput file:')
                print('\t',t0, time.ctime(os.path.getctime(output_filename)))
                exit(0)
                return True
    
    return False

def run_stage(input_list, output_list):
    '''
    check if stage needs to be run
    '''
    # check if inputs exist
    for input_filename in input_list :
        assert os.path.exists(input_filename), f'Error: input to stage does not exist, {input_filename}'
    
    # check if outputs exist
    for output_filename in output_list :
        if not os.path.exists(output_filename) :
            return True

    # check if inputs are newer than existing outputs
    #return newer_than(input_list, output_list)
    return False


def get_seg_fn(dirname, y, resolution, filename, suffix=''):
    filename = re.sub('.nii.gz',f'_y-{int(y)}_{resolution}mm{suffix}.nii.gz', os.path.basename(filename))
    return '{}/{}'.format(dirname, filename )

def gen_2d_fn(prefix,suffix,ext='.nii.gz'):
    return f'{prefix}{suffix}{ext}'

def save_sections(file_list, vol, aff) :
    xstep = aff[0,0]
    ystep = aff[1,1]
    zstep = aff[2,2]

    xstart = aff[0,3]
    zstart = aff[2,3]

    for fn, y in file_list:
        affine = np.array([  [xstep,  0, 0, xstart ],
                                [0, zstep, 0, zstart ],
                                [0, 0,  0.02, 0 ],
                                [0, 0,  0, 1]])
        i=0
        print(y,fn)
        if np.sum(vol[:,int(y),:]) == 0 :
            # Create 2D srv section
            # this little while loop thing is so that if we go beyond  brain tissue in vol,
            # we find the closest y segement in vol with brain tissue
            while np.sum(vol[:,int(y-i),:]) == 0 :
                i += (ystep/np.abs(ystep)) * 1
        sec = vol[ :, int(y-i), : ]
        nib.Nifti1Image(sec , affine).to_filename(fn)

def get_to_do_list(df,out_dir,str_var,ext='.nii.gz'):
    to_do_list=[]
    for idx, (i, row) in enumerate(df.iterrows()):
        y = int(row['slab_order'])
        assert int(y) >= 0, f'Error: negative y value found {y}'
        prefix = f'{out_dir}/y-{y}' 
        fn = gen_2d_fn(prefix,str_var,ext=ext)
        print(fn)
        if not os.path.exists(fn) : to_do_list.append( [fn, y])
    return to_do_list

def create_2d_sections( df,  srv_fn, resolution, output_dir,clobber=False) :
    fx_to_do=[]
    
    tfm_dir = output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    fx_to_do = get_to_do_list(df, tfm_dir, '_fx') 

    if len( fx_to_do ) > 0 :
        srv_img = nib.load(srv_fn)
        affine = srv_img.affine
        srv = srv_img.get_fdata()
        save_sections(fx_to_do, srv, affine)
        

def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)

def v2w(i, step, start) :
    return start + i * step   
def splitext(s):
    try :
        ssplit = os.path.basename(s).split('.')
        ext='.'+'.'.join(ssplit[1:])
        basepath= re.sub(ext,'',s)
        return [basepath, ext]
    except TypeError :  
        return s
def setup_tiers(df, tiers_str):
    tiers_list = [ j.split(",") for j in tiers_str.split(";")]
    i=1
    df["tier"]=[0] * df.shape[0]
    if tiers_str != '' :
        for tier in tiers_list :
            for ligand in tier :
                df["tier"].loc[ df.ligand == ligand ] = i
            
            i += 1
        df=df.loc[df["tier"] != 0 ]
    return(df)

def pad_size(d):
    if d % 2 == 0 :
        pad = (int(d/2), int(d/2))
    else :
        pad = ( int((d-1)/2), int((d+1)/2))
    return(pad)
    

def add_padding(img, zmax, xmax):
    z=img.shape[0]
    x=img.shape[1]
    
    dz = zmax - z
    dx = xmax - x
    
    z_pad = pad_size(dz)
    x_pad = pad_size(dx)
    
    img_pad = np.pad(img, (z_pad,x_pad), 'constant', constant_values=np.max(img))
    #img_pad = np.zeros([zmax,xmax])
    #img_pad[z_pad[0]:, x_pad[0]: ] = img
    return img_pad


def gm_segment(img):
    mid = np.mean(img[img>0])
    upper=np.max(img)
    if upper > 0 :
        init=np.array([0,mid,upper]).reshape(-1,1)
        cls = KMeans(3, init=init).fit_predict(img.reshape(-1,1)).reshape(img.shape)
        cls[ cls != 2 ] = 0
        return cls.astype(float)
    return img

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

def shell(cmd, verbose=False,exit_on_failure=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout=""
    if verbose :
        print(cmd)


    process=Popen( cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose :
            print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print ("Error:")
        print ("Command:", cmd)
        print ("Stderr:", stdout)
        if exit_on_failure : exit(errorcode)
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

from scipy.ndimage import zoom
from nibabel.processing import resample_to_output


def safe_imread(fn) :
    img = imageio.imread(fn)
    if len(img.shape) > 2 :
        img = np.mean(img,axis=2)
    return img

    
def downsample(img, subject_fn="", step=0.2, raw_step=0.02, interp=3):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * raw_step 
    l1 = img.shape[1] * raw_step

    #Calculate the length for the downsampled image
    dim0=int(np.ceil(l0 / step))
    dim1=int(np.ceil(l1 / step))

    #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
    
    #Downsample
    img_dwn = resize(img.astype(float), (dim0, dim1), order=int(interp) ) 
     
    if subject_fn != "" : 
        imageio.imsave(subject_fn, img_dwn.astype(np.uint16))

    return(img_dwn)


def downsample_and_crop(source_lin_dir, lin_dwn_dir,crop_dir, affine, step=0.2, clobber=False):

    for f in glob(source_lin_dir+"/*.TIF") :
        dwn_fn = lin_dwn_dir + splitext(basename(f))[0] + '.nii.gz'
        if not os.path.exists(dwn_fn) or clobber :
            try :
                base=sub('#L','',basename(splitext(f)[0]))
                path_string=crop_dir+"/detailed/**/"+base+"_bounding_box.png"
                print("path:", path_string)

                crop_fn = glob(path_string)[0]
            except IndexError :
                print("\t\tDownsample & Crop : Skipping ", f)
                continue
            img = imageio.imread(f)
            if len(img.shape) == 3 : img = np.mean(img,axis=2)

            bounding_box = imageio.imread(crop_fn) 
            if np.max(bounding_box) == 0 : 
                bounding_box = np.ones(img.shape)
            else : 
                bounding_box = bounding_box / np.max(bounding_box)
            img = img * bounding_box 
            nib.processing.resample_to_output(nibabel.Nifti1Image(img, affine), step, order=5).to_filename(dwn_fn)
            print("downsampled filename", dwn_fn)
            #nib.Nifti1Image(img, affine).to_filename(dwn_fn)

def world_center_of_mass(vol, affine):
    affine = np.array(affine)
    ndim = len(vol.shape)
    r = np.arange(ndim).astype(int)
    com = center_of_mass(vol)
    steps = affine[r,r]
    starts = affine[r,3]
    wcom = com * steps + starts
    return wcom

#def reshape_to_min_dim(vol):
from scipy.ndimage import shift
def recenter(vol, affine, direction=np.array([1,1,-1])):
    affine = np.array(affine)
    
    nib.Nifti1Image(vol, affine).to_filename('test_not-shifted.nii.gz')
    vol_sum_1=np.sum(vol)
    wcom1 = world_center_of_mass(vol,affine) 

    ndim = len(vol.shape)
    
    coi = np.array(vol.shape) / 2
    com = center_of_mass(vol)
    d_vox = np.rint(coi - com)
    d_vox[1]=0
    d_world = d_vox * affine[range(ndim),range(ndim)]
    d_world *= direction 
    affine[range(ndim),3] -= d_world
    vol = shift(vol,d_vox, order=0)

    nib.Nifti1Image(vol, affine).to_filename('test_shifted.nii.gz')
    
    wcom2 = world_center_of_mass(vol,affine) 
    steps = affine[range(ndim),range(ndim)]
    #assert np.sum(np.abs(wcom1- wcom2)) < np.sum(np.abs(steps))*3, f'Error: center of mass does not match {wcom1}, {wcom2}'
    #assert np.abs(vol_sum_1 - np.sum(vol))/vol_sum_1 < 0.02, f'Error: total intensities does not match after recentering'
    return vol, affine


def prefilter_and_downsample(input_filename, new_resolution, output_filename, 
                            reference_image_fn='',
                            new_starts=[None, None, None], recenter_image=False ):

    print(input_filename)
    img = nib.load(input_filename)
    direction = ants.image_read(input_filename).direction
    vol = img.get_fdata()
    affine = img.affine
    
    ndim = len(vol.shape)


    if recenter_image : 
        vol, affine = recenter(vol,affine)

    new_affine = np.copy( affine )
    for i, new_start in enumerate(new_starts) :
        if new_start != None : 
            new_affine[3,i] = float( new_start )
    
    new_resolution = np.array(new_resolution).astype(float)

    if ndim ==3 : 
        if vol.shape[2] == 1 : vol = vol.reshape([vol.shape[0],vol.shape[1]])
    
    if ndim == 2 :
        new_affine[0,0] = new_resolution[0]
        new_affine[1,1] = new_resolution[1]
        
        steps = np.array( [ affine[0,0], affine[1,1] ] )

    elif ndim == 3 :
        steps = np.array( [ affine[0,0], affine[1,1], affine[2,2] ] )
        new_affine[0,0] = new_resolution[0]
        new_affine[1,1] = new_resolution[1]
        new_affine[2,2] = new_resolution[2]
    else :
        print(f'Error: number of dimensions ({ndim}) does not equal 2 or 3 for {input_filename}')
        exit(1)
    
    sd = (  np.array(new_resolution) / steps  ) / np.pi

    vol = gaussian_filter(vol, sd)


    if reference_image_fn == '' :
        vol = resample_to_output( nibabel.Nifti1Image(vol, affine), new_resolution, order=5).get_fdata()
        new_dims = [ vol.shape[0], vol.shape[1] ]
        if len(vol.shape) == 3 :
            if vol.shape[2] != 1 :
                new_dims.append(vol.shape[2])
        # Warning: This reshape step is absolutely necessary to correctly apply ants transforms.
        #           nibabel's resampling function will change the dimensions from, say, (x,y) to (x,y,1)
        #           This throws things off for ants so the volume has to be reshaped back to original dimensions.
        vol = vol.reshape(*new_dims) 
        nib.Nifti1Image(vol, new_affine,  direction=direction).to_filename(output_filename)
        #write_nifti(vol, new_affine, output_filename)
    else :
        vol = resample_from_to(img, nib.load(reference_image_fn), order=5).get_fdata()
        nib.Nifti1Image(vol, nib.load(reference_image_fn).affine ).to_filename(output_filename, direction=direction)
        #write_nifti(vol, read_affine(reference_image_fn), output_filename)


def rgb2gray(rgb): return np.mean(rgb, axis=2)

def find_min_max(seg):
    m = np.max(seg)
    fmin = lambda a : min(np.array(range(len(a)))[a == m])
    fmax = lambda a : max(np.array(range(len(a)))[a == m])
    
    xmax = [ fmax(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]==m) > 0 ]
    xmin = [ fmin(seg[i,:]) for i in range(seg.shape[0]) if np.sum(seg[i,:]==m) > 0 ]
    
    y_i = [ i for i in range(seg.shape[0]) if np.sum(seg[i,:]) != 0  ]

    ymax = [ fmax(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]==m) > 0 ]
    ymin = [ fmin(seg[:,i]) for i in range(seg.shape[1]) if np.sum(seg[:,i]==m) > 0 ]
    x_i =  [ i for i in range(seg.shape[1]) if np.sum(seg[:,i]) != 0 ]

    if xmin == [] : xmin = [ 0 ]
    if xmax == [] : xmax = [ seg.shape[1] ]
    if  y_i == [] : y_i = [0]

    if ymin == [] : ymin = [ 0 ]
    if ymax == [] : ymax = [ seg.shape[0] ]
    if  x_i == [] : x_i = [0]
    
    return ymin, ymax, xmin, xmax, y_i, x_i

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
    for i in range(1,7) : #FIXME : Should not be hardcoded
        slice_order_fn = "MR1_R_slab_"+str(i)+"_section_numbers.csv"
        if os.path.exists(slice_order_fn) : 
            df0=pd.read_csv(slice_order_fn)
            df0["slab"]= [i] * df0.shape[0]
            slice_order_list.append(df0)
    slice_order = pd.concat(slice_order_list)
    slice_order["global_order"] =slice_order["number"].astype(int)

    slice_order_unique = np.sort(slice_order["slab"].unique())
    for i in slice_order_unique[1:] :
        prev_slab = i - 1
        prev_slab_max = slice_order["global_order"].loc[ slice_order["slab"] == prev_slab ].max() 
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
        
        ar=[f0]+ar
        processing_str=""
        ar_enum = enumerate(ar)
        n_cols = len(cols)
        n_ar = len(ar)
        if n_ar  > n_cols :
            k = n_ar - n_cols + 1
            ar_short = ar[(n_cols-k):(n_ar-1)]
            #print( (n_cols-k), (n_ar-1))
            #print(ar_short)
            
            processing_string = '-'.join(ar_short)

            ar = ar[0:(len(cols)-k)] + [processing_string] + [ar[-1]]
        
        if len(ar) != len(cols) : 
            print("Warning! Skipping :")
            print(ar)
            continue    
        df0=pd.DataFrame([ar], columns=cols)
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
    
def set_csv(source_files, output_dir, include_str="", exclude_str="", slice_order_fn="", out_fn="receptor_slices.csv", clobber=False, df_with_order=False):
    
    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    if not os.path.exists(output_dir+os.sep+out_fn ) or clobber:
        #Load csv files with order numbers
        cols=["filename", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        if df_with_order :
            cols=["filename","order", "a","b","mri","slab","hemisphere","ligand","sheet","repeat", "processing","ext"] 
        slice_order = set_slice_order()
        df = set_slice_name(source_files, cols, include_str, exclude_str)
        df = sort_slices(df, slice_order )
        ###df = set_slab_border(df)
        ###if os.path.exists(slice_order_fn)  :
        ###else : df.sort_values(by=["mri","slab","ligand","sheet","repeat"], inplace=True)
        if output_dir != "" : 
            df.to_csv(output_dir+os.sep+out_fn)
    else :
        df = pd.read_csv(output_dir+os.sep+out_fn)

    return(df)


