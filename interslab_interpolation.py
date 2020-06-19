import os
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np

from ants import  from_numpy,  apply_transforms

from ants import image_read, registration



def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def w2v(i, step, start):
    return np.round( (i-start)/step ).astype(int)
def v2w(i, step, start) :
    return start + i * step

def get_tfm(out_dir, fixed, moving, from_position, to_position,clobber=False):
    prefix='%sfrom-%s_to-%s_'%(out_dir, from_position, to_position)
    mat_fn='%s0GenericAffine.mat'%prefix

    if not os.path.exists(mat_fn) or clobber :
        reg = registration(fixed=from_numpy(fixed), moving=from_numpy(moving), type_of_transform='Affine', reg_iterations=(100,70,10), aff_metric='GC', syn_metic='CC', outprefix=prefix  )
    return mat_fn

def subslab_alignment(ant_position, pos_position, vol, ant_subslab, pos_subslab, ant_subslab_rec, pos_subslab_rec,  width, out_dir, tfm_dict, out,step=1, clobber=False) :
    position_width = ant_position - pos_position
    for cur_position in range( int(ant_position), int(pos_position), int(step) ) :
        print('\tCurrent Position', cur_position, cur_position+width)
        x0 = min(cur_position,cur_position+width)
        x1 = max(cur_position,cur_position+width)
        cur_subslab = vol[ :, x0:x1, : ]

        ant_tfm = get_tfm(out_dir, cur_subslab, ant_subslab, ant_position, cur_position, clobber=clobber)
        pos_tfm = get_tfm(out_dir, cur_subslab, ant_subslab, ant_position, cur_position, clobber=clobber)
        print('\t\tApplying Anterior Transformation') 
        ant_rsl = apply_transforms(from_numpy(cur_subslab) , from_numpy(ant_subslab_rec), [ant_tfm]).numpy()
        print('\t\tApplying Posterior Transformation') 
        pos_rsl = apply_transforms(from_numpy(cur_subslab) , from_numpy(pos_subslab_rec), [pos_tfm]).numpy()
        
        nib.Nifti1Image(cur_subslab, mri_img.affine).to_filename('test_target.nii.gz')
        nib.Nifti1Image(ant_rsl, mri_img.affine).to_filename('test_ant_pos.nii.gz')
        nib.Nifti1Image(pos_rsl, mri_img.affine).to_filename('test_ant_rec.nii.gz')
        
        #    ___       ___
        #   |   |  c  |   |
        #   |___|  |  |___|
        #       p     a
        #
        #    d = (c - p)/(a - p)
        print('\t\tInterpolating inter slab position')
        d = (cur_position - pos_position) / (ant_position - pos_position)
        interp_vol = ant_rsl * d + pos_rsl *(1-d)
        xm = x0 + np.ceil(width/2).astype(int)


        nib.Nifti1Image(interp_vol, mri_img.affine).to_filename('test_interp.nii.gz')
        print(x0, x1, xm, np.max(interp_vol[:,int(np.floor(width/2)) ,:]))
        out[:,xm,:]=interp_vol[:,int(np.floor(width/2)) ,:] 
        
    return out
out_dir = '/project/def-aevans/tfunck/output/MR1/interslab/'
mri_fn = '/home/tfunck/srv/mri1_gm_bg_srv.nii.gz'
mri_rsl_fn='/home/tfunck/srv/mri1_gm_bg_srv_rsl-receptor.nii.gz'

slab_fn_list =  ['/project/def-aevans/tfunck/output/MR1/R_slab_4/final/musc_space-mni.nii.gz',
        '/project/def-aevans/tfunck/output/MR1/R_slab_5/final/musc_space-mni.nii.gz',
        '/project/def-aevans/tfunck/output/MR1/R_slab_6/final/musc_space-mni.nii.gz']

if not os.path.exists(out_dir) : os.makedirs(out_dir)

slabs = load_slabs(slab_fn_list)
mri_img = nib.load(mri_fn)
rec_aff = slabs[0].affine
print(rec_aff)
print(mri_img.affine)
mri = mri_img.get_data()
mri_ystart = mri_img.affine[1,3]
mri_ystep = mri_img.affine[1,1]

vox_width = 20

x_dim = slabs[0].shape[0]
y_dim = mri.shape[1] 
z_dim = slabs[0].shape[2]

print(y_dim)
out =np.zeros([x_dim,y_dim,z_dim])
out_interp=np.zeros([x_dim,y_dim,z_dim])
nslabs = len(slabs)
tfm_dict={}
for i in range(nslabs-1) :
    print('Anterior Slab:', i)
    j = i + 1

    #  pos (n) <-  ant (0)
    #    ____     ____     ____
    #   |    |   |    |   |    |
    #   | S2 |   | S1 |   | S0 |
    #   |____|   |____|   |____|
    #       ||   ||  ||   || 
    #        -> <-    -> <-                
    ant_ydim = slabs[i].shape[1]
    pos_ydim = slabs[j].shape[1]
    ant_ystart = slabs[i].affine[1,3]
    ant_ystep = slabs[i].affine[1,1]



    ant_0 = w2v( v2w(vox_width, ant_ystep, ant_ystart), mri_ystep , mri_ystart )
    ant_1 = w2v( v2w(0,ant_ystep, ant_ystart), mri_ystep , mri_ystart )
    ant_start=min(ant_0,ant_1)
    ant_end=max(ant_0,ant_1)
    rec_ant = slabs[i].get_data() 
    rec_pos = slabs[i+1].get_data() 
    print(ant_1)
    print(v2w(vox_width ,slabs[i].affine[1,1], slabs[i].affine[1,3]))
    rec_ant_y_start = w2v( v2w(vox_width ,slabs[i].affine[1,1], slabs[i].affine[1,3]),  mri_ystep , mri_ystart )
    print(rec_ant_y_start)
    out[:,rec_ant_y_start:(rec_ant_y_start+rec_ant.shape[1]),:] = rec_ant
    
    pos_ystart = slabs[j].affine[1,3]
    pos_ystep = slabs[j].affine[1,1]

    pos_0 = w2v( v2w( pos_ydim-vox_width, pos_ystep, pos_ystart), mri_ystep , mri_ystart ) 
    pos_1 = w2v( v2w(pos_ydim, pos_ystep, pos_ystart), mri_ystep , mri_ystart )
    pos_start=min(pos_0,pos_1)
    pos_end=max(pos_0,pos_1)
    

    ant_subslab=mri[:,ant_start:ant_end,:]
    pos_subslab=mri[:,pos_start:pos_end,:]

    ant_subslab_rec = rec_ant[:,0:vox_width ,:]
    pos_subslab_rec = rec_pos[:,(pos_ydim-vox_width):pos_ydim,:]

    ant_width = ant_0 - ant_1
    pos_width = pos_0 - pos_1
    start = min(ant_end, pos_start)
    end   = max(ant_end, pos_start)
    print(start, end)
    out_interp = subslab_alignment(start, end, mri, ant_subslab, pos_subslab, ant_subslab_rec, pos_subslab_rec, vox_width, out_dir, tfm_dict, out_interp, clobber=False)

nib.Nifti1Image(out, slabs[-1].affine).to_filename('test_intraslab.nii.gz')
nib.Nifti1Image(out_interp, slabs[-1].affine).to_filename('test_intraslab_interp.nii.gz')



