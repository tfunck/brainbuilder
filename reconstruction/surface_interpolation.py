import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
import stripy as stripy
import numpy as np
import vast.surface_tools as surface_tools
import ants
import tempfile
import time
from c_upsample_mesh import upsample
from nibabel.processing import resample_to_output
from utils.apply_ants_transform_to_obj import apply_ants_transform_to_obj
from re import sub
from glob import glob
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from ants import  from_numpy,  apply_transforms, apply_ants_transform, read_transform
from ants import image_read, registration
from utils.utils import shell, w2v, v2w
from vast.surface_volume_mapper import SurfaceVolumeMapper

global surf_fn_str
surf_fn_str='{}/mri1_{}_surface_right_{}{}.surf.gii'


def save_gii(coords, triangles, reference_fn, out_fn):
    img = nib.load(reference_fn) 
    ar1 = nib.gifti.gifti.GiftiDataArray(data=coords, intent='NIFTI_INTENT_POINTSET') 
    ar2 = nib.gifti.gifti.GiftiDataArray(data=triangles, intent='NIFTI_INTENT_TRIANGLE') 
    out = nib.gifti.GiftiImage(darrays=[ar1,ar2], header=img.header, file_map=img.file_map, extra=img.extra, meta=img.meta, labeltable=img.labeltable) 
    out.to_filename(out_fn) 
    print(out.print_summary())
 
def apply_ants_transform_to_gii( in_gii_fn, tfm_list, out_gii_fn, invert):
    print("transforming", in_gii_fn)
    faces, coords = nib.load(in_gii_fn).agg_data(('triangle', 'pointset'))

    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    if np.sum(tfm.fixed_parameters) != 0 : flip=-1
    
    in_file = open(in_gii_fn, 'r')
    
    path, ext = os.path.splitext(in_gii_fn)
    coord_fn = path+ '_ants_reformat.csv'

    #read the csv with transformed vertex points
    with open(coord_fn, 'w+') as f :  
        f.write('x,y,z,t,label\n') 
        for x,y,z in coords :  
            f.write('{},{},{},{},{}\n'.format(flip*x,flip*y,z,0,0 ))

    temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv'
    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert[0]}]  -o {temp_out_fn}',verbose=True)

    # save transformed surfaced as an gii file
    with open(temp_out_fn, 'r') as f :
        #read the csv with transformed vertex points
        for i, l in enumerate( f.readlines() ):
            if i == 0 : continue
            x,y,z,a,b = l.rstrip().split(',')
            coords[i-1] = [flip*float(x),flip*float(y),float(z)]
    
    save_gii(coords,faces,in_gii_fn,out_gii_fn)

def upsample_and_inflate_surfaces(surf_dir, wm_surf_fn, gm_surf_fn, resolution, depth_list, slab, n_vertices=81920) :
    # create depth mesh
    gm_dict = load_mesh_geometry(gm_surf_fn) 
    wm_dict = load_mesh_geometry(wm_surf_fn)
    
    d_coords = gm_dict['coords'] - wm_dict['coords'] 
    
    ngh = np.array([i for j in wm_dict['neighbours']  for i in j  ]).astype(np.int32)
    ngh_count = wm_dict['neighbour_count']
    
    for depth in depth_list :
        coords = wm_dict['coords'] + depth * d_coords
        print("Upsampling depth:", depth, np.max(ngh_count), np.max(ngh))
       
        #upsample surface and save in an intermediate .csv file
        upsample_fn="{}/surf_{}_{}mm_{}.csv".format(surf_dir,slab,resolution,depth)
        if not os.path.exists(upsample_fn)  :
            upsample(np.array(coords).flatten().astype(np.float32), 
             ngh, 
             np.array(ngh_count).flatten().astype(np.int32), 
             upsample_fn, float(resolution), 
             int(coords.shape[0]))
            
        #convert the upsampled .csv points into a proper gifti surfer
        surf_upsample_fn = "{}/surf_{}_{}mm_{}.surf.gii".format(surf_dir,slab,resolution,depth)
        if not os.path.exists(surf_upsample_fn)  :
            coords_rsl, ngh_rsl, faces_rsl = read_coords(upsample_fn,coords)
            save_gii( coords_rsl, faces_rsl, wm_surf_fn, surf_upsample_fn )
            del coords_rsl
            del ngh_rsl
            del faces_rsl
        
        exit(0)
        #inflate surface to sphere using freesurfer software
        surf_sphere_fn = "{}/surf_{}_{}mm_{}_inflate.surf.gii".format(surf_dir,slab,resolution,depth)
        if not os.path.exists(surf_sphere_fn):
            shell('~/freesurfer/bin/mris_inflate {} {}'.format(surf_upsample_fn, surf_sphere_fn))


def load_slabs(slab_fn_list) :
    slabs=[]
    for slab_fn in slab_fn_list :
        slabs.append( nib.load(slab_fn) )
    return slabs

def vol_surf_interp(src, coords, affine,  clobber=0 ):
    steps = [ affine[0,0], affine[1,1], affine[2,2]   ]
    starts =  [ affine[0,3], affine[1,3], affine[2,3]   ] #[-72, -126,-90 ] #
    vx=np.array([ w2v(i, steps[0], starts[0]) for i in coords[:,0]]).astype(int)
    vy=np.array([ w2v(i, steps[1], starts[1]) for i in coords[:,1]]).astype(int)
    vz=np.array([ w2v(i, steps[2], starts[2]) for i in coords[:,2]]).astype(int)

    idx = (vx >= 0) & (vy >= 0) & (vz >= 0) & (vx < src.shape[0]) & ( vy < src.shape[1]) & ( vz < src.shape[2] )
    if np.sum(idx) == 0 : 
        print('Error: no voxels found to interpolate over')
        exit(1)
    val=np.zeros(vx.shape)
    val[idx] = src[vx[idx], vy[idx], vz[idx]]#.reshape(-1,)
    df = pd.DataFrame({'x':coords[:,0], 'y':coords[:,1],'z':coords[:,2],'vx':vx,'vy':vy,'vz':vz,'val':val})
    return val


def write_nii(ar, fn, affine, dtype, clobber):
    if not os.path.exists(fn) or clobber >= 1 :
        nib.Nifti1Image(ar['data'][:].astype(dtype), affine).to_filename(fn)

def thicken_sections(array_src, slab_df, resolution ):
    width = np.round(resolution/(0.02*2)).astype(int)
    print('\t\tThickening sections to ',0.02*width*2)
    dim=[array_src.shape[0], 1, array_src.shape[2]]
    rec_vol = np.zeros_like(array_src)
    for row_i, row in slab_df.iterrows() : 
        i = int(row['volume_order'])
        
        # Conversion of radioactivity values to receptor density values
        section = array_src[:, i, :].reshape(dim) * row['conversion_factor']

        assert np.sum(section) !=0, f'Error: empty frame {i}'
        i0 = (i-width) if i-width > 0 else 0
        i1 = (i+width) if i+width <= array_src.shape[1] else array_src.shape[1]
        
        #put ligand sections into rec_vol
        print('i0 i1', i0,i1)
        rec_vol[:, i0:i1, :] = np.repeat(section,i1-i0, axis=1)
        
    assert np.sum(rec_vol) != 0, 'Error: receptor volume for single ligand is empty'

    return rec_vol


def read_coords(fn,orig_coords=None):
    coords_dict={}
    index_dict={}
    face_list=[]
    coords_list=[]
    with open(fn,'r') as F :
        for line_i , l in enumerate(F.readlines()) :
            coords_str = l.rstrip().split(",")
            coords = [ float(i) for i in coords_str[0:3]]
            ngh = [ int(i) for i in coords_str[3:6]]
            face = [ngh[0], ngh[0], ngh[1]]
            #face.sort()
            coords_list.append([coords])
            face_list.append([face])

    faces = np.unique(np.array(face_list),axis=0).reshape(-1,3)
    coords = np.unique(np.array(coords_list),axis=0).reshape(-1,3)
    #orig_coords=np.round(orig_coords,4).astype(np.float16)
    #coords = np.round(coords,4).astype(np.float16)

    print(orig_coords.shape, coords.shape) 
    #with open('test.csv','w') as F:
    #    for x,y,z in coords :
    #        F.write("{},{},{}\n".format(x,y,z))
    #with open('test2.csv','w') as F:
    #    for x,y,z in orig_coords :
    #        F.write("{},{},{}\n".format(x,y,z))
    tol=0.0001
    for coord in orig_coords :
        matches = coords[ ( abs(coords[:,0] - coord[0])<tol) & ( abs(coords[:,1] - coord[1])<tol) & (abs(coords[:,2] - coord[2])<tol) ] 
        nmatches = len(matches)
        #print(np.sum((coords[:,0] - coord[0]<tol)) , np.sum(coords[:,1]- coord[1]<tol) , np.sum(coords[:,2] - coord[2]<tol),np.sum((coords[:,0] - coord[0]<tol) & (coords[:,1] - coord[1]<tol) & (coords[:,2] - coord[2]<tol)) )
        if nmatches==0  :
            print("Missing coord:", coord)

    print("unique faces", faces.shape,"shape max",np.max(faces), "coords unique",coords.shape)
    exit(0)
    coords_list=[]
    ngh_list=[]
    index_map_dict={}
    index=0
    for x, y_dict in coords_dict.items():
        for y, z_dict  in y_dict.items():
            for z, ngh  in z_dict.items():
                coords_list.append([x,y,z])
                ngh_list.append( np.unique(ngh) )
                for orig_index in index_dict[x][y][z] : 
                    index_map_dict[orig_index] = index
                index += 1

    coords = np.array(coords_list)
    #faces = np.array(faces)
    remap=np.vectorize(lambda x: index_map_dict[x])
    print('old faces max:', np.max(faces))
    faces = remap(faces) 
    faces = np.unique(faces,axis=0)
    faces=faces.reshape(-1,3)
    print(faces.shape, np.max(faces))
    ngh_list = [ [ index_map_dict[x] for x in row] for row in ngh_list ]
    exit(0)
    return np.array(coords), ngh_list, faces.astype(np.double)

def get_slab_profile( slab_df, depth_index, surf_upsample_fn, array_src, affine, profiles, resolution):
    
    rec_vol = thicken_sections(array_src, slab_df, resolution)
    #nib.Nifti1Image(rec_vol, affine).to_filename('/project/def-aevans/tfunck/test.nii.gz')
    print('\t\t {}\n\t\t Depth:'.format( np.mean(np.sum(np.abs(wm_coords-gm_coords),axis=1)) ) )
    #for depth_i, depth_fn in enumerate(depth_fn_list) :
    coords, ngh, faces = read_coords(surf_upsample_fn)
    profiles[:,depth_i] += vol_surf_interp( rec_vol, coords, affine, clobber=2)
    if depth_i == 0 :
        idx = profiles[:,depth_i]>0
    return profiles
                 
def get_profiles(surf_dir, surf_mid_list, surf_wm_list, surf_gm_list, depth_list,  profiles_fn,vol_list, slab_list, df_ligand, surf_upsample_str, resolution ):
    dt = 1.0/ n_depths
    depth_list = np.arange(0., 1+dt, dt)
    
    nrows = pd.read_csv(surf_mid_list[0]).shape[0]

    profiles=np.zeros([nrows, len(depth_list)])

    for i, slab in enumerate(slab_list) :

        array_img = nib.load(vol_list[i])
        array_src = array_img.get_fdata()

        assert np.sum(array_src) != 0 , f'Error: input receptor volume has is empym {vol_list[i]}'

        for depth_index, depth in enumerate(depth_list):
            surf_upsample_fn=surf_upsample_str.format(surf_dir,slab,resolution,depth)
            print(f'\tslab: {slab}')
            slab_df=df_ligand.loc[df_ligand['slab'].astype(int)==int(slab)]
        
            profiles += get_slab_profile( slab_df, depth_index, surf_upsample_fn, array_src, array_img.affine, profiles, resolution)

    profiles[ profiles < 0.1 ] = 0
   
    profiles_raw_fn = sub('.csv','_raw.csv', profiles_fn) 
    pd.DataFrame(profiles).to_csv(profiles_raw_fn, index=False, header=False)

    profiles = interpolate_over_surface(sphere_obj_fn,profiles)

    pd.DataFrame(profiles).to_csv(profiles_fn, index=False, header=False)
     
def interpolate_over_surface(sphere_obj_fn,profiles):
        
    sphere = load_mesh_geometry(sphere_obj_fn) 

    # get coordinates from dicitonary with mesh info
    spherical_coords = surface_tools.spherical_np(sphere['coords'])

    for i in range(profiles.shape[1]):
        surface_val = profiles[:,i]

        #define a mask of verticies where we have receptor densitiies
        surface_mask = profiles[:,i] != 0

        #define vector with receptor densities 
        surface_val_src = surface_val[ surface_mask.astype(bool) ]

        #define vector without receptor densities
        surface_val_tgt = surface_val[ ~surface_mask.astype(bool) ]

        # get coordinates for vertices in mask
        spherical_coords_src = spherical_coords[ surface_mask.astype(bool), : ]

        # get spherical coordinates from cortical mesh vertex coordinates
        lats_src, lons_src = spherical_coords_src[:,1]-np.pi/2, spherical_coords_src[:,2]

        # create mesh data structure
        mesh = stripy.sTriangulation(lons_src, lats_src)
        lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]

        # interpolate over the sphere
        interp_val, interp_type = mesh.interpolate(lons,lats, zdata=surface_val[surface_mask], order=1)
            
        profiles[:,i] = interp_val

    return profiles

def transform_surf_to_slab(interp_dir, surf_gm_fn, surf_wm_fn, brain, hemi, tfm_list, slab_list, clobber=0):
    surf_rsl_dict={}
    for i, slab in enumerate( slab_list ) :
        gm_rsl_fn="{}/slab-{}_{}".format(interp_dir,slab,os.path.basename(surf_gm_fn))
        if not os.path.exists(gm_rsl_fn) or clobber >= 1 : 
            apply_ants_transform_to_gii(surf_gm_fn, [tfm_list[i]], gm_rsl_fn, [0])
    
        wm_rsl_fn="{}/slab-{}_{}".format(interp_dir,slab,os.path.basename(surf_wm_fn))
        if not os.path.exists(wm_rsl_fn) or clobber >= 1 : 
            apply_ants_transform_to_gii(surf_wm_fn, [tfm_list[i]], wm_rsl_fn, [0])

        surf_rsl_dict[slab]={'wm':wm_rsl_fn, 'gm':gm_rsl_fn}
        
    return surf_rsl_dict

def surface_interpolation(tfm_list, vol_list, slab_list, out_dir, interp_dir, brain, hemi, resolution, df, mni_fn, n_depths=3, surf_dir='civet/mri1/surfaces/surfaces/', n_vertices = 327696, clobber=0):
    #make sure resolution is interpreted as float
    resolution=float(resolution) 

    surf_rsl_dir = interp_dir +'/surfaces/' 
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(surf_rsl_dir, exist_ok=True)
    
    #Interpolate at coordinate locations
    surf_gm_fn = surf_fn_str.format(surf_dir,'gray', n_vertices,'')
    surf_wm_fn = surf_fn_str.format(surf_dir,'white', n_vertices,'')
    sphere_obj_fn = surf_fn_str.format(surf_dir,'mid', n_vertices,'_sphere')
    

    print("\tGet surface mask and surface values")
    # Load dimensions for output volume
    starts = np.array([-72, -126,-90 ])
    mni_vol = nib.load(mni_fn).get_fdata()
    dwn_res = 0.25
    nat_res = 0.02

    #set depths
    dt = 1.0/ n_depths
    depth_list = np.arange(dt, 1, dt)

    dimensions = np.array([ mni_vol.shape[0] * dwn_res/resolution, 
                            mni_vol.shape[1] * dwn_res/resolution, 
                            mni_vol.shape[2] * dwn_res/resolution]).astype(int)
    del mni_vol

    print("\tTransforming surfaces to slab space") 
    #For each slab, transform the mesh surface to the receptor space
    #TODO: transform points into space of individual autoradiographs
    surf_rsl_dict = transform_surf_to_slab(surf_rsl_dir, surf_gm_fn, surf_wm_fn, brain, hemi, tfm_list,  slab_list)

    #surf_upsample_fn = '{}/surf_{}_{}mm_{}.surf.gii'
    #surf_upsample_str = '{}/surf_{}_{}mm_{}.obj'

    #upsample transformed surfaces to given resolution
    for slab, surf_dict in surf_rsl_dict.items() :
        upsample_and_inflate_surfaces(surf_rsl_dir, surf_dict['wm'],surf_dict['gm'], resolution, depth_list, slab)
    exit(0)
    
    # Create an object that will be used to interpolate over the surfaces
    mapper = SurfaceVolumeMapper(white_surf=surf_wm_fn, gray_surf=surf_gm_fn, resolution=[resolution]*3, mask=None, dimensions=dimensions, origin=starts, filename=None, save_in_absence=False, interp_dir=interp_dir )

    for ligand, df_ligand in df.groupby(['ligand']):
        print('\tInterpolating for ligand:',ligand)
        profiles_fn  = f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm_profiles.csv'
        interp_fn  = f'{interp_dir}/{brain}_{hemi}_{ligand}_{resolution}mm.nii.gz'

        # Extract profiles from the slabs using the surfaces 
        if not os.path.exists(profiles_fn) or clobber >= 1 :
            get_profiles( surf_rsl_dir, depth_list, profiles_fn, vol_list, slab_list, df_ligand, surf_filename_str, resolution)
            
        # Interpolate a 3D receptor volume from the surface mesh profiles
        if not os.path.exists(interp_fn) or clobber : 

            print('Map Vector to Block')
            profiles = pd.read_csv(profiles_fn, header=None).values
            vol_interp = mapper.map_profiles_to_block(profiles)

            assert np.sum(vol_interp) != 0 , 'Error: interpolated volume is empty'

            receptor_img = nib.Nifti1Image(vol_interp, np.array([[resolution, 0, 0, starts[0]],
                                                                 [0, resolution, 0, starts[1]],
                                                                 [0, 0, resolution, starts[2]],
                                                                 [0, 0, 0, 1]]) )

            print(f'\n\tResample interpolated volume to {resolution}')
            receptor_img_rsl = resample_to_output(receptor_img, [resolution]*3, order=1)

            print(f'\tWrite volumetric interpolated values to {interp_fn} ',end='\t')
            receptor_img.to_filename(interp_fn)
            #receptor_img_rsl.to_filename(interp_fn)
            print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')
    parser.add_argument('--brain', dest='brain', type=str, help='brain')
    parser.add_argument('--hemi', dest='hemi', type=str,  help='hemi')
    parser.add_argument('--out-dir', dest='out_dir', type=str,  help='Clobber results')
    parser.add_argument('--nl-tfm-str', dest='nl_tfm_str', type=str,  help='Clobber results')
    parser.add_argument('--lin-df-fn', dest='lin_df_fn', type=str,  help='Clobber results')
    parser.add_argument('--slab-str', dest='slab_str', type=str,  help='Clobber results')
    parser.add_argument('--n-depths', dest='n_depths', type=int,  default=8, help='Clobber results')
    args = parser.parse_args()
    
    lin_df_fn = args.lin_df_fn
    out_dir = args.out_dir
    brain = args.brain
    hemi = args.hemi
    n_depths = args.n_depths
    nl_tfm_str = args.nl_tfm_str
    slab_str = args.slab_str
    clobber=args.clobber
