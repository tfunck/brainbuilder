import utils.ants_nibabel as nib 
import nibabel 
import numpy as np 
import nibabel

fn="templates/MEBRAINS_T1.nii.gz"

def write_gii(coords,faces,output_filename): 
    gif = nibabel.gifti.gifti.GiftiImage()
    gif.add_gifti_data_array( nibabel.gifti.gifti.GiftiDataArray(coords,'NIFTI_INTENT_POINTSET', coords.dtype) )
    gif.add_gifti_data_array( nibabel.gifti.gifti.GiftiDataArray(faces,'NIFTI_INTENT_TRIANGLE', faces.dtype) )

    nibabel.gifti.giftiio.write(gif, output_filename)

def swap_axis(coords,i,j):
    col = coords[:,i].copy()
    coords[:,i] = coords[:,j]
    coords[:,j] = col
    return coords

def fix_surface(coords):
    coords = swap_axis(coords, 1,2) #02,01
    coords[:,1] *= -1
    coords[:,0] =  -1 * coords[:,0]
    return coords

def fix_orientation(fn, out_fn):
    img = nib.load(fn) 

    vol = img.get_fdata() 
    vol = np.flip(vol,axis=(0,1,2))
    nib.Nifti1Image(vol, img.affine).to_filename(out_fn) 

if __name__ == '__main__':
    fn_list = (("templates/MEBRAINS_mask.nii.gz", "templates/MEBRAINS_mask_RAS.nii.gz"),
                ("templates/MEBRAINS_T1.nii.gz", "templates/MEBRAINS_T1_RAS.nii.gz"),
                ("templates/MEBRAINS_segmentation_NEW.nii.gz", "templates/MEBRAINS_segmentation_NEW_RAS.nii.gz"))

    #for fn1, fn2 in fn_list:
    #    fix_orientation(fn1, fn2)

    #fn="templates/MEBRAINS_segmentation_NEW_RAS.nii.gz"
    fn="templates/MEBRAINS_segmentation_NEW.nii.gz"
    img = nib.load(fn)
    data = img.get_fdata()
    r = np.copy(data) 
    l = np.copy(data) 
    r[ r != 2017 ] = 0
    l[ l != 1017 ] = 0
    nib.Nifti1Image(l,img.affine,direction_order='lpi').to_filename("templates/MEBRAINS_segmentation_NEW_gm_left.nii.gz")
    nib.Nifti1Image(r,img.affine,direction_order='lpi').to_filename("templates/MEBRAINS_segmentation_NEW_gm_right.nii.gz")
    exit(0)
    from nibabel.freesurfer.io import read_geometry, write_geometry, write_annot, write_morph_data
    gm_coords, gm_faces = read_geometry('templates/template_surfaces/lh.pial')
    wm_coords, wm_faces = read_geometry('templates/template_surfaces/lh.white')

    
    gm_coords = fix_surface(gm_coords)
    wm_coords = fix_surface(wm_coords)
    write_geometry('templates/template_surfaces/lh_ras.pial',gm_coords,gm_faces)
    write_geometry('templates/template_surfaces/lh_ras.white',wm_coords,wm_faces)
    write_morph_data('templates/template_surfaces/y.curv', wm_coords[:,1], wm_faces.shape[0] )
    write_morph_data('templates/template_surfaces/x.curv', wm_coords[:,0].astype(np.float32), wm_faces.shape[0] )

    #write_gii(gm_coords, gm_faces, 'templates/template_surfaces/lh_ras_pial.surf.gii')
    #write_gii(wm_coords, wm_faces, 'templates/template_surfaces/lh_ras_white.surf.gii')
