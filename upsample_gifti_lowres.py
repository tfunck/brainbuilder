
from utils.utils import get_section_intervals, apply_ants_transform_to_gii


# transform native surface to each of the slab spaces
def get_ngh(triangles):
    d={}
    for i,j,k in triangles:
        d = add_entry(d,i,[j,k])
        d = add_entry(d,j,[i,k])
        d = add_entry(d,k,[j,i])

    for key in d.keys():
        #d[key] =np.unique(d[key])
        d[key] = list(np.array(np.unique(d[key])))
    del triangles 
    return d 

def upsample_within_slab(ligand_vol_fn, surf_fn, nl_3d_tfm_fn,mni_fn, ext='.nii.gz'):
    surf_slab_space_fn =  
    
    img = nib.load(ligand_vol_fn)
    ligand_vol = img.get_fdata()
    step = img.affine[1,1]
    start = img.affine[1,3]

    apply_ants_transform_to_gii(upsample_fn, [nl_3d_tfm_fn], surf_slab_space_fn, 0, upsample_fn, upsample_fn, ext, mni_fn)

    intervals_voxel = get_section_intervals(ligand_vol)

    coords, faces, surf_info = load_mesh(surf_fn)

    edges = get_edges_from_faces(faces)

    edge_y_coords = np.hstack([coords[:,edges[:,0],:],coords[:,edges[:,1],:]])
    idx_1 = np.argsort(edge_y_coords,axis=1)
    sorted_edge_y_coords = edge_y_coords[idx_1]
    edges = edges[idx_1]

    idx_0=np.argsort(sorted_edge_y_coords)
    sorted_edge_y_coords=sorted_edge_y_coords[idx_0]
    edges=edges[idx_0]

    ligand_y_profile = np.sum(ligand_vol,axis=(0,2))
    section_numbers = np.where(ligand_y_profile > 0)
    
    section_counter=0
    curr_section_vox = section_numbers[section_counter]
    curr_section_world = curr_section_vox * step + start

    edges_to_split = np.zeros_like(edges)
    
    for i in sorted_edges_idx:
        y0,y1 = sorted_edge_y_coords[i]
        e0,e1 = edges[i]
        
        if y0 <= current_section_world & y1 > current_section_world :
            edges_to_split[i]=2
        
        if y0 > current_section_world :
            section_counter += 1
            if section_counter >= section_counter.shape :
               somethingsomething 
            current_section_vox = section_numbers[section_counter]
            current_section_world = current_section_vox * step + start


    #   
    #   for each section:
    #       apply map to edges (that have not been flagged) to test if edge crosses section
    #       flag crossing edges
    #       flag non-crossing edges whose origin is before s0

def upsample_gifti_lowres(mni_fn):
    for slab in slab_list :
        nl_3d_tfm_fn
        upsample_within_slab(surf_fn, nl_3d_tfm_fn, mni_fn)


    #       
    #   apply one pass of upsample_gifti.py for crossing edges
