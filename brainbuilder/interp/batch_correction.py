import os
import pandas as pd
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import seaborn as sns

import brainbuilder.utils.ants_nibabel as nib

from brainbuilder.utils.mesh_utils import load_values, load_mesh_ext, visualization


def apply_batch_correction(
        sect_info:pd.DataFrame,
        profiles_fn:str,
        surf_depth_mni_dict:dict,
        surf_depth_chunk_dict:dict,
        chunk_info:pd.DataFrame,
        depth_list:list,
        resolution:float,
        struct_vol_rsl_fn:str,
        output_dir:str, 
        clobber:bool=False) -> (pd.DataFrame, str) :
    '''
    apply batch correction to the profiles_fn to correct for mean shifts between chunks
    :param sect_info: dataframe containing section information
    :param profiles_fn: path to the profiles file
    :param surf_depth_mni_dict: dictionary containing information about the surfaces
    :param surf_depth_chunk_dict: dictionary containing information about the surfaces
    :param chunk_info: dataframe containing chunk information
    :param depth_list: list of depths to use for interpolation
    :param output_dir: path to output directory
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    '''

    mid_depth_index = int(np.ceil(len(depth_list)/2))	
    mid_depth = depth_list[mid_depth_index]

    mid_depth_chunk_dict = {k: v for k, v in surf_depth_chunk_dict.items() if k == mid_depth}

    surface_labels_filename, df_ligand, label_to_slab_dict = create_surface_section_labels(
            df_ligand, 
            values_filename, 
            mid_depth_chunk_dict, 
            output_dir, 
            qc_surface_fn = surf_depth_mni_dict[mid_depth]['depth_rsl_fn'],
            clobber=clobber) 

    params, _ = calc_batch_correction_params(
        surface_labels_filename,
        values_filename, 
        surf_depth_mni_dict[mid_depth]['sphere_rsl_fn'], #stereo_sphere_filename
        surf_depth_mni_dict[mid_depth]['depth_rsl_fn'], #stereo_cortex_filename
        f'{output_dir}/batch_correction/',
        clobber=clobber)

    sect_info = update_df_with_correction_params(sect_info, params, label_to_slab_dict)

    # recreate profiles_fn with batch corrected values
    profiles_fn = generate_surface_profiles(
            chunk_info, 
            sect_info, 
            surf_depth_chunk_dict,
            surf_depth_mni_dict,
            resolution, 
            depth_list,
            struct_vol_rsl_fn,
            output_dir, 
            clobber = clobber )


    return sect_info, profiles_fn


def create_surface_section_labels(
        acquisition_df:pd.DataFrame,
        values_filename:str,
        chunk_surface_dict:dict,
        out_dir:str,
        qc_surface_fn:str='',
        clobber:bool=False):

    acquisition = sect_info['acquisition'].values[0]
    out_fn = f'{out_dir}/{acquisition}_section_labels'
    qc_fn = f'{out_dir}/{acquisition}_section_labels_qc.png'
  
    values = load_values(values_filename).reshape(-1,)

    label_to_chunk_dict={}

    sect_info['label'] = [0] * sect_info.shape[0] #- np.min(labels_for_df) +1

    labels=np.array([])

    if not os.path.exists(out_fn) or clobber:
        
        max_chunk = sect_info['chunk'].max() + 1

        for chunk, chunk_df in sect_info.groupby(['chunk']) :
            surf_filename = chunk_surface_dict.surfaces[chunk]
            coords = load_mesh_ext(surf_filename)[0]
            
            y = coords[:,1]
            
            ymin = np.min(chunk_df['y0w'])
            ymax = np.max(chunk_df['y1w'])

            ylo = ymin+(ymax-ymin)*0.25
            yhi = ymin+(ymax-ymin)*0.75

            for i, row in chunk_df.iterrows() :
                if len(labels) == 0 :
                    labels = np.zeros(coords.shape[0])

                if row['y0w'] < ylo :
                    label = (max_chunk - row['chunk'])*2 - 1
                    label_to_chunk_dict[label] = chunk
                elif row['y1w'] > yhi :
                    label = (max_chunk - row['chunk'])*2 
                    label_to_chunk_dict[label] = chunk
                else:
                    label=0
                
                sect_info['label'].loc[ (sect_info['chunk'] == chunk) & (sect_info['y0w'] == row['y0w']) ]  = label

                idx = (y > row['y0w']) & (y <= row['y1w']) & (values > np.min(values) ) 
                labels[idx] = label
       
        np.savez(out_fn, data=labels)
        
        if qc_surface_fn & isinstance(qc_surface_fn, str) & os.path.exists(qc_surface_fn):
            visualization(qc_surface_fn, labels, qc_fn)


   
    return out_fn, sect_info, label_to_chunk_dict


def update_df_with_correction_params(
        df: pd.DataFrame,
        params: pd.DataFrame,
        label_to_slab_dict : dict 
        ) -> pd.DataFrame :
    '''
    Update the dataframe with the correction parameters
    :param df: dataframe containing section information
    :param params: dataframe containing the correction parameters
    :param label_to_slab_dict: dictionary containing the mapping between labels and slabs
    :return: dataframe with updated correction parameters
    '''

    print('Update df with correction parameters') 
    df['batch_offset'] = [0]*df.shape[0]
    df['batch_multiplier']=[1]*df.shape[0]

    for i, row in params.iterrows() :

        idx = df['slab'] == label_to_slab_dict[row['label'].astype(int)]

        df['batch_offset'].loc[ idx ] = row['offset']
        df['batch_multiplier'].loc[ idx ] = row['multiplier']

        slab = df['slab'].loc[idx].values[0]
        print('slab', slab, 'label', row['label'], 'offset', row['offset'])
    
    for slab, slab_df in  df.groupby('slab'):
        print(slab)
        print(slab_df['batch_offset'])
    return df 

def get_border_y(df, slab, thr, start, step, caudal = True):
    dfs = df.sort_values(['slab_order']) 
    ymax = dfs['slab_order'].max()
    
    if caudal:
        dfs = dfs.iloc[0:thr]
        y_list = (dfs['slab_order'].min(), dfs['slab_order'].max()+1) ##add +1 so that it is inclusive range
    else : #rostral
        dfs = dfs.iloc[ -thr: ] 
        y_list = (dfs['slab_order'].min(), dfs['slab_order'].max()+1 )

    y_list = sorted([y*step+start for y in y_list])
    #need to convert y to world coordinates

    return y_list

global CAUDAL_LABEL 
global ROSTRAL_LABEL

CAUDAL_LABEL = 0.33
ROSTRAL_LABEL= 0.66

import numpy as np

def interleve(x,step):
    '''

    '''
    return np.concatenate([ x[i:x.shape[0]+1:step] for i in range(step) ] )

def magnitude(X,Y):
    D = np.power(X - Y, 2)
    if  len(D.shape)==1:
        D=np.sum(D)
    else :
        D=np.sum(D, axis=1)
    return np.sqrt(D)

def pairwise_coord_distances(c0:np.ndarray, c1:np.ndarray) -> np.ndarray:
    '''
    calculate the pairwise distance between
        c0: 3d coordinates
        c1: 3d coordinates
        
    '''
    c0 = c0.astype(np.float16)
    c1 = c1.astype(np.float16)
    try :
        X = np.repeat(c0, c1.shape[0], axis=0)
        # create array with repeated columns of c0
        Y = interleve(np.repeat(c1, c0.shape[0], axis=0), c0.shape[0])
        # 
        D = magnitude(X,Y).reshape(c0.shape[0],c1.shape[0])
    except MemoryError:
        print('Warning: OOM. Defaulting to slower pairwise distance calculator', end='...\n')
        D=np.zeros([c0.shape[0],c1.shape[0]], dtype=np.float16)
        for i in range(c0.shape[0]) : #iterate over rows of co
            if i%100==0: print(f'Completion: {np.round(100*i/c0.shape[0],1)}',end='\r')
            #calculate magnitude of point in row i of c0 versus all points in c1
            D[i] = magnitude(c1,c0[i])

    return D
 

if False:   # test to make sure that pairsewise_coord_distances works correctly 
    def calc2(c0,c1):
         d=np.zeros([c0.shape[0], c1.shape[0]])
         for i, x in enumerate(c0) :
             for j, y in enumerate(c1) :
                  d[i][j] = np.sqrt(np.sum(np.power(x-y,2)))
         return d
    c0 = np.random.normal(0,1,(10,3))
    c1 = np.random.normal(1,1,(14,3))
    d = pairwise_coord_distances(c0,c1)
    d1 = calc2(c0,c1)
    print(d - d1)
    print(d[0])
    print(d1[0])
    assert np.sum(np.abs(d - d1)) < 1e-8, 'Error: pairwise_coord_distances function failed to return correct distances'
    print(d.shape)
    exit(1)

def average_vertex_values(coords, values, smoothing_percentile, total=None, n=None, p=2):
    # create a 2d pairwise distance matrix
    dist_matrix = pairwise_coord_distances(coords, coords)
    smoothing_radius = np.percentile(dist_matrix, [smoothing_percentile], axis=1).T
    n_values =  dist_matrix.shape[0] 
    values_vector = np.repeat(values, n_values)
    values_matrix = np.reshape(values_vector, [n_values,n_values])

    idx = dist_matrix<=smoothing_radius
    #wgts = dist_matrix.copy()
    
    #wgts[ ~idx ] = 0
    values_matrix[~idx] = 0
    
    wgts = np.sum(idx,axis=1)
    print(wgts)
    #den = np.sum(wgts,axis=1)
    num = np.sum(values_matrix/wgts, axis=1).reshape(coords.shape[0])
    smoothed_values = num 
    
    return smoothed_values



def get_rostral_caudal_borders(label_surf_values, surfaces, out_dir,  perc=0.05, clobber=False):
    border_fn = f'{out_dir}/borders' 

    #WARNING the label volumes and surfaces must be in same coord space
    if not os.path.exists(border_fn) or clobber :

        labels = load_values(label_surf_values)
        slabs = np.unique(labels[labels>0])
        n_slabs=len(slabs)
        
        assert type(surfaces) == list, 'Error: surfaces is not a list'
        if len(surfaces) == 1:
            surfaces = surfaces * n_slabs

        n_vtx = load_mesh_ext(surfaces[0])[0].shape[0]

        border=np.zeros(n_vtx)
        
        for i, slab in enumerate(slabs) :
            
            coords = load_mesh_ext(surfaces[i])
            y=coords[:,1]
            
            ymin=np.min(y)
            ymax=np.max(y)

            y0 = np.min(y[labels==slab])
            y1 = np.max(y[labels==slab])
            y = coords[:,1]

            perc=0.25

            c0 = y0
            c1 = y0 + np.abs(y0-y1)*perc

            r0 = y1 - np.abs(y1-y0)*perc
            r1 = y1

            caudal_idx = (y > c0 ) & ( y < c1)
            assert np.sum(caudal_idx)>0, f'Error: no vertices found between {c0} and {c1}'

            rostral_idx = (y > r0 ) & ( y < r1 ) 
            assert np.sum(rostral_idx)>0, f'Error: no rostral vertices found between {r0} and {r1}'

            border[caudal_idx] = slab + CAUDAL_LABEL
            border[rostral_idx] = slab + ROSTRAL_LABEL

        
        np.savez(border_fn, data=border)
    return border_fn

        

        



def exclude_poorly_aligned_regions(coords, steps, starts, ref_vol):
    vox = np.rint( (coords- starts)/steps).astype(int)
    
    try :
        dice = ref_vol[ vox[0], vox[1], vox[2] ] # vox[:,0], vox[:,1], vox[:,2] ]
    except IndexError:
        print('\t\tWarning: Vertex not in slab')
        dice=0

    return dice 

def read_ref_info(vol_fn,surf_fn):
    ref_img = nibabel.load(vol_fn)
    ref_vol = ref_img.get_fdata() 
    starts = ref_img.affine[0:3,3]
    steps = ref_img.affine[[0,1,2],[0,1,2]]
    surf_coords = load_mesh_ext(surf_fn)[0]
    return ref_img, ref_vol, steps, starts, surf_coords



def create_dataframe_for_pairs(dist,avg_values, next_range, c_vtx):

    arg_m = np.argmin(dist, axis=1) 
    min_dist = dist[range(dist.shape[0]),arg_m]
    del dist

    n_vtx = next_range[arg_m]
   
    next_values = avg_values[n_vtx]
    curr_values = avg_values[c_vtx]
    print('avg ros', np.mean(curr_values), 'avg cau', np.mean(next_values) ) 
    print('diff', np.mean(np.abs(curr_values-next_values)) )

    paired_values = pd.DataFrame({ 
                            'curr_idx': c_vtx,
                            'next_idx': n_vtx,
                            'curr_values': curr_values,
                            'next_values': next_values,
                            'distance': min_dist
                        })
    return paired_values


def find_pairs_between_labels(curr_label, next_label, stereo_sphere_coords, labels, avg_values ):

    curr_idx = labels == float(curr_label) #+ ROSTRAL_LABEL)
    next_idx = labels == float(next_label) #+ CAUDAL_LABEL )

    vtx_range = np.arange(stereo_sphere_coords.shape[0]).astype(int)
    
    curr_range = vtx_range[curr_idx]
    next_range = vtx_range[next_idx]

    next_coords = stereo_sphere_coords[next_idx]
    curr_coords = stereo_sphere_coords[curr_idx]

    dist = pairwise_coord_distances(curr_coords, next_coords) 
    
    paired_values = create_dataframe_for_pairs(dist, avg_values, next_range, curr_range)
    
    paired_values['curr_label'] = [curr_label] * paired_values.shape[0]
    paired_values['next_label'] = [next_label] * paired_values.shape[0]
    paired_values['vtx_pair_id'] = np.arange(paired_values.shape[0]) 

    return paired_values

def get_paired_values(paired_values_csv, unique_paired_values_csv,  stereo_sphere_filename, label_filename, values_fn, out_dir, label_start=1, label_offset=2, clobber=False):
    
    if not os.path.exists(paired_values_csv) or clobber:
        # find corresponding points between caudal and rostral 
        depth_values=[] 
       
        labels = load_values(label_filename)
        assert np.sum(labels>0), 'Error: no labels in '+label_filename

        stereo_sphere_coords = load_mesh_ext(stereo_sphere_filename)[0]
        slabs = np.unique(labels.astype(int))[1:]
        max_slab = np.max(slabs)
        n_slabs = len(slabs)

        avg_values = load_values(values_fn).reshape(labels.shape)
        
        # Unlabel (i.e. set to 0) labeled vertices that have a value of 0
        labels[ avg_values==0 ] = 0
        assert np.sum( avg_values[labels>0] == 0) == 0, 'Error: 0 values found in vertices that shouldnt be zero, ie where labels >0 '

        # examples of slab structure
        #   | 1 |   | 2 |   | 3 |
        #   c   r   c   r   c   r
        #   1   2   3   4   5   6

        label_list = np.unique(labels)[1:]
        paired_values = pd.DataFrame({})
        curr_labels = label_list[label_start:-1:label_offset]
        next_labels = label_list[label_start+1::label_offset]

        for curr_label, next_label in zip(curr_labels, next_labels) :
            print('curr label', curr_label, next_label)
            curr_paired_values = find_pairs_between_labels(curr_label, next_label, stereo_sphere_coords, labels, avg_values)
            paired_values=paired_values.append(curr_paired_values)

        paired_values.to_csv(paired_values_csv, index=False)
    
    paired_values = pd.read_csv(paired_values_csv, index_col=False)
     
    if not os.path.exists(unique_paired_values_csv) or clobber :
        unique_paired_values = get_unique_pairs(paired_values, 'curr')
        unique_paired_values = get_unique_pairs(unique_paired_values, 'next')
        unique_paired_values.to_csv(unique_paired_values_csv, index=False)
    unique_paired_values = pd.read_csv(unique_paired_values_csv, index_col=False)

    return unique_paired_values 


def get_unique_pairs(paired_values, direction, n_points=1) :
    unique_paired_values=pd.DataFrame({})

    for i_slab, i_df in paired_values.groupby([f'{direction}_label']) :
    
        for counter, (j_idx, j_df) in enumerate(i_df.groupby([f'{direction}_idx'])) :
            min_distance_idx = np.argsort(j_df['distance'])[:n_points]
            row = j_df.iloc[min_distance_idx]
                
            unique_paired_values = unique_paired_values.append(row)

    return unique_paired_values



def draw_pair_plots(surf_coords:np.ndarray, paired_values:type(pd.DataFrame), out_fn:str, hue_string_0:str='vtx_pair_id', hue_string_1:str='vtx_pair_id') -> None:

    d_list = []

    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
    fig, axes = plt.subplots(2,3, figsize=(40,10))

    for ax in axes.ravel() :
        ax.grid(False)
        for i, (curr_slab, tdf) in enumerate(paired_values.groupby(['curr_label'])): 
            ax = axes.ravel()[i]

            cidx = tdf['curr_idx'].values.astype(int)
            ridx = tdf['next_idx'].values.astype(int)

            xc = surf_coords[cidx,0]
            zc = surf_coords[cidx, 2]
            xr = surf_coords[ridx, 0]
            zr = surf_coords[ridx, 2]
            
            o = 1.5*(np.max(xc) - np.min(xc))
            ax.set_title(f'Current Slab {curr_slab}')
            sns.scatterplot(x=xc, y=zc, hue=tdf[hue_string_0], palette='nipy_spectral',alpha=0.3,ax=ax)
            sns.scatterplot(x=o+xr, y=zr, hue=tdf[hue_string_1], palette='nipy_spectral', alpha=0.3,ax=ax)
            sns.despine(left=True, bottom=True)
            ax.grid(False)
            ax.get_legend().remove()

    print('\tWriting', out_fn)
    plt.savefig(out_fn)


def qc_paired_values(out_dir,paired_values, surf_coords, clobber=False):


    idx_pairs_fn = f'{out_dir}/validate_paired_idx.png'
    values_pairs_fn = f'{out_dir}/validate_paired_values.png'
    out_vol_fn = f'{out_dir}/validated_paired_distances.nii.gz'
    clobber=True
    if  not os.path.exists(idx_pairs_fn) or clobber :
        draw_pair_plots(surf_coords, paired_values, idx_pairs_fn)
    if  not os.path.exists(values_pairs_fn) or clobber :
        draw_pair_plots(surf_coords, paired_values, values_pairs_fn, hue_string_0='curr_values', hue_string_1='next_values')


def simple_slab_correction(paired_values):

    params = pd.DataFrame({})

    row_dict = {    'label': paired_values['curr_label'].min(),
                    'multiplier': [1],
                    'offset': [0] }

    params = params.append(pd.DataFrame(row_dict))

    total_offset = 0

    #
    #   5   10   20
    #   cu  ne1 ne2 
    # d0 = cu-ne1 = 5-10=-5 --> ne1 offset => -5 
    # d1 = ne2 -ne1 = 10 - 20 = -10 --> ne2 offset => -10 + ne1 offset = -15
    plt.figure(figsize=(12,12))

    for curr_label, df in paired_values.groupby(['curr_label']):
        print('curr label', curr_label)
        #print(df)
        n=df.shape[0]
        next_label = df['next_label'].values[0]
        curr_values = df['curr_values'].values
        next_values = df['next_values'].values
        
        print('cur vs next values', np.mean(curr_values), np.mean(next_values))

        diff2 = curr_values - next_values

        mean_diff = np.mean( diff2 )

        error = np.sum(np.abs(curr_values-(next_values+mean_diff)))/curr_values.shape[0]
        offset = mean_diff
        print('offset', offset, np.mean(curr_values) - np.mean(next_values), total_offset)
        print('error', error)
        total_offset += offset

        row_dict = {    'label': [next_label],
                        'multiplier': [1],
                        'offset': [total_offset] }

        params = params.append(pd.DataFrame(row_dict))

        X = [(curr_label, next_label)]*n
        Y = np.column_stack([curr_values, next_values])
    
        
        for x, y in zip(X,Y):
            plt.scatter(x,y,c='r', alpha=0.5)
            plt.plot(x, y, c='b', alpha=0.3)
    exit(0)
    plt.savefig('/tmp/tmp.png')
    return params

        
def calc_batch_correction_params(
        label_filename:str,
        values_filename:str, 
        stereo_sphere_filename:str,
        stereo_cortex_filename:str,
        out_dir:str,
        clobber=False
        ):
    '''
        Identify the vertices within a given slab and find the vertices that are part of the anterior and posterior border of that slab.

    :param label_filename: file with a scalar for each vertx where chunks are labeled with discrete integer values. 
    :param values_filename: file with scalar for each vertex. values that need to be corrected
    :param stereo_sphere_filename: surface mesh inflated to sphere
    :param stereo_cortex_filename: surface mesh of cortex
    :param out_dir: output directory
    :param clobber: overwrite existing files
    :return: params, paired_values
    '''
    paired_values_csv = f'{out_dir}/paired_values.csv'
    unique_paired_values_csv = f'{out_dir}/unique_paired_values.csv'
    pairs_fn = f'{out_dir}/validate_paired_values.png'
    params_fn = f'{out_dir}/params.csv'

    nvtx = load_mesh_ext(stereo_sphere_filename)[0].shape[0]
    vtx_range = np.arange(nvtx).astype(int)
    labels = np.unique( load_values(label_filename))[1:]
    n_labels = len(labels)

    print('\t\tGet Paired Values')
    paired_values = get_paired_values(paired_values_csv, unique_paired_values_csv,  stereo_sphere_filename, label_filename, values_filename, out_dir, clobber=clobber)

    if not os.path.exists(params_fn) or clobber :
        params = simple_slab_correction(paired_values)
        params.to_csv(params_fn,index=False)
    params = pd.read_csv(params_fn)

    qc_paired_values(out_dir,paired_values, load_mesh_ext(stereo_cortex_filename)[0], clobber=clobber)

    return params, paired_values
