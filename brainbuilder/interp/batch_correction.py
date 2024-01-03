import os
import pandas as pd
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel 

from brainbuilder.utils import utils


import brainbuilder.utils.ants_nibabel as nib
import brainbuilder.segment as segment

from brainbuilder.interp.surfinterp import generate_surface_profiles
from brainbuilder.utils.mesh_utils import load_values, load_mesh_ext, visualization, pairwise_coord_distances, mesh_to_volume


def apply_batch_correction(
        chunk_info : pd.DataFrame,
        sect_info : pd.DataFrame,
        chunk_info_thickened_csv : str,
        surf_raw_values_dict: str,
        profiles_fn : str,
        surf_depth_mni_dict : dict,
        surf_depth_chunk_dict : dict,
        depth_list : list,
        resolution : float,
        struct_vol_rsl_fn : str,
        output_dir : str, 
        clobber : bool=False
        ) -> (pd.DataFrame, str) :
    '''
    apply batch correction to the profiles_fn to correct for mean shifts between chunks
    :param sect_info: dataframe containing section information
    :param surf_raw_values_dict: dictionary containing information about the surfaces
    :param surf_depth_mni_dict: dictionary containing information about the surfaces
    :param surf_depth_chunk_dict: dictionary containing information about the surfaces
    :param chunk_info: dataframe containing chunk information
    :param depth_list: list of depths to use for interpolation
    :param output_dir: path to output directory
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    '''
    os.makedirs(f'{output_dir}', exist_ok=True)

    mid_depth_index = int(np.ceil(len(depth_list)/2))	
    mid_depth = depth_list[mid_depth_index]

    # Get dict with mid surfaces for each chunk
    mid_depth_chunk_dict = dict(surf_depth_chunk_dict[mid_depth].items())
   
    # Get raw values file for mid depth surface
    values_fn =  surf_raw_values_dict[mid_depth]

    # Load raw values 
    values = load_values(values_fn).reshape(-1,)
    
    surface_labels_filename, sect_info, label_to_chunk_dict = create_surface_section_labels(
            chunk_info,
            sect_info, 
            values, 
            mid_depth_chunk_dict, 
            output_dir, 
            resolution,
            qc_surface_fn = surf_depth_mni_dict[mid_depth]['depth_rsl_fn'],
            ref_vol_fn = struct_vol_rsl_fn,
            clobber = clobber
            ) 

    profiles = np.load(profiles_fn+'.npz')['data']

    params, paired_values = calc_batch_correction_params(
            chunk_info,
            chunk_info_thickened_csv,
            surface_labels_filename,
            profiles, 
            surf_depth_mni_dict[mid_depth]['sphere_rsl_fn'], #stereo_sphere_filename
            surf_depth_mni_dict[mid_depth]['depth_rsl_fn'], #stereo_cortex_filename
            f'{output_dir}',
            label_start=0,
            label_offset=1,
            clobber=clobber 
            )

    coords = load_mesh_ext(surf_depth_mni_dict[mid_depth]['depth_rsl_fn'])[0]
    
    #for i, row in paired_values.iterrows() :
    #    c0 = coords[int(row['curr_idx'])]
    #    c1 = coords[int(row['next_idx'])]
    #    distance = np.sqrt(np.sum(np.power(c0-c1,2)))
    #    print(row['curr_values'], distance)

    sect_info = update_df_with_correction_params(sect_info, params, label_to_chunk_dict)



    return sect_info, paired_values


def create_surface_section_labels(
        chunk_info : pd.DataFrame,
        sect_info : pd.DataFrame,
        values: np.ndarray, 
        chunk_surface_dict : dict,
        out_dir : str,
        resolution:float,
        perc:float=0.25,
        qc_surface_fn : str='',
        ref_vol_fn : str='',
        clobber : bool=False
        ):

    acquisition = sect_info['acquisition'].values[0]
    out_fn = f'{out_dir}/{acquisition}_section_labels'
    qc_fn = f'{out_dir}/{acquisition}_section_labels_qc.png'
    vol_qc_fn = f'{out_dir}/{acquisition}_section_labels_qc.nii.gz'
    print(qc_fn)

    label_to_chunk_dict = {}

    sect_info['label'] = [0] * sect_info.shape[0] #- np.min(labels_for_df) +1

    labels = np.zeros(values.shape[0])

    if not os.path.exists(out_fn+'.npz') or clobber:
        
        width = utils.get_thicken_width(resolution/2)
        
        max_chunk = sect_info['chunk'].max() + 1

        for (chunk,), chunk_df in chunk_info.groupby(['chunk']) :

            surf_filename = chunk_surface_dict[chunk]
            coords = load_mesh_ext(surf_filename)[0]
            
            y_coords = coords[:,1]

            array_img = nibabel.load(chunk_df['nl_2d_vol_fn'].values[0])
            ystep = array_img.affine[1,1]
            ystart = array_img.affine[1,3]

            sect_chunk_df = sect_info[sect_info['chunk'] == chunk]
            
            y = sect_chunk_df['sample'].values

            sect_chunk_df['yw'] = (y * ystep + ystart).reshape(-1,1)
            
            # get the y limits for the chunk
            # |                         |   
            # |-------------------------|
            # ymin                     ymax 

            ymax = sect_chunk_df['yw'].max()
            ymin = sect_chunk_df['yw'].min()
            
            perc = 0.5
            # Posterior, caudal, bound for the chunk
            ylo = ymin + (ymax-ymin)*perc

            # Anterior, rostral, bound for the chunk
            yhi = ymax - (ymax-ymin)*perc

            #          ylo   yhi
            # |         |     |         |   
            # |-------------------------|
            # ymin                     ymax 

            # Set labels to 0 by default
            sect_chunk_df['label'] = [0] * sect_chunk_df.shape[0]

            # Define the label values
            label_0 = (max_chunk-chunk)*2 - 1
            label_1 = (max_chunk-chunk)*2

            # Identify sections that in the caudal portion of the chunk
            idx0 = (sect_chunk_df['yw'] <= ylo) & (sect_chunk_df['yw'] >= ymin)

            # Identify sections that in the rostral portion of the chunk
            idx1 = (sect_chunk_df['yw'] >= yhi) & (sect_chunk_df['yw'] <= ymax)

            #          ylo   yhi
            # |   idx0  |     |   idx1  |   
            # |-------------------------|
            # ymin                     ymax 

            sect_chunk_df['label'].iloc[ idx0 ] = label_0
            sect_chunk_df['label'].iloc[ idx1 ] = label_1

            label_to_chunk_dict[label_0] = chunk
            label_to_chunk_dict[label_1] = chunk

            vtx0 = (y_coords <= ylo) & (y_coords >= ymin)
            vtx1 = (y_coords >= yhi) & (y_coords <= ymax)

            #labels[vtx0] = label_0
            #labels[vtx1] = label_1

            labels[vtx0] = chunk
            labels[vtx1] = chunk
            '''
            for i, row in sect_chunk_df.iterrows() :
                if len(labels) == 0 :
                    labels = np.zeros(coords.shape[0])
                label = row['label'] 
                yw = row['yw']
                ymin = row['ylo']
                ymax = row['yhi']

                idx = ((y > ymin) & (y <= ymax) & (values > np.min(values) )).reshape(-1,)

                labels[idx] = label
            '''
            labels[values==0] = 0
            np.savez(out_fn, data=labels)
            
            print('Label png :', qc_fn)
            visualization(qc_surface_fn, labels, qc_fn)
            img = nibabel.load(ref_vol_fn)
            aff = img.affine
            starts = aff[0:3,3]
            steps = aff[[0,1,2],[0,1,2]]
            dimensions = img.shape
            
            print(qc_surface_fn)
            print(starts, steps)
            vol, n = mesh_to_volume(
                load_mesh_ext(qc_surface_fn)[0],
                labels,
                dimensions,
                starts,
                steps,
            )
            vol[n>0] = vol[n>0] / n[n>0]
            nib.Nifti1Image(vol,aff,direction_order='lpi').to_filename(vol_qc_fn)

    return out_fn, sect_info, label_to_chunk_dict


def update_df_with_correction_params(
        df: pd.DataFrame,
        params: pd.DataFrame,
        label_to_chunk_dict : dict 
        ) -> pd.DataFrame :
    '''
    Update the dataframe with the correction parameters
    :param df: dataframe containing section information
    :param params: dataframe containing the correction parameters
    :param label_to_chunk_dict: dictionary containing the mapping between labels and chunks
    :return: dataframe with updated correction parameters
    '''

    print('Update df with correction parameters') 
    df['batch_offset'] = [0]*df.shape[0]
    df['batch_multiplier']=[1]*df.shape[0]

    for i, row in params.iterrows() :

        #idx = df['chunk'] == label_to_chunk_dict[row['label'].astype(int)]
        idx = df['chunk'] == row['label'] #label_to_chunk_dict[row['label'].astype(int)]

        df['batch_offset'].loc[ idx ] = row['offset']
        df['batch_multiplier'].loc[ idx ] = row['multiplier']
        print(row['label'], row['offset'])

    return df 

def get_border_y(df, chunk, thr, start, step, caudal = True):
    dfs = df.sort_values(['chunk_order']) 
    ymax = dfs['chunk_order'].max()
    
    if caudal:
        dfs = dfs.iloc[0:thr]
        y_list = (dfs['chunk_order'].min(), dfs['chunk_order'].max()+1) ##add +1 so that it is inclusive range
    else : #rostral
        dfs = dfs.iloc[ -thr: ] 
        y_list = (dfs['chunk_order'].min(), dfs['chunk_order'].max()+1 )

    y_list = sorted([y*step+start for y in y_list])
    #need to convert y to world coordinates

    return y_list

global CAUDAL_LABEL 
global ROSTRAL_LABEL

CAUDAL_LABEL = 0.33
ROSTRAL_LABEL= 0.66

import numpy as np




 

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


def get_rostral_caudal_borders(label_surf_values, surfaces, out_dir,  perc=0.05, clobber=False):
    border_fn = f'{out_dir}/borders' 

    #WARNING the label volumes and surfaces must be in same coord space
    if not os.path.exists(border_fn) or clobber :

        labels = load_values(label_surf_values)
        chunks = np.unique(labels[labels>0])
        n_chunks=len(chunks)
        
        assert type(surfaces) == list, 'Error: surfaces is not a list'
        if len(surfaces) == 1:
            surfaces = surfaces * n_chunks

        n_vtx = load_mesh_ext(surfaces[0])[0].shape[0]

        border=np.zeros(n_vtx)
        
        for i, chunk in enumerate(chunks) :
            
            coords = load_mesh_ext(surfaces[i])
            y=coords[:,1]
            
            ymin=np.min(y)
            ymax=np.max(y)

            y0 = np.min(y[labels==chunk])
            y1 = np.max(y[labels==chunk])
            y = coords[:,1]

            perc=0.5

            c0 = y0
            c1 = y0 + np.abs(y0-y1)*perc

            r0 = y1 - np.abs(y1-y0)*perc
            r1 = y1

            caudal_idx = (y > c0 ) & ( y < c1)
            assert np.sum(caudal_idx)>0, f'Error: no vertices found between {c0} and {c1}'

            rostral_idx = (y > r0 ) & ( y < r1 ) 
            assert np.sum(rostral_idx)>0, f'Error: no rostral vertices found between {r0} and {r1}'

            border[caudal_idx] = chunk + CAUDAL_LABEL
            border[rostral_idx] = chunk + ROSTRAL_LABEL

        
        np.savez(border_fn, data=border)
    return border_fn


def create_dataframe_for_pairs(
        dist,
        avg_values,
        next_range, 
        c_vtx
        ):

    arg_m = np.argmin(dist, axis=1) 
    min_dist = dist[range(dist.shape[0]),arg_m]
    del dist

    n_vtx = next_range[arg_m]
   
    next_values = avg_values[n_vtx]
    curr_values = avg_values[c_vtx]

    paired_values = pd.DataFrame({ 
                            'curr_idx': c_vtx,
                            'next_idx': n_vtx,
                            'curr_values': curr_values,
                            'next_values': next_values,
                            'distance': min_dist
                        })

    thresholds = [5, 95]
    curr_perc_min, next_perc_max = np.percentile(curr_values, thresholds)
    next_perc_min, curr_perc_max = np.percentile(next_values, thresholds)
    #curr_perc_min, curr_perc_max = np.max(curr_values)*0.9, np.max(curr_values)*0.1
    #next_perc_min, next_perc_max = np.max(next_values)*0.9, np.max(next_values)*0.1
    
    idx0 = (curr_values > curr_perc_min) & (curr_values < curr_perc_max)
    idx1 = (next_values > next_perc_min) & (next_values < next_perc_max)

    idx = idx0 & idx1
    print('Sum of valid pairs', np.sum(idx))
    
    temp_paired_values = paired_values[ idx ] 

    if temp_paired_values.shape[0] > 0 :
        paired_values = temp_paired_values

    return paired_values


def find_pairs_between_labels(curr_label, next_label, stereo_sphere_coords, labels, avg_values ):
    # find vertex points in curr and next label
    curr_idx = labels == float(curr_label) 
    next_idx = labels == float(next_label)

    # get range of vertex points in curr and next label
    vtx_range = np.arange(stereo_sphere_coords.shape[0]).astype(int)
    
    # get range of vertex points in curr and next label
    curr_range = vtx_range[curr_idx]
    next_range = vtx_range[next_idx]

    # get coordinates of vertex points in curr and next label
    next_coords = stereo_sphere_coords[next_idx]
    curr_coords = stereo_sphere_coords[curr_idx]

    dist = pairwise_coord_distances(curr_coords, next_coords) 
    
    paired_values = create_dataframe_for_pairs(dist, avg_values, next_range, curr_range)
    
    paired_values['curr_label'] = [curr_label] * paired_values.shape[0]
    paired_values['next_label'] = [next_label] * paired_values.shape[0]
    paired_values['vtx_pair_id'] = np.arange(paired_values.shape[0]) 

    return paired_values

def simple_get_paired_values(
        chunk_info,
        volumes_df,
        paired_values_csv:str, 
        unique_paired_values_csv:str,  
        clobber:bool = False):

    if not os.path.exists(paired_values_csv) or clobber:
        # find corresponding points between caudal and rostral 
        paired_values=pd.DataFrame({})
        for (sub, hemisphere, chunk), chunk_df in chunk_info.groupby(['sub','hemisphere', 'chunk']) :
            rec_fn = volumes_df['thickened'].loc[ volumes_df['chunk']  == chunk].values[0]
            rec_cls_fn = chunk_df['nl_2d_vol_cls_fn'].values[0]
            gm_fn = chunk_df['ref_space_nat'].values[0]

            rec = nib.load(rec_fn).get_fdata()
            rec_cls = nib.load(rec_cls_fn).get_fdata()
            gm = nib.load(gm_fn).get_fdata()

            seg = segment.multi_threshold(rec)

            curr_values = np.median(rec[ (gm>0.9) & (rec_cls>rec_cls.max()*0.5) & (seg>seg.max()*0.5) ])

            row = pd.DataFrame({    'sub': sub, 
                                    'hemisphere': hemisphere,
                                    'curr_label': chunk,
                                    'curr_values': curr_values,
                                }, index=[0])
            paired_values = pd.concat([paired_values, row])
        paired_values['next_label'] = paired_values['curr_label'].shift(-1)
        paired_values['next_values'] = paired_values['curr_values'].shift(-1)
        #paired_values.dropna(inplace=True)

        paired_values.to_csv(paired_values_csv, index=False)
        paired_values.to_csv(unique_paired_values_csv, index=False)
    paired_values = pd.read_csv(paired_values_csv, index_col=False)

    return paired_values

def get_paired_values(
        paired_values_csv:str, 
        unique_paired_values_csv:str,  
        stereo_sphere_filename:str, 
        label_filename:str, 
        values:np.ndarray, 
        out_dir:str, 
        label_start:int = 1, 
        label_offset:int = 2, 
        clobber:bool = False):


    if not os.path.exists(paired_values_csv) or clobber:
        # find corresponding points between caudal and rostral 
        depth_values=[] 
       
        labels = load_values(label_filename)
        assert np.sum(labels>0), 'Error: no labels in '+label_filename

        stereo_sphere_coords = load_mesh_ext(stereo_sphere_filename)[0]
        chunks = np.unique(labels.astype(int))[1:]
        max_chunk = np.max(chunks)
        n_chunks = len(chunks)

        # Unlabel (i.e. set to 0) labeled vertices that have a value of 0
        labels[ values==0 ] = 0
        assert np.sum( values[labels>0] == 0) == 0, 'Error: 0 values found in vertices that shouldnt be zero, ie where labels >0 '

        # examples of chunk structure
        #   | 1 |   | 2 |   | 3 |
        #   c   r   c   r   c   r
        #   1   2   3   4   5   6

        label_list = np.unique(labels)[1:]
        paired_values = pd.DataFrame({})
        curr_labels = label_list[label_start:-1:label_offset]
        next_labels = label_list[label_start+1::label_offset]
        for curr_label, next_label in zip(curr_labels, next_labels) :
            print(f'Finding pairs between {curr_label} and {next_label}')

            curr_paired_values = find_pairs_between_labels(curr_label, next_label, stereo_sphere_coords, labels, values)
            paired_values = pd.concat([paired_values, curr_paired_values])

        paired_values.to_csv(paired_values_csv, index=False)

    paired_values = pd.read_csv(paired_values_csv, index_col=False)
    #FIXME testing if unqiue values are actually needed 
    #if not os.path.exists(unique_paired_values_csv) or clobber :
    #    unique_paired_values = get_unique_pairs(paired_values, 'curr')
    #    unique_paired_values = get_unique_pairs(unique_paired_values, 'next')
    #    unique_paired_values.to_csv(unique_paired_values_csv, index=False)
    #unique_paired_values = pd.read_csv(unique_paired_values_csv, index_col=False)
    unique_paired_values = paired_values
    unique_paired_values.to_csv(unique_paired_values_csv, index=False)


    png_fn = f'{out_dir}/paired_values.png'

    plt.cla();  plt.clf();
    plt.figure(figsize=(10,10))
    if not os.path.exists(png_fn) or clobber :
        for i, row in unique_paired_values.iterrows() :

            continue
            x = [ row['curr_label'], row['next_label'] ] 
            y = [ row['curr_values'], row['next_values'] ]
            plt.scatter(x,y, c='r')
            plt.plot(x,y, c='b', alpha=0.1)
        plt.savefig(png_fn)
    print('Done')
    return unique_paired_values 


def get_unique_pairs(paired_values, direction, n_points=1) :
    unique_paired_values=pd.DataFrame({})

    for (i_chunk,), i_df in paired_values.groupby([f'{direction}_label']) :
    
        for counter, ((j_idx,), j_df) in enumerate(i_df.groupby([f'{direction}_idx'])) :
            min_distance_idx = np.argsort(j_df['distance'])[:n_points]
            row = j_df.iloc[min_distance_idx]
                
            unique_paired_values = pd.concat([ unique_paired_values, row])

    return unique_paired_values



def draw_pair_plots(
        surf_coords:np.ndarray,
        paired_values:type(pd.DataFrame), 
        out_fn:str, 
        hue_string_0:str='vtx_pair_id', 
        hue_string_1:str='vtx_pair_id'
        ) -> None:

    d_list = []
    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
    fig, axes = plt.subplots(2,3, figsize=(40,10))

    for ax in axes.ravel() :
        ax.grid(False)
        for i, ((curr_chunk,), tdf) in enumerate(paired_values.groupby(['curr_label'])): 
            ax = axes.ravel()[i]


            cidx = tdf['curr_idx'].values.astype(int)
            ridx = tdf['next_idx'].values.astype(int)

            xc = (surf_coords[cidx, 0] - start[0]) / step[0]
            yc = (surf_coords[cidx, 1] - start[1]) / step[1]
            zc = (surf_coords[cidx, 2] - start[2]) / step[2]

            xr = (surf_coords[ridx, 0] - start[0]) / step[0]
            yr = (surf_coords[ridx, 1] - start[1]) / step[1]
            zr = (surf_coords[ridx, 2] - start[2]) / step[2] 
            
            o = 1.5*(np.max(xc) - np.min(xc))
            ax.set_title(f'Current Slab {curr_chunk}')
            sns.scatterplot(x=xc, y=zc, hue=tdf[hue_string_0], palette='nipy_spectral',alpha=0.3,ax=ax)
            sns.scatterplot(x=o+xr, y=zr, hue=tdf[hue_string_1], palette='nipy_spectral', alpha=0.3,ax=ax)
            sns.despine(left=True, bottom=True)
            ax.grid(False)
            ax.get_legend().remove()

    print('\tWriting', out_fn)
    plt.savefig(out_fn)
    plt.cla()
    plt.clf()


def qc_paired_values(out_dir,paired_values, surf_coords, clobber=False):


    idx_pairs_fn = f'{out_dir}/validate_paired_idx.png'
    values_pairs_fn = f'{out_dir}/validate_paired_values.png'
    out_vol_fn = f'{out_dir}/validated_paired_distances.nii.gz'
    #if  not os.path.exists(idx_pairs_fn) or clobber :
        #draw_pair_plots(surf_coords, paired_values, idx_pairs_fn)
    #if  not os.path.exists(values_pairs_fn) or clobber :
    #    draw_pair_plots(surf_coords, paired_values, values_pairs_fn, hue_string_0='curr_values', hue_string_1='next_values')


def simple_chunk_correction(paired_values):

    params = pd.DataFrame({})

    row_dict = {    'label': paired_values['curr_label'].min(),
                    'multiplier': [1],
                    'offset': [0] }

    params = pd.concat([params, pd.DataFrame(row_dict)] )

    total_offset = 0


    for (curr_label,), df in paired_values.groupby(['curr_label']):
        n=df.shape[0]
        next_label = df['next_label'].values[0]
        curr_values = df['curr_values'].values
        next_values = df['next_values'].values

        #dist = df['distance'].values
        #inv_dist = 1/dist
        #wgts = inv_dist/np.sum(inv_dist)
        
        #diff2 = curr_values - next_values

        #mean_diff = np.mean( diff2 )
        #mean_diff = np.sum( diff2 * wgts )
        mean_diff = curr_values - next_values

        error = np.sum(np.abs(curr_values-(next_values+mean_diff)))/curr_values.shape[0]
        offset = mean_diff
        total_offset += offset
        print(curr_values, next_values, offset, total_offset)

        row_dict = {    'label': next_label,
                        'multiplier': 1,
                        'offset': total_offset,
                        'chunk_offset': offset,
                    }

        params = pd.concat([params, pd.DataFrame(row_dict, index=[0]) ] )

    return params

        
def calc_batch_correction_params(
        chunk_info : pd.DataFrame,
        chunk_info_thickened_csv : str,
        label_filename:str,
        values:np.ndarray, 
        stereo_sphere_filename:str,
        stereo_cortex_filename:str,
        out_dir:str,
        label_start:int=1,
        label_offset:int=2,
        clobber=False
        ):
    '''
        Identify the vertices within a given chunk and find the vertices that are part of the anterior and posterior border of that chunk.

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

    volumes_df = pd.read_csv(chunk_info_thickened_csv)
    
    params_fn = f'{out_dir}/params.csv'

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(params_fn) or clobber :

        nvtx = load_mesh_ext(stereo_sphere_filename)[0].shape[0]
        vtx_range = np.arange(nvtx).astype(int)
        labels = np.unique( load_values(label_filename))[1:]
        print('\t\tNumber of labels', len(labels), labels)
        n_labels = len(labels)

        print('\t\tGet Paired Values')
        '''
        paired_values = get_paired_values(
                paired_values_csv, 
                unique_paired_values_csv,  
                stereo_sphere_filename, 
                label_filename, 
                values, 
                out_dir,
                label_start = label_start,
                label_offset = label_offset,
                clobber=clobber
                )
        '''

        paired_values = simple_get_paired_values(
                chunk_info,
                volumes_df,
                paired_values_csv,
                unique_paired_values_csv,
                )

        if not os.path.exists(params_fn) or clobber :
            params = simple_chunk_correction(paired_values)
            params.to_csv(params_fn,index=False)

        params = pd.read_csv(params_fn)

        qc_paired_values(out_dir, paired_values, load_mesh_ext(stereo_cortex_filename)[0], clobber=clobber)

    params = pd.read_csv(params_fn)
    paired_values = pd.read_csv(unique_paired_values_csv)

    return params, paired_values
