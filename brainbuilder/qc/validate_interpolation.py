import utils.ants_nibabel as nib
#import nibabel as nib
import seaborn as sns
import numpy as np
import stripy
import pandas as pd
import vast.surface_tools as surface_tools
import argparse
import matplotlib.pyplot as plt
import h5py
import os
from utils.mesh_utils import load_mesh_ext
#from matplotlib_surface_plotting import plot_surf
from glob import glob
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import norm

atlas_dict = {


        }

import re

def plot_r2(df, out_fn):

    df.dropna(inplace=True)
    plt.figure(figsize=(12,12)) 
    nligands = len(np.unique(df['Ligand']))
    col_wrap=4 if nligands > 4 else nligands
    sns.set(rc={'figure.figsize':(12,12)})
    df['True Density'] = df['TrueDensity']
    g = sns.lmplot(y='Estimated', x='True Density', data=df, hue='Ligand', col='Ligand', scatter_kws={'alpha':0.15}, col_wrap=col_wrap, sharex=False, sharey=False, fit_reg=True, legend=True)
        
    def annotate(data, **kws):
        import scipy as sp
        r, p = sp.stats.pearsonr(data['True Density'], data['Estimated'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p<0.001'.format(r, p), transform=ax.transAxes)#{:.2g}
        
    g.map_dataframe(annotate)

    print('Writing',out_fn)
    plt.savefig(out_fn)

def plot_validation(filename='validation.csv', out_fn='validation.png', area_dict=None):
    df = pd.read_csv(filename)
    df_factor = df.copy()

    df_factor.rename(columns = {'Error':'Error (%)', 'Distance':'Distance of interpolated vertices (mm)', 'TotalArea':'N. Vertices', 'StdDev':'Receptor Density (Std. Dev.)', 'TrueDensity':'Receptor Density (Mean)'}, inplace = True)
    factors_out_fn = re.sub('.png','_factors.png',out_fn)
    regr_out_fn = re.sub('.png','_r2.png',out_fn)
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    #g = sns.PairGrid(df_factor, y_vars=["Error (%)"], x_vars=['Distance (mm)'],  hue='Ligand', height=10)
    #g.map(sns.regplot)
    sns.scatterplot(y ="Error (%)", x ='Distance of interpolated vertices (mm)', data=df_factor)

    plt.subplot(2,2,2)
    #g = sns.PairGrid(df_factor, y_vars=["Error (%)"], x_vars=['N. Vertices'],  hue='Ligand', height=10)
    #g.map(sns.regplot)
    sns.scatterplot(y ="Error (%)", x ='N. Vertices', data=df_factor)

    plt.subplot(2,2,3)
    sns.scatterplot(y ="Error (%)", x ='Receptor Density (Std. Dev.)', data=df_factor)
    #g = sns.PairGrid(df_factor, y_vars=["Error (%)"], x_vars=['Std. Dev.'],  hue='Ligand', height=10)
    #g.map(sns.regplot)

    plt.subplot(2,2,4)
    sns.scatterplot(y ="Error (%)", x ='Receptor Density (Mean)', data=df_factor)
    #g = sns.PairGrid(df_factor, y_vars=["Error (%)"], x_vars=['True Density (fmol/mg)'],  hue='Ligand', height=10)
    #g.map(sns.regplot)
    plt.savefig(factors_out_fn)


    df['Error'].loc[df['Error'] > 100 ] = 100
    plt.figure(figsize=(9,9))
    plt.title('Histogram of interpolation error when estimating receptor densities')
    plt.hist(df['Error'].values, bins=100)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylabel('Count')
    plt.xlabel('Interpolation Error % (|Predicted-True Density|/True Density)')
    #plt.minorticks_on()
    xticks = np.arange(0,105,5).astype(int)
    xticklabels = xticks.astype(str)
    xticklabels[-1] = xticklabels[-1] + '+'
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xticklabels)

    n5 = np.sum( df['Error'] <= 5 )
    n10 = np.sum( df['Error'] <= 10 )

    print('Percent below 5% error:', np.round(100. * n5 / df.shape[0],2))
    print('Percent below 10% error:', np.round(100. * n10 / df.shape[0],2) )
    
    plt.savefig(out_fn)

    plt.figure(figsize=(12,12)) 
    sns.set(rc={'figure.figsize':(18,18)})
    sns.set(font_scale=2)
    sns.lmplot(x='Estimated', y='TrueDensity', data=df, hue='Ligand', col='Ligand', col_wrap=4)

    plt.savefig(out_regr_fn)


def area(a, b, c) :
    return 0.5 * norm( np.cross( b-a, c-a ) )


def read_gifti(mesh_fn):
    mesh = nib.load(mesh_fn)

    coords = mesh.agg_data('NIFTI_INTENT_POINTSET')
    faces =  mesh.agg_data('NIFTI_INTENT_TRIANGLE')
    return coords, faces.astype(int)

def get_ngh(coords, faces):
    ngh={}

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    for counter, (i0, i1, i2 ) in enumerate(faces) :
        def f(ngh, i,j,k):
            try :
                ngh[i] += [j,k]
            except KeyError :
                ngh[i] = [j,k]
            return ngh
        ngh = f(ngh, i0, i1, i2)
        ngh = f(ngh, i1, i0, i2)
        ngh = f(ngh, i2, i0, i1)
        #ax=plt.subplot()
        #ax.set_color_cycle(['c', 'm', 'y', 'k'])
    for key, value in ngh.items() :
        ngh[key] = np.unique(value)
        
        #if len(ngh[key]) <= 3 :
            #print(len(ngh[key]))
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(coords[key,0],coords[key,1],coords[key,2], c='g')
            #ax.scatter(coords[ngh[key]][:,0],coords[ngh[key]][:,1],coords[ngh[key]][:,2], c='r')
            #plt.show()    
    return ngh

def calc_dist(i, coords,ngh) :
    distList=[]
    for ngh_i in ngh[i] :
        if ngh_i != i :
            temp_dist = np.sqrt(np.sum(np.power(coords[i] - coords[ngh_i],2)))
            distList.append(temp_dist)
    distListMean = np.mean(distList)
    return distListMean 

def get_border(border,densities, ngh, ngh_inner_all):
    new_border=[]
    for j in border :
        if densities[j] > 0 :
            new_border += [ jj for jj in ngh[j] if not jj in ngh_inner_all and densities[jj] > 0 ]
    return new_border

def get_core_vertices(ngh, densities, max_depth, core_depth, i) :
    print('get core vertices')
    ngh_inner_core = list([i])
    ngh_inner_all = list([i])
    border = list(ngh[i])

    for depth in range(max_depth):
        new_ngh = []
        new_border = get_border(border,densities, ngh, ngh_inner_all)

        if depth < core_depth : ngh_inner_core += border

        if depth < max_depth : ngh_inner_all += border 
        
        border = new_border

        print('\t\t',depth,core_depth,max_depth,len(ngh_inner_core), len(ngh_inner_all),len(border))
        #break_early = True if len(border) == 0 else False
        #if break_early: 
        #    print('Break Early!')
        #    max_depth=core_depth+1
        #    break
    border = get_border(border,densities, ngh, ngh_inner_all)
    return np.array(ngh_inner_core), np.array(ngh_inner_all), np.array(border)

def calculate_neighbours(i, sphere_coords, cortex_coords, core_depth, int_depth, max_depth):
    idx_range = np.arange(sphere_coords.shape[0]).astype(int)

    curr_vtx = sphere_coords[i]
    d = np.sqrt(np.sum(np.power(sphere_coords[i] - sphere_coords,2), axis=1))
    core_ngh_idx = d <= core_depth
    all_ngh_idx = d <= int_depth
    border_idx = (d <= max_depth) & (d > int_depth)

    ngh_inner_core = idx_range[ core_ngh_idx ]
    ngh_inner_all = idx_range[ all_ngh_idx ]
    border = idx_range[ border_idx ]

    perimeter_dist = np.mean(np.sqrt(np.sum(np.power(cortex_coords[i] - cortex_coords[border], 2), axis=1)))

    assert len(set(ngh_inner_core) & set(border)) == 0 , 'Error: border and core vertices overlap'
    return ngh_inner_core, ngh_inner_all, border,  perimeter_dist
    
    

def calculate_neighbours_old(i, ngh, coords, cortex_coords, densities, max_depth, core_depth=2) :
    # ngh_inner_core are the vertices to be estimated
    # ngh_inner_all are all the vertices within the outer border
    # border are the vertices with known receptor densities and which are used to estimate the densities at ngh_inner_core
    ngh_inner_core, ngh_inner_all, border = get_core_vertices(ngh, densities, max_depth, core_depth, i)
    border = list(np.unique(border))
    ngh_inner_core = list(np.unique(ngh_inner_core))
    ngh_inner_all = list(np.unique(ngh_inner_all))

    # penumbra vertices are those that are treated as unknown but which will not be estimated
    penumbra = [ i for i in ngh_inner_all if not i in ngh_inner_core  ]

    # sanity check
    if len(penumbra) == 0 and not max_depth-1 == core_depth :
        print('Error in core/penumbra/border', max_depth, core_depth)
        print(ngh_inner_core)
        print(ngh_inner_all)
        return [], [], [], []

    #calculate the distance between vertices in penumbra and their neighbours
    avg_border_dist_list = [ calc_dist(i,cortex_coords,ngh) for i in penumbra ]
    # take mean of distances between penumbra and their neihgbours. this basically gives an
    # approximate average edge length in the penumbra
    avg_border_dist = np.mean([ i for i in avg_border_dist_list if not np.isnan(i)])
    print('\taverage border distance:', avg_border_dist)
    
    if np.isnan(avg_border_dist) :
        print('\t avg core depth', core_depth, 'max_depth', max_depth )
        print(penumbra)
        print()

    #calculate the distance between the core and the border by multiplying the number of edges
    # between these two by the average edge length
    perimeter_dist = (max_depth - core_depth) * avg_border_dist if not np.isnan(avg_border_dist) else 0
    
    print('max depth:',max_depth,'perimeter dist',perimeter_dist,'core size:',len(ngh_inner_core))
    
    return np.array(ngh_inner_core), np.array(ngh_inner_all), np.array(border),  perimeter_dist

def interpolate_over_sphere(densities, coords_all,  idx_inner_core, idx, border, i):
    spherical_coords = surface_tools.spherical_np(coords_all)

    spherical_coords += np.random.normal(0,.000001,spherical_coords.shape) 

    coords_src = spherical_coords[border]
    coords = spherical_coords[np.concatenate([idx,border])]

    lats, lons = coords[:,1]-np.pi/2, coords[:,2]
    lats_src, lons_src = coords_src[:,1]-np.pi/2, coords_src[:,2]

    mesh = stripy.sTriangulation(lons_src, lats_src, permute=True)

    # interpolate over the sphere
    try :
        interp_val, interp_type = mesh.interpolate(lons,lats, zdata=densities[border], order=1)
    except ValueError :
        print('Error in stripy interpolation')
        return np.nan
    
    #error = np.abs(np.mean(interp_val[0:len(idx)]) - np.mean(densities[idx][0:len(idx)])) /  np.mean(densities[idx][0:len(idx)])
    estimated = np.mean(interp_val[0:len(idx_inner_core)])
    ground_truth = np.mean(densities[idx][0:len(idx_inner_core)])
    error = 100. * np.abs(estimated - ground_truth ) /  ground_truth
    #if error > 1 :
        #print( len(interp_val[0:len(idx)]), len(densities[idx][0:len(idx)]))
        #plt.scatter(interp_val[0:len(idx)],densities[idx][0:len(idx)])
        #plt.show()
    #error = np.abs(np.mean(interp_val[0:len(idx_inner_core)]) - np.mean(densities[idx][0][0:len(idx)])) /  np.mean(densities[idx][0][0:len(idx)])

    '''
    plt.subplot(2,1,1)
    plt.title('observed')
    plt.scatter(lons[0:len(idx)], lats[0:len(idx)],s=400,cmap='hot', c=densities[idx])
    plt.scatter(lons[len(idx):len(idx+border)], lats[len(idx):len(idx+border)],s=500,cmap='hot',marker='p', c=densities[border])
    plt.subplot(2,1,2)
    plt.title('interpolated')
    plt.scatter(lons[0:len(idx)], lats[0:len(idx)],s=400,cmap='hot', c=interp_val[0:len(idx)])
    plt.scatter(lons[len(idx):len(idx+border)], lats[len(idx):len(idx+border)],s=500,cmap='hot',marker='p', c=densities[border])

    plt.show()
    '''
    return  error, estimated, ground_truth

def iterate_over_vertices(valid_idx_range, idx,  coords, cortex_coords, faces, densities, ligand, resolution, max_depth=5, n_samples=10000):
    resolution=float(resolution)
    densities = densities.reshape(-1,)
    min_densities = 0.01 * np.mean(densities[densities > 0]) 
    print('Min Densities:', min_densities)

    output_df_list = []
    counter = 0 
    for i, (x,y,z) in zip(valid_idx_range, coords[valid_idx_range,:]): 
        core_depth = np.random.uniform(resolution/2,resolution) 
        int_depth = np.random.uniform(core_depth,resolution*2)
        max_depth = np.random.uniform(int_depth,resolution*3)
        if i % 10 == 0 : print(100*i/idx.shape[0],end='\r')
        # calculate neighbours
        #core, inner_all, border, perimeter_dist = calculate_neighbours(i, coords, cortex_coords, densities, depth, core_depth)
        core, inner_all, border, perimeter_dist = calculate_neighbours(i, coords, cortex_coords, core_depth, int_depth, max_depth)
    
        if True in [ len(idx_list) < 4 for idx_list in [core, inner_all, border] ] : continue

        core = core[densities[core] > min_densities] 
        inner_all = inner_all[densities[inner_all]> min_densities] 

        n = len(border)
        border = border[densities[border]> min_densities] 

        if len(core) < 1 or len(border) < 4 : continue 
        else : counter += 1
        
        # interpolate small patch
        error, estimated, ground_truth = interpolate_over_sphere(densities, coords,   core, inner_all, border, i)
        sd = np.std( densities[ np.concatenate([inner_all, border]) ].astype(np.float128) )
        if np.isinf(sd) :
            print(sd)
            print(densities[ np.concatenate([inner_all, border])] )
            exit(1)

        #print('Error:', np.round(error,2), 'Est', estimated,'True', ground_truth, 'sd', sd)
        
        total_area=len(core) #np.sum([area(cortex_coords[ngh[i][0]], cortex_coords[ngh[i][1]],cortex_coords[ngh[i][2]]) for i in inner_all ])
        
        tdf= pd.DataFrame({'Ligand':[ligand],
                                            'Distance':[perimeter_dist],
                                            'TotalArea':[total_area],
                                            'Error':[error],
                                            'n':[n],
                                            'StdDev':[sd],
                                            'Estimated':[estimated],
                                            'TrueDensity':[ground_truth] } ) 
        print(tdf)
        output_df_list.append(tdf)

        if counter >= n_samples : break
    output_df = pd.concat(output_df_list)

    return output_df

def validate_interpolation(ligand_densities_fn, sphere_mesh_fn, cortex_mesh_fn,  output_dir, resolution, ligand='flum', n_samples=10000, max_depth=5,  clobber=False):
    
    output_filename = f'{output_dir}/validate_interpolation_{ligand}.csv'
    output_image = f'{output_dir}/validate_interpolation_{ligand}.png'
    if not os.path.exists(output_filename) or clobber : 
        #load ligand densities
        densities = pd.read_csv(ligand_densities_fn,header=None).values.astype(np.float16)

        idx = densities > 0.0
        idx = idx.reshape(-1,)
        
        idx_range = np.arange(idx.shape[0]).astype(int)
        idx_range = idx_range[idx]
        
        np.random.shuffle(idx_range)
        
        valid_idx_range = idx_range #[0:n_samples]

        #load coords and neighbours
        print(sphere_mesh_fn)
        print(cortex_mesh_fn)
        print(ligand_densities_fn)
        coords, faces = load_mesh_ext(sphere_mesh_fn)
       
        cortex_coords, faces = load_mesh_ext(cortex_mesh_fn)
        assert densities.shape[0] == coords.shape[0], 'Error: densities does not equal coords' 
        output_df = iterate_over_vertices(valid_idx_range, idx,  coords, cortex_coords, faces, densities, ligand, resolution, n_samples=n_samples, max_depth=max_depth)

        output_df.to_csv(output_filename)
    
    #if not os.path.exists(output_image) or clobber :
    #    plot_validation(output_filename, output_image)
    output_df = pd.read_csv(output_filename)

    return output_df

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--out-dir', dest='out_dir', default='/scratch/tfunck/output2/5_surf_interp/', help='output directory')
    parser.add_argument('--mesh', dest='mesh_fn', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='mesh fn')
    parser.add_argument('--ligand', dest='file_list', nargs='+', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='csv ligand densities')
    args = parser.parse_args()
    out_dir = args.out_dir
    sphere_mesh_fn = 'surf_0.75mm_0.0_inflate_rsl.surf.gii'
    cortex_mesh_fn = 'surf_0.75mm_0.0_rsl.surf.gii'

    file_list = args.file_list

    out_fn='validation.csv'
    output_file = open(out_fn, 'w+')
    output_file.write('ligand,perimeter_dist,area,error,n\n')
    
    max_depth = 7
    file_list = glob('MR1_R_*_0.25mm_profiles_0.0_raw.csv')
    n_samples=1000   
    for filename in file_list :
        ligand = filename.split('_')[2]
        print(filename, ligand)
        #output_file = validate_interpolation(filename, sphere_mesh_fn, cortex_mesh_fn, output_file, ligand, n_samples=n_samples,max_depth=max_depth)
        break
    output_file.close()
    plot_validation(out_fn)




