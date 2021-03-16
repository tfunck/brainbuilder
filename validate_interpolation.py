import nibabel as nib
import numpy as np
import stripy
import pandas as pd
import vast.surface_tools as surface_tools
import argparse
import matplotlib.pyplot as plt
from plot_validation import plot_validation
from matplotlib_surface_plotting import plot_surf
from glob import glob

from numpy.linalg import norm

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

def calc_dist(i, ngh_inner_core,coords,ngh) :
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
            new_border += [ jj for jj in ngh[j] if not jj in ngh_inner_all ]
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
    return ngh_inner_core, ngh_inner_all, border

def calculate_neighbours(i, ngh, coords, cortex_coords, densities, max_depth, core_depth=2) :
    ngh_inner_core, ngh_inner_all, border = get_core_vertices(ngh, densities, max_depth, core_depth, i)
    border = list(np.unique(border))
    ngh_inner_core = list(np.unique(ngh_inner_core))
    ngh_inner_all = list(np.unique(ngh_inner_all))

    penumbra = [ i for i in ngh_inner_all if not i in ngh_inner_core  ]
    if len(penumbra) == 0 and not max_depth-1 == core_depth :
        print('Error in core/penumbra/border', max_depth, core_depth)
        print(ngh_inner_core)
        print(ngh_inner_all)
        return [], [], [], []

    avg_border_dist_list = [ calc_dist(i,ngh_inner_core,cortex_coords,ngh) for i in penumbra ]
    avg_border_dist = np.mean([ i for i in avg_border_dist_list if not np.isnan(i)])
    
    if np.isnan(avg_border_dist) :
        print('\t avg core depth', core_depth, 'max_depth', max_depth )
        print(penumbra)
        print()
    perimeter_dist = (max_depth - core_depth) * avg_border_dist if not np.isnan(avg_border_dist) else 0
    
    
    print('max depth:',max_depth,'perimeter dist',perimeter_dist,'core size:',len(ngh_inner_core))
    
    return ngh_inner_core, ngh_inner_all, border,  perimeter_dist

def interpolate_over_sphere(densities, coords_all, faces, idx_inner_core, idx, border, i):
    spherical_coords = surface_tools.spherical_np(coords_all)

    spherical_coords += np.random.normal(0,.000001,spherical_coords.shape) 

    coords_src = spherical_coords[border]
    coords = spherical_coords[idx+border]

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
    error = np.abs(np.mean(interp_val[0:len(idx_inner_core)]) - np.mean(densities[idx][0:len(idx_inner_core)])) /  np.mean(densities[idx][0:len(idx_inner_core)])
    if error > 1 :
        print( len(interp_val[0:len(idx)]), len(densities[idx][0:len(idx)]))
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
    return  error, interp_val[0:len(idx)]

def iterate_over_vertices(idx_range, idx, ngh, coords, cortex_coords, faces, densities, output_file,ligand,max_depth=5, n_samples=10000):
    for i, (x,y,z) in zip(idx_range, coords[idx,:]): 
        depth = np.random.randint(2,max_depth+1)
        core_depth =  np.random.randint(1,depth)
        if i % 10 == 0 : print(100*i/idx.shape[0],end='\r')
        #print(f'U(1,{max_depth})->{depth}, {i}',end='\r')
        # calculate neighbours
        core, inner_all, border, perimeter_dist = calculate_neighbours(i, ngh, coords, cortex_coords, densities, depth, core_depth)
        
        n = len(border)
        if n < 4 : 
            print('\tSkipping')
            continue
        # interpolate small patch
        error, interp = interpolate_over_sphere(densities, coords,  ngh, core, inner_all, border, i)
        #error = 5 if error > 5 else error
        #if error > 0.5 : print(ligand, dist, error, n) 
        #out_obs = np.zeros(coords.shape[0])
        #out_obs[inner_all] = densities[inner_all].reshape(-1,)
        #out_obs[border] =  densities[border].reshape(-1,)
        #np.savetxt( f'obs_{depth}_{error}.txt', out_obs)
        
        #out_int = np.zeros(coords.shape[0])
        #out_int[inner_all] = interp.reshape(-1,) 
        #out_int[border] =  densities[border].reshape(-1,)
        #np.savetxt(f'int_{depth}_{error}.txt', out_int)
        print('Error:', np.round(error,2))
        '''
        plot_surf(coords,faces,out_obs, rotate=[180,0], z_rotate=-30)
        plt.show()
        plt.cla()
        plot_surf(coords,faces,out_int, rotate=[180,0], z_rotate=-30)
        plt.show()
        '''
        total_area=np.sum([area(cortex_coords[ngh[i][0]], cortex_coords[ngh[i][1]],cortex_coords[ngh[i][2]]) for i in inner_all ])
        if total_area > 300 : 
            for ii, i in enumerate(inner_all) :
                print('\t',ii,area(coords[ngh[i][0]], coords[ngh[i][1]],coords[ngh[i][2]]))
                print('\t\t',coords[ngh[i][0]], coords[ngh[i][1]],coords[ngh[i][2]])
            exit(0)
        
        output_file.write(f'{ligand},{perimeter_dist},{total_area},{error},{n}\n')
    print()
    return output_file

def validate_for_ligand(filename, sphere_mesh_fn, cortex_mesh_fn, output_file, ligand, n_samples=10000, max_depth=5):
    #load ligand densities
    densities = pd.read_csv(filename,header=None).values.astype(np.float16)

    idx = densities > 0.0
    idx = idx.reshape(-1,)
    
    idx_range = np.arange(idx.shape[0]).astype(int)
    
    valid_idx_range = idx_range[idx][0:n_samples]

    #load coords and neighbours
    coords, faces = read_gifti(sphere_mesh_fn)
    cortex_coords, _ = read_gifti(cortex_mesh_fn)
    ngh = get_ngh(coords,faces)
   
    output_file = iterate_over_vertices(valid_idx_range, idx, ngh, coords, cortex_coords, faces, densities, output_file, ligand, n_samples=n_samples, max_depth=max_depth)

    return output_file

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
    file_list = glob('MR1_R_*_0.75mm_profiles_0.0_raw.csv')
    n_samples=1000   
    for filename in file_list :
        ligand = filename.split('_')[2]
        print(filename, ligand)
        output_file = validate_for_ligand(filename, sphere_mesh_fn, cortex_mesh_fn, output_file, ligand, n_samples=n_samples,max_depth=max_depth)
        break
    output_file.close()
    plot_validation(out_fn)




