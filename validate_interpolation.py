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
        if len(ngh[key]) <= 3 :
            print(len(ngh[key]))
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(coords[key,0],coords[key,1],coords[key,2], c='g')
            #ax.scatter(coords[ngh[key]][:,0],coords[ngh[key]][:,1],coords[ngh[key]][:,2], c='r')
            #plt.show()    
    return ngh

def calculate_neighbours(i, ngh, coords, densities, max_depth) :
    ngh_core = [i]
    ngh_core_all = [i]
    ngh_list = ngh[i]
    dist_list = []
    #print('\n',coords.shape[0])
    #max_depth=10
    for depth in range(max_depth):
        #print('\tdepth', max_depth, depth, len(ngh_list))
        new_ngh=[]
        border=[]
        dist_list=[]
        for ii in ngh_list :
            if densities[ii] > 0 :
                new_ngh += list(ngh[ii])
                border.append(ii)

        if depth == 0 :
            ngh_core += border

        if depth < max_depth-1:
            ngh_core_all += border 
        del ngh_list
        ngh_list = np.unique(new_ngh)
    
    border = list(np.unique([ i for i in border if not i in ngh_core_all ] ))
    ngh_core = list(np.unique(ngh_core))
    ngh_core_all = list(np.unique(ngh_core_all))

    for ii in border :
        dist_list.append( np.sqrt(np.sum(np.power(coords[i] - coords[ii], 2))) )

    dist = np.mean(dist_list)
    
    return ngh_core, ngh_core_all, border, dist

def interpolate_over_sphere(densities, coords_all, faces, idx_core, idx, border, i):
    spherical_coords = surface_tools.spherical_np(coords_all)
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
    
    error = np.abs(np.mean(interp_val[0:len(idx_core)]) - np.mean(densities[idx][0][0:len(idx)])) /  np.mean(densities[idx][0][0:len(idx)])


    plt.subplot(2,1,1)
    plt.title('observed')
    plt.scatter(lons[0:len(idx)], lats[0:len(idx)],s=400,cmap='hot', c=densities[idx])
    plt.scatter(lons[len(idx):len(idx+border)], lats[len(idx):len(idx+border)],s=500,cmap='hot',marker='p', c=densities[border])
    plt.subplot(2,1,2)
    plt.title('interpolated')
    plt.scatter(lons[0:len(idx)], lats[0:len(idx)],s=400,cmap='hot', c=interp_val[0:len(idx)])
    plt.scatter(lons[len(idx):len(idx+border)], lats[len(idx):len(idx+border)],s=500,cmap='hot',marker='p', c=densities[border])

    plt.show()
    return  error, interp_val[0:len(idx)]

def iterate_over_vertices(idx_range, idx, ngh, coords, faces, densities, output_file,ligand,max_depth=5, n_samples=10000):
    for i, (x,y,z) in zip(idx_range, coords[idx,:]): 
        depth = np.random.randint(1,max_depth+1)
        depth = 7
        
        if i % 10 == 0 : print(100*i/idx.shape[0],end='\r')
        #print(f'U(1,{max_depth})->{depth}, {i}',end='\r')
        # calculate neighbours
        core, core_all, border, dist = calculate_neighbours(i, ngh, coords, densities, depth)
        
        n = len(border)
        if n < 4 : 
            #print('\tSkipping')
            continue
        # interpolate small patch
        error, interp = interpolate_over_sphere(densities, coords, ngh, core, core_all, border, i)
        #error = 5 if error > 5 else error
        #if error > 0.5 : print(ligand, dist, error, n) 
        exit(0)
        out_obs = np.zeros(coords.shape[0])
        out_obs[core_all] = densities[core_all].reshape(-1,)
        out_obs[border] =  densities[border].reshape(-1,)
        np.savetxt( f'obs_{depth}_{error}.txt', out_obs)
        
        out_int = np.zeros(coords.shape[0])
        out_int[core_all] = interp.reshape(-1,) 
        out_int[border] =  densities[border].reshape(-1,)
        np.savetxt(f'int_{depth}_{error}.txt', out_int)
        print('Error:', error)
        plot_surf(coords,faces,out_obs, rotate=[180,0], z_rotate=-30)
        plt.show()
        plt.cla()
        plot_surf(coords,faces,out_int, rotate=[180,0], z_rotate=-30)
        plt.show()
        exit(0)

        output_file.write(f'{ligand},{dist},{error},{n}\n')
    print()
    return output_file

def validate_for_ligand(filename, output_file, ligand, n_samples=10000, max_depth=5):
    #load ligand densities
    densities = pd.read_csv(filename,header=None).values.astype(np.float16)

    idx = densities > 0.0
    idx = idx.reshape(-1,)
    
    idx_range = np.arange(idx.shape[0]).astype(int)
    
    valid_idx_range = idx_range[idx][0:n_samples]

    #load coords and neighbours
    coords, faces = read_gifti(mesh_fn)
    ngh = get_ngh(coords,faces)
   
    output_file = iterate_over_vertices(valid_idx_range, idx, ngh, coords, faces, densities, output_file, ligand, n_samples=n_samples, max_depth=max_depth)

    return output_file

    return output_file
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--out-dir', dest='out_dir', default='/scratch/tfunck/output2/5_surf_interp/', help='output directory')
    parser.add_argument('--mesh', dest='mesh_fn', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='mesh fn')
    parser.add_argument('--ligand', dest='file_list', nargs='+', default='/scratch/tfunck/output2/5_surf_interp/surfaces/surf_0.5mm_1.0_inflate_rsl.surf.gii', help='csv ligand densities')
    args = parser.parse_args()
    out_dir = args.out_dir
    mesh_fn = args.mesh_fn
    file_list = args.file_list

    out_fn='validation.csv'
    output_file = open(out_fn, 'w+')
    output_file.write('ligand,distance,error,n\n')
    
    max_depth = 40 
    file_list = glob('MR1_R_*_0.75mm_profiles_0.0_raw.csv')
    n_samples=1000    
    for filename in file_list :
        ligand = filename.split('_')[2]
        print(filename, ligand)
        output_file = validate_for_ligand(filename, output_file, ligand, n_samples=n_samples,max_depth=max_depth)
        break
    output_file.close()
    plot_validation(out_fn)




