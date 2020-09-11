import ants
import numpy as np
import argparse
import os
import tempfile
from sys import argv
from mesh_io import  save_obj, read_obj
from re import sub
from utils.utils import shell, splitext


def apply_ants_transform_to_obj( in_obj_fn, tfm_list, out_obj_fn, invert):
    coords, faces = read_obj(in_obj_fn)
    tfm = ants.read_transform(tfm_list[0])
    flip = 1
    if np.sum(tfm.fixed_parameters) != 0 : flip=-1
    
    in_file = open(in_obj_fn, 'r')
    out_file=open(out_obj_fn, 'w+')

    coord_fn = sub('.obj', '.csv', in_obj_fn)
    out_coord_fn = sub('.obj', '.csv', out_obj_fn)

    #read the csv with transformed vertex points
    with open(coord_fn, 'w+') as f :  
        f.write('x,y,z,t,label\n') 
        for x,y,z in coords :  
            f.write('{},{},{},{},{}\n'.format(flip*x,flip*y,z,0,0 ))

    #-t [{tfm_list[0]},{invert[0]}] 
    temp_out_fn=tempfile.NamedTemporaryFile().name+'.csv'
    shell(f'antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert[0]}]  -o {temp_out_fn}',verbose=True)

    # save transformed surfaced as an obj file
    with open(temp_out_fn, 'r') as f :
        #read the csv with transformed vertex points
        for i, l in enumerate( f.readlines() ):
            if i == 0 : continue
            x,y,z,a,b = l.rstrip().split(',')
            coords[i-1] = [flip*float(x),flip*float(y),float(z)]
    
    with open(out_coord_fn, 'w+') as f :
        #read the csv with transformed vertex points
        f.write('x,y,z,t,label\n')
        for i, ( x,y,z ) in enumerate(coords) :
           f.write('{},{},{},0,0\n'.format(x,y,z))

    save_obj(out_obj_fn, coords, faces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--in','-i', dest='in_obj_fn', type=str,  default='in_obj_fn', help='')
    parser.add_argument('--tfm','-t', dest='tfm_list', type=str, nargs='+', default=[], help='')
    parser.add_argument('--out','-o', dest='out_obj_fn', type=str,  default='out_obj_fn', help='')
    parser.add_argument('--invert','-v', dest='invert',nargs='+', default=[],  help='')
    args = parser.parse_args()

    apply_ants_transform_to_obj( args.in_obj_fn, args.tfm_list, args.out_obj_fn, args.invert)

