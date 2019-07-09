import os
import json
import numpy as np
import nibabel as nib
import argparse
from nibabel.processing import resample_from_to
from utils.utils import shell, downsample_and_crop
from sys import exit
from re import sub
from kmeans_vol import kmeans_vol
from glob import glob
from classifyReceptorSlices import classifyReceptorSlices
from receptorCrop import crop_source_files
from utils.utils import downsample_y
from receptorInterpolate import receptorInterpolate, receptorSliceIndicator
from receptorAdjustAlignment import receptorRegister
from detectLines import apply_model
from findMRIslab import createSRV
from slab import Slab

'''
Set global variables
'''
global ERROR_RUNTIME
global ERROR_FILENOTFOUND
ERROR_RUNTIME=1
ERROR_FILENOTFOUND=2


'''
    Helper functions
'''

def check_file(fn):
    '''
        Verify that a file exists. If it does not return with error, otherwise return path.
    '''
    if not os.path.exists(fn) :
        print("Error could not find path for", self.raw_source)
        exit(ERROR_FILENOTFOUND)
    return fn

'''
    Classes for reconstruction
'''
class Autoradiographs():
    def __init__ (self, args):
        self.raw_source=check_file(args.source+os.sep+'raw')
        self.lin_source=check_file(args.source+os.sep+'lin')
        self.brains_to_run = args.brains_to_run
        self.hemispheres_to_run = args.hemispheres_to_run
        self.slabs_to_run = args.slabs_to_run
        self.output = args.output
        self.brain={}

        
        self.brains={}
         
        # Find list of autoradiograph brains
        brain_list = self._get_brain_list()
        print(brain_list)
        # 
        for brain_path in brain_list :
            brain_id=os.path.basename(brain_path)
            brain_raw_path = self.raw_source + os.sep + brain_id
            brain_lin_path = self.lin_source + os.sep + brain_id
            if self.brains_to_run == [] or brain_id in brains_to_run :
                print(brain_id)
                self.brain[brain_id] = Brain(brain_id, self, args)

    def _get_brain_list(self) :
        brains_raw=glob(self.raw_source+os.sep+'MR*')
        brains_lin=glob(self.lin_source+os.sep+'MR*')

        brain_base_raw = set([os.path.basename(f) for f in brains_raw ])
        brain_base_lin = set([os.path.basename(f) for f in brains_lin ])

        brain_list = brain_base_raw.intersection(brain_base_lin)

        if brain_list == [] : 
            print("Error: no common directories found in")
            print("\t", brains_raw)
            print("\t",brains_lin)
            exit(ERROR_FILENOTFOUND)
        return brain_list


    def generate_mri_gm_mask(self) :
        for brain_id, brain in self.brain.items() :
            print(brain, brain_id)
            for hemi_id, hemi in brain.hemispheres.items() :
                hemi._generate_mri_gm_mask()

    def reconstruct(self,args) :
        print(self.brain)
        for brain_id, brain in self.brain.items() :
            print(brain, brain_id)
            for hemi_id, hemi in brain.hemispheres.items() :
                hemi._generate_mri_gm_mask()
                for slab_id, slab in hemi.slabs.items() :
                    print("Brain:", brain_id, "Hemisphere:", hemi_id, "Slab:", slab_id)
                    slab._reconstruct(args)


class Brain():
    def __init__ (self, brain_id, autoInstance, args):
        self.brain_id=brain_id
        self.brain_raw_path=check_file(autoInstance.raw_source+os.sep+brain_id)
        self.brain_lin_path=check_file(autoInstance.lin_source+os.sep+brain_id)
        self.hemispheres={}

        all_slabs=glob(self.brain_lin_path+os.sep+'*_slab_*')
        all_hemispheres=[]

        #Extract hemispheres from dir names. Assumes dirs are named as <hemi>_slab_<slab>, e.g., R_slab_1
        for dir_path in all_slabs :
            try :
                all_hemispheres.append(os.path.basename(dir_path).split("_")[0])
            except IndexError :
                print("Error : Incorrectly formatted path to autoradiograph slab. Directory should have format <hemisphere>_slab_<slab>, e.g., R_slab_1. Instead received :\n\t", dir_path )
          
        unique_hemispheres=np.unique(all_hemispheres)
        for hemi in unique_hemispheres :
            if args.hemispheres_to_run == [] or hemi in args.hemispheres_to_run :
                print("Adding hemisphere", hemi)
                self.hemispheres[hemi] = Hemisphere(hemi, self, args)

class Hemisphere():
    def __init__ (self,  hemi, brainInstance, args):
        self.hemi=hemi
        self.brain_id = brainInstance.brain_id
        self.brain_raw_path = brainInstance.brain_raw_path
        self.brain_lin_path = brainInstance.brain_lin_path
        self.args=args

        self.slabs={}

        slab_raw_paths = glob(self.brain_raw_path+os.sep+'*')
        slab_lin_paths = glob(self.brain_lin_path+os.sep+'*')
        
        for raw_path in slab_raw_paths :
            try :
                slab_id = raw_path.split("_")[2]
            except IndexError :
                print("Error : Incorrectly formatted path to autoradiograph slab. Directory should have format <hemisphere>_slab_<slab>, e.g., R_slab_1. Instead received :\n\t", dir_path )
                exit(ERROR_FILENOTFOUND)

            lin_path = [ f for f in slab_lin_paths if slab_id in f ] 
            
            try :
                lin_path[0]
            except IndexError : 
                print("Error : could not find corresponding path for linearized autoradiograph directory for raw autoradiograph directory :", raw_path, slab_id)
            if args.slabs_to_run == [] or slab_id in args.slabs_to_run :
                print("Adding slab:", slab_id)
                self.slabs[slab_id] = Slab( raw_path, lin_path[0], slab_id, brainInstance.brain_id, self.hemi, args )

    def _generate_mri_gm_mask(self):
        print("Generating MRI GM mask")
        gm_mask_paths = [ False for slab in self.slabs.keys() if not os.path.exists("srv/mri1_gm_bg_srv_slab-"+str(slab)+".nii.gz") ]
        if False in gm_mask_paths or args.clobber:
            createSRV(self.brain_id, self.hemi, srv_fn = "srv/mri1_gm_bg_srv.nii.gz" ) #FIXME will need to be changed when using other MRI/hemi

'''
    Command line argument parsing
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input','-i', dest='source', default='data/',  help='Directory with raw images')
    parser.add_argument('--output','-o', dest='output', default='output/', help='Directory name for outputs')
    parser.add_argument('--brains','-b', dest='brains_to_run', type=str, nargs='+', default=[],help='Brains to reconstruct. Default = run all.')
    parser.add_argument('--hemispheres', '-m', dest='hemispheres_to_run',type=str, nargs='+', default=[], help='Brains to reconstruct. Default = reconstruct all hemispheres.')
    parser.add_argument('--slabs','-s', dest='slabs_to_run', type=str,nargs='+', default=[],  help='Slabs to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--ligands','-l', dest='ligands_to_run', type=str,nargs='+', default=["flum"],  help='Ligands to reconstruct. Default = reconstruct all slabs.')
    parser.add_argument('--mri-gm-mask', dest='generate_mri_gm_mask', action='store_true', default=False, help='Only generate GM masks from donor MRI')
    parser.add_argument('--preprocess', dest='run_preprocess', action='store_true', default=False, help='Only run preprocessing in reconstruction')
    parser.add_argument('--init-alignment', dest='run_init_alignment', action='store_true', default=False, help='Only run initial alignment in reconstruction')
    parser.add_argument('--mri-to-receptor', dest='run_mri_to_receptor', action='store_true', default=False, help='Only run alignment of mri to receptor in reconstruction')
    parser.add_argument('--receptor-interpolate', dest='run_receptor_interpolate', action='store_true', default=False, help='Only run receptor interpolation to receptor in reconstruction')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    data = Autoradiographs( args )

    if args.generate_mri_gm_mask :
        data.generate_mri_gm_mask()
    else :    
        data.reconstruct(args)

