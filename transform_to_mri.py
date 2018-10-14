from utils.utils import shell
import SimpleITK as sitk
from glob import glob
import os
import re
import pandas as pd
from utils.utils import set_csv

def transform_to_mri(source_dir, output_dir, slice_order_fn, clobber=False):
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    #source_files=glob(source_dir+os.sep+ "**" + os.sep + "*.tfm" )
    #transform_fn_list=[ re.sub("cropped_reg", "cropped-reg", f) for f in glob(source_dir+os.sep + "*.txt" ) ]
    transform_fn_list = glob(source_dir+os.sep + "*.txt" ) 
   
    if not os.path.exists( output_dir + os.sep + "receptor_slices.csv") :
        df = set_csv(transform_fn_list, output_dir,"","",slice_order_fn=slice_order_fn, clobber=clobber, df_with_order=True)
    else :
        df = pd.read_csv( output_dir + os.sep + "receptor_slices.csv")
    
    print( df.columns ) ;
    for index, row in df.iterrows() :
        fn = row["filename"]
        slab = row["slab"]
        slab_tfm_fn = "receptor_to_mri/slab_"+str(slab)+".tag"
        print("fn", fn)
        print("slab_tfm_fn", slab_tfm_fn)
        composite = sitk.Transform(2, sitk.sitkComposite )
        slice_transform = sitk.ReadTransform(fn)
        slab_transform = sitk.ReadTransform(slab_tfm_fn)
        composite.AddTransform(slice_transform)
        composite.AddTransform(slab_transform)
        sitk.WriteTransform(composite, 'euler2D.tfm')
        exit(0)
