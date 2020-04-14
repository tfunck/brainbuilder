from glob import glob
import matplotlib.pyplot as plt
import imageio
import os
import re
import shutil

def image_read(fn):
    img = imageio.imread(fn)
    if len(img.shape) == 3 :
        img = np.mean(img,axis=2)
    return img

train_fn_list = glob("test/train/*tif")

if not os.path.exists('test/qc') :
    os.makedirs('test/qc')

if not os.path.exists('test/train_no_labels') :
    os.makedirs('test/train_no_labels')

for train_fn in train_fn_list :
    
    plt.clf()
    plt.subplot(2,1,1)
    plt.title(train_fn)
    plt.imshow(image_read(train_fn))

    f2_root=os.path.splitext(os.path.basename(train_fn))[0]
    f2_root=re.sub('#L' ,'' , re.sub('_downsample', '',  re.sub('#_downsample', '', f2_root)))
    
    label_fn = glob("test/label/*%s*tif"%f2_root)
    print(f2_root, label_fn)

    if len(label_fn) == 1 :
        plt.subplot(2,1,2)
        plt.title(label_fn[0])
        plt.imshow(image_read(label_fn[0]),vmin=0,vmax=2)
    else : 
        shutil.move(train_fn, 'test/train_no_labels/'+os.path.basename(train_fn))

    plt.tight_layout()
    plt.savefig('test/qc/%s_qc.png'%f2_root)

