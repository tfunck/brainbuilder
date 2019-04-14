import shutil
import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import h5py
from time import time
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from imageio import imread, imwrite
from glob import glob
from skimage.transform import rescale
from skimage.exposure import  equalize_hist
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu, threshold_yen
from sklearn.utils.class_weight import compute_class_weight


def gen_rand_locations(dim,step,n_samples):
    values=np.arange(0,dim-step,step,dtype=int)
    np.random.shuffle(values)
    return values[0:n_samples]

def create_qc_img( qc_fn, imsubset, im):
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(imsubset)
    plt.plot([130,170,170,130,130],[130,130,170,170,130],c='r',linewidth=1.0)
    plt.savefig(qc_fn)

def threshold(im2) : 
    thr1 = apply_threshold(threshold_otsu, im2)
    thr2 = apply_threshold(threshold_yen, im2)
    thr3 = myKMeans(im2)
    thr = (thr1 + thr2 + thr3 ) / 3.
    thr[thr < 0.5] = 0
    thr[thr > 0.5] = 1
    return thr 

def generate_training_data(source_dir, qc_dir, tissue_dir, label_dir,  n_samples=10, xstep=300, zstep=300, total=3000, clobber=0, model=None ) :
    for d in [qc_dir, tissue_dir, label_dir] :
        if not os.path.exists(d) :
            os.makedirs(d)
    
    source_files=glob(source_dir+"/*")
    
    np.random.shuffle(source_files)
    idx=0

    for i in range(total) :
        idx = np.random.choice(range(len(source_files)), 1 )[0]
        f = source_files[idx]
        im_orig = np.max(imread(f), axis=2)

        im = im_orig
        #if True in [ True for i in  ["oxot", "epib", "ampa", "uk14", "mk80", "pire" ] if i in f ] :
        #    im = equalize_hist(im)
        
        im2 = gaussian_filter(im, 1)
        
        if np.max(im2) == np.min(im2) : continue

        threshold(im2)

        z_values = gen_rand_locations(im.shape[1],zstep,n_samples)
        x_values = gen_rand_locations(im.shape[0],xstep,n_samples)

        splitf = os.path.splitext(os.path.basename(f))

        for x, z in zip(x_values, z_values) :
            x1=x+xstep
            z1=z+zstep
            im_subset = im_orig[x:x1, z:z1 ] 
            thr_subset = thr[x:x1, z:z1 ]
            
            #Skip patches with less than 20% labels == 1
            if np.sum(thr_subset) / (xstep*zstep) < 0.2  : continue
            
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(im2 / (1+np.max(im2)) )
            plt.plot([z,z,z1,z1,z],[x,x1,x1,x,x], c='r', linewidth=1)
            plt.subplot(2,2,2)
            plt.imshow(thr)
            plt.plot([z,z,z1,z1,z],[x,x1,x1,x,x], c='r', linewidth=1)
            plt.subplot(2,2,3)
            plt.imshow(im_subset / np.max(im_subset))
            plt.subplot(2,2,4)
            plt.imshow(im_subset * thr_subset / np.max(im_subset))
            qc_fn = qc_dir + os.sep + splitf[0] +'_'+str(x)+'_'+str(z)+'.png'
            plt.savefig(qc_fn, figsize=(40,40), dpi=300)

            img_fn = tissue_dir + os.sep + splitf[0] + '.tif'
            imwrite(img_fn, im_subset)
            
            label_fn =label_dir + os.sep + splitf[0] + '.png'
            imwrite(label_fn, thr_subset)

    

def pad_size(d):
    if d % 2 == 0 :
        pad = (int(d/2), int(d/2))
    else :
        pad = ( int((d-1)/2), int((d+1)/2))
    return(pad)

def add_padding(img, zmax, xmax):
    z=img.shape[0]
    x=img.shape[1]
    dz = zmax - z
    dx = xmax - x
    z_pad = pad_size(dz)
    x_pad = pad_size(dx)
    img_pad = np.pad(img, (z_pad,x_pad), 'minimum')
    return img_pad

def get_batch_size(train_fn, val_fn, limit=300) :
    train_x = h5py.File(train_fn, 'r')
    val_x = h5py.File(val_fn, 'r')
    
    n=val_x["x"].shape[0]
    m=train_x["x"].shape[0]
    batch_size_list = np.array([  i for i in range(1,m+1) if m % i == 0 ])
    batch_size = max(batch_size_list[batch_size_list < limit])
    batch_size=10
    #batch_size=1
    steps=int(np.floor(m/batch_size))
    val_steps=int(np.floor(n/batch_size))
    return batch_size, steps, val_steps 

def get_image_filenames(tissue_dir, label_dir):
    ###Load Images
    tissue=glob(tissue_dir+"/*.tif")
    label=glob(label_dir+'/*.png')
    x = np.sort( tissue)
    y = np.sort( label )

    idx = np.random.choice(range(len(x)), len(x), replace=False)
    x = x[idx]
    y = y[idx]
    return x, y

def read_format_images(fn, start_idx, end_idx, xstep, zstep, x,y, use_augmentation=False  ):
    identity = lambda x : x
   
    n=end_idx - start_idx

    data = h5py.File(fn, "w")
    data.create_dataset("x", (n, xstep, zstep), dtype='float16')
    data.create_dataset("y", (n, xstep, zstep), dtype='float16')

    for i, files in enumerate(zip(x[start_idx:end_idx], y[start_idx:end_idx])):
        img = imread(files[0])
        label = imread(files[1])
        
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        if np.max(img) != 0 and np.max(img) != np.min(img) : 
            img = (img - np.min(img)) / (np.max(img)- np.min(img))

        if np.max(label) != 0 : 
            label = label / np.max(label)
        else :
            print("Error: all 0 label image ", files[1])
            exit(1)
        data["x"][i] = img.reshape(1,*label.shape)
        data["y"][i] = label.reshape(1,*label.shape)

    return 0

def save_h5(fn, data) :
    f = h5py.File(fn, "w")
    X = f.create_dataset("image", *data.shape, dtype='float16')
    X = data

def create_train_val_data(train_fn, val_fn, train_ratio, val_ratio,tissue_dir, label_dir, xstep=300, zstep=300, clobber=0):
    x, y = get_image_filenames(tissue_dir, label_dir)

    ### Create Training and Validation sets
    train_idx = int(len(x) * train_ratio)
    val_idx = int(train_idx + len(x) * val_ratio)
    new_class_weights=False

    if not os.path.exists(train_fn) or not os.path.exists(val_fn) or clobber > 1  :
        print("Compiling New Training Data")
        read_format_images(train_fn, 0, train_idx, xstep, zstep, x, y )
        read_format_images(val_fn, train_idx, val_idx, xstep, zstep, x , y)
        new_class_weights=True

    return new_class_weights

def gen(fn, batch_size=1,xdim=300,zdim=300):
    from keras.utils import to_categorical
    data = h5py.File(fn, 'r')
    n=data["x"].shape[0]
    while True :
        for i in range(0,n,batch_size) :
            batch_size_0 = batch_size if i+batch_size < n else n - i
            x=data["x"][i:(i+batch_size_0), :, :]
            y=to_categorical(data["y"][i:(i+batch_size_0), :, :], 2)
            x=x.reshape( *x.shape, 1)
            y=y.reshape( batch_size_0, xdim*zdim*2 )
            yield x, y 


def make_compile_model(train_x_shape, batch_size, xdim=300,zdim=300 ) :
    import keras
    from keras.models import Model
    from keras.engine.topology import Input
    from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
    from keras.layers import BatchNormalization, ZeroPadding2D, Concatenate
    from keras.layers.core import Dropout, Dense, Flatten, Reshape
    from keras.layers import LeakyReLU, MaxPooling2D, concatenate
    from keras.activations import relu
    print(train_x_shape[1:]) 
    IN = Input(shape=(  *train_x_shape[1:],1  ))
    DO=0.05

    ks=5
    nk=32
    #1 --> 300
    CONV1A=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(IN))
    CONV1B=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1A))
    CONV1C=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1B))
    CONV1D=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1C))
    CONV1E=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1D))
    #CONV1F=Dropout(DO)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1E))
    CONV5B=CONV1E
    
    #POOL1 = MaxPooling2D(pool_size=(2,2),padding='same')(CONV1B)
    #2 --> 150
    #CONV2A=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(POOL1))
    #CONV2B=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV2A))
    #POOL2=MaxPooling2D(pool_size=(2,2), padding='same')(CONV2B)
    #3 --> 75
    #CONV3A=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(POOL2))
    #CONV3B=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV3A))
    #3 --> 150
    #UP1 = UpSampling2D(size=(2, 2))(CONV3B)
    #CONC1 = Concatenate(axis=3)([UP1, CONV2B])
    #CONV4A=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONC1))
    #CONV4B=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV4A))
    #2 --> 300
    #UP2 = UpSampling2D(size=(2, 2))(CONV4B)
    #CONC2 = Concatenate(axis=3)([UP2, CONV1B])
    #CONV5A=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONC2))
    #CONV5B=Dropout(0.2)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV5A))
    CONV6 = Conv2D( 2, kernel_size=[1,1],activation='softmax',padding='same')(CONV5B)
    OUT = Reshape((xdim*zdim*2,))(CONV6)
    model = Model(inputs=[IN], outputs=OUT)

    ada = keras.optimizers.Adam(0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer=ada, weighted_metrics=['acc'] )
    print(model.summary())
    return model

def to_2D(x, xstep=300,zstep=300):
    return np.argmax(x.reshape(xstep,zstep,2),axis=2).reshape(xstep,zstep)


def get_class_weights(train_fn, class_weights_fn, new_class_weights ):
    if not os.path.exists(class_weights_fn) or new_class_weights :
        train_y=h5py.File(train_fn, 'r')["y"]
        train_y_shape=train_y.shape
        class_weights = compute_class_weight("balanced", np.array([0,1]), np.array(train_y).flatten())
        class_weights_dict = {"0":str(class_weights[0]), "1":str(class_weights[1])}
        json.dump(class_weights_dict, open(class_weights_fn, 'w'))
    else :
        class_weights_str = json.load(open(class_weights_fn, 'r'))
        class_weights={ 0:float(class_weights_str["0"]), 1:float(class_weights_str["1"])} 
    print("Class Weights:", class_weights)
    return class_weights

def train_model( model_fn, tissue_dir,label_dir, ratios=[.75,0.25], epochs=50, clobber=0):
    from keras.callbacks import History, ModelCheckpoint
    train_ratio=ratios[0]
    val_ratio=ratios[1]
    train_fn = tissue_dir+"/train.h5"
    val_fn = tissue_dir+"/val.h5"
    new_class_weights = create_train_val_data(train_fn, val_fn, train_ratio, val_ratio,tissue_dir, label_dir, clobber=clobber)

    ### Calculate batch size
    batch_size, steps, val_steps = get_batch_size(train_fn, val_fn, limit=10)
    
    ### Get Class Weights
    class_weights = get_class_weights(train_fn, label_dir + '/class_weights.json', new_class_weights)

    ### Get shape of data
    train_x_shape=h5py.File(train_fn, 'r')["x"].shape

    ### Make Model
    model = make_compile_model(train_x_shape, batch_size, xdim=300,zdim=300)
    print("Batch Size:", batch_size)
    #print('Data Points:', train_x_shape, val_x.shape[0])

    ### Fit Model
    history = model.fit_generator( gen(train_fn,  batch_size=batch_size), 
                steps_per_epoch=steps, 
                validation_data=gen(val_fn), 
                validation_steps=val_steps, 
                epochs=epochs,
                class_weight=class_weights
                )
    model.save(model_fn)
    

    return model


def apply_model_to_validation(model_fn,val_fn,out_dir):

    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)
    if not os.path.exists(val_fn) :
        print("Error: could not find ", val_fn)
    val_x = np.array(h5py.File(val_fn, 'r')['x'])
    val_x = val_x.reshape(*val_x.shape,1)
    val_y = h5py.File(val_fn, 'r')['y']

    val_y_predicted = model.predict(val_x, batch_size=10)
    for i in range(0, val_x.shape[0] , 1):
        print(val_y[i].dtype)
        o=val_x[i].reshape(300,300).astype(float)
        gt=np.array(val_y[i]).astype(int)
        p=to_2D(val_y_predicted[i])

        plt.subplot(3,1,1)
        plt.title('Original')
        plt.imshow(o)
        plt.subplot(3,1,2)
        plt.title('Ground Truth Label '+str(np.round(np.sum(gt==1)/(300*300),1)))
        plt.imshow(gt)
        plt.subplot(3,1,3)
        plt.title('Predicted by Network')
        plt.imshow(p)
        out_fn=out_dir+os.sep+'val_'+str(i)+'_.png'
        print(out_fn)
        plt.tight_layout()
        plt.savefig(out_fn)
        plt.clf()
        del o
        del gt
        del p 

def apply_model_to_images(model, in_dir, out_dir, xstep=300, zstep=300, batch_size=5):
    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)
    
    source_files = glob(in_dir+"/*.tif")
    np.random.shuffle(source_files)

    stride=300
    for f in source_files[1:40] :
        print(os.path.basename(f))
        img = np.max(imread(f), axis=2)
        img = (img -np.min(img))/ (np.max(img)-np.min(img))
        out = np.zeros( img.shape)
        pts_array = np.zeros( img.shape )
        points=[]
        subsetList=[]
        for x in range(0, img.shape[0], stride) :
            for z in range(0,img.shape[1], stride) :
                subset = img[x:(x+xstep),z:(z+zstep)]
                if subset.shape[0] < xstep or subset.shape[1] < zstep :
                    subset = add_padding(subset, xstep, zstep)
                subset = subset.reshape(1,xstep,zstep,1)
                #subsetList.append( subset )
                #points.append( [x, z] )     

                xstep_0 = xstep if x+xstep < img.shape[0] else img.shape[0] - x
                zstep_0 = zstep if z+zstep < img.shape[1] else img.shape[1] - z
                predicted =to_2D( model.predict(subset))
                out[x:(x+xstep_0),z:(z+zstep_0)]=predicted[0:xstep_0,0:zstep_0]
        #subsetVolume = np.concatenate(subsetList, axis=0)
        #subsetVolume = subsetVolume.reshape((*subsetVolume.shape,1))
        
        #predicted_array = model.predict(subsetVolume, batch_size=batch_size )
        #print(np.min(predicted_array), np.max(predicted_array), np.mean(predicted_array))
        #tissue_predicted=[]
        #for i in range(0,predicted_array.shape[0]):
        #    print("\t", subsetVolume[i].shape)
        #    print("\t",np.min(subsetVolume[i]), np.max(subsetVolume[i]), np.median(subsetVolume[i]))
        #    img_slice = to_2D(predicted_array[i])
        #    print("\t\t",np.min(img_slice[i]), np.max(img_slice[i]), np.median(img_slice[i]))
        #    x=points[i][0]
        #    z=points[i][1]
        #    pts_array[x:(x+xstep),z:(z+zstep)] += 1
        #    xstep_0 = xstep if x+xstep < img.shape[0] else img.shape[0] - x
        #    zstep_0 = zstep if z+zstep < img.shape[1] else img.shape[1] - z

        #    out[x:(x+xstep_0),z:(z+zstep_0)]=np.ones((xstep_0,zstep_0)) * i #img_slice[0:xstep_0,0:zstep_0]

        plt.subplot(2,1,1)
        plt.imshow(img / np.max(img))
        plt.subplot(2,1,2)
        plt.imshow(out )
        plt.savefig(out_dir+os.sep+os.path.splitext(os.path.basename(f))[0]+'.png' )
        plt.clf()

def apply_threshold(func, im ) :
    oo = np.zeros(im.shape)
    t = func(im)
    oo[ im < t ] = 1
    return oo

def myKMeans(s):
    s = np.max(s) - s 
    upper=np.max(s)
    mid = np.median(s[s>0])
    init=np.array([0, mid, upper]).reshape(-1,1)
    cls = KMeans(3, init=init).fit_predict(s.reshape(-1,1)).reshape(s.shape)
    cls[ cls != 2 ] = 0
    cls[cls == 2] = 1
    return cls

def get_x_z(f):
    fsplit = os.path.splitext(f)[0].split('_')
    x=int(fsplit[-2])
    z=int(fsplit[-1])
    return x, z

def get_source_from_basename(base, source_dir) :
    fn=source_dir + os.path.splitext(base)[0].split('_')[0] + '*.tif'
    
    source_fn =  glob(fn)
    if len(source_fn) == 0 : 
        return 1
    source_fn = source_fn[0]
    return source_fn

def fill_in_from_qc(f,source_dir,xstep=300,zstep=300):
    base = os.path.basename(os.path.splitext(f)[0])
    base = base.split('_')[0]
    splitf = os.path.splitext(os.path.basename(f))
    
    source_fn = get_source_from_basename(base, source_dir) 
    if source_fn == 1 : 
        print("\tCould not find source for\n\t\t", source_dir,"\t\t", base)
        return 1

    x, z = get_x_z(f)

    im = np.max(imread(source_fn), axis=2)
    x1=x+xstep
    z1=z+zstep

    imsubset = im[x:(x+xstep), z:(z+zstep)]
    im2 = gaussian_filter(im, 5)
    thr = threshold(im2)
    thr_subset = thr[x:x1, z:z1 ]

    img_fn = tissue_dir + os.sep + splitf[0] + '.tif'
    imwrite(img_fn, imsubset)
    
    label_fn = label_dir + os.sep + splitf[0] + '.png'
    #label = imread(label_fn)
    #label = np.max(label) - label
    imwrite(label_fn, thr_subset)

def validate_data( qc_dir, label_dir, tissue_dir, source_dir ):
    func = lambda x : [ os.path.basename(os.path.splitext(f)[0]) for f in x ] 
    qc_files = glob(qc_dir+"/*png")
    label_files = glob(label_dir+"/*png")
    tissue_files = glob(tissue_dir+"/*.tif")
    qc_basenames = func( qc_files  )
    label_basenames = func( label_files )
    tissue_basenames = func(tissue_files)

    for f, base in zip(label_files, label_basenames) :
        if not base in qc_basenames or not base in tissue_basenames: os.remove(f)
        if np.max(imread(f)) == 0 : 
            #os.remove(f)
            pass

    for f, base in zip(tissue_files, tissue_basenames) :
        if not base in qc_basenames or not base in label_basenames  : os.remove(f)

    for f, base in zip(qc_files, qc_basenames) :
        if not base in label_basenames or not base in tissue_basenames :
            print("fill in ", f)
            #os.remove(f)
            fill_in_from_qc(f,source_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')
    parser.add_argument('--apply-model', dest='apply_model', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--create-init-data', dest='create_init_data', type=int,  default=0, help='Clobber results')
    parser.add_argument('--generate-new-data', dest='generate_data_from_model', type=int, default=0, help='Clobber results')
    parser.add_argument('--ratios', dest='ratios', nargs='+', default=[0.7,0.3], help='Clobber results')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Clobber results')

    args = parser.parse_args()

    ratios=[float(i) for i in args.ratios ] 
    
    create_init_data= args.create_init_data
    generate_data_from_model = args.generate_data_from_model
    clobber= args.clobber
    
    #Base Options
    tissue_dir='tissue_subset/cls/tissue'
    label_dir='tissue_subset/cls/label/'
    qc_dir='tissue_subset/cls/qc'
    test_qc_dir='tissue_subset/cls/test_qc'
    source_dir = "img_orig/R_slab_*/"


    if create_init_data :
        generate_training_data(source_dir, test_qc_dir, tissue_dir, label_dir,  total=create_init_data)
    
    #Get Model
    model_fn = "tissue_subset/model.h5"
    if not os.path.exists(model_fn) or clobber > 0 :
        validate_data( qc_dir, label_dir, tissue_dir, source_dir  )
        model=train_model(model_fn, tissue_dir, label_dir, epochs=args.epochs, ratios=ratios, clobber=clobber)

    if generate_data_from_model > 0 :
        from keras.models import load_model
        model=load_model(model_fn)
        generate_training_data(source_dir, test_qc_dir, tissue_dir, label_dir, model=model, total=generate_data_from_model)
    
    if args.apply_model :
        from keras.models import load_model
        model=load_model(model_fn)
        #Apply model to images
        #apply_model_to_validation(model, tissue_dir+'/val.h5',"tissue_subset/predict")
        apply_model_to_images(model, "img_orig/R_slab_1", "tissue_subset/predict")
