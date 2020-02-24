import os
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
import h5py as h5
from re import sub
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import tensorflow as tf
from tensorflow.keras.layers import *
#from tensorflow.layers.merge_ops import merge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, merge, ZeroPadding2D
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import History, ModelCheckpoint
from utils import *
from glob import glob
from utils.utils import downsample
from keras import backend as K
from skimage.transform import rotate, resize 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from train_model import dice
from scipy.ndimage.filters import gaussian_filter



def gen(data,batch_size=1, idx_range=None):
    if idx_range==None : idx_range=(0,X.shape[0])
    i=idx_range[0]
    mod=idx_range[1]-idx_range[0]
  
    while True :
        x=data['x'][i:(i+batch_size)]
        y=data['y'][i:(i+batch_size)]
        dim = list(x.shape[0:3])+[1]
        x = x.reshape( dim )
        
        y[ y <= 90] = 0
        y[ (y > 90 ) & (y<200)] = 1
        y[ y>=200 ] = 2
        #plt.clf()
        #plt.imshow(y[0,:,:].astype(float))
        #plt.show()
        y = to_categorical(y.reshape( dim ).astype(np.uint16), 3)

        i = idx_range[0] + (i - idx_range[0] + batch_size) % data['x'].shape[0]
        yield x, y 

def dice(y_true, y_pred):
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return (2*overlap +1e-10) / (total + 1e-10)


def load_image(fn,  step, clobber, interp=2) :
    
    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = np.mean(img, axis=2)
    if not os.path.exists(fn2) or clobber :
        img = downsample(img, step=step, interp=2)
        if interp == 0 : 
            #plt.subplot(1,3,2)
            #plt.imshow(img)
            idx = img > np.max(img) * 0.05
            img[ idx ] = 1
            img[ ~idx ] = 0
            #print(np.sum(idx), np.sum(~idx))
            #if np.max(img) == 0 :
            #plt.subplot(1,3,3)
        imageio.imsave(fn2, img)
    else :
        img = imageio.imread(fn2)
        if len(img.shape) == 3 : img = np.mean(img, axis=2)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return(img)

def pair_train_and_labels(train_list, labels_list, step) :
    new_train_list=[]
    new_label_list=[]
    
    for i, f  in enumerate(train_list) :
        if i % 100 == 0 : print(1.*i/len(train_list))
        f2_root=os.path.splitext(os.path.basename(f))[0]
        f2 = [ f for f in glob('test/label/' + f2_root + "*") if not 'downsample' in f ]

        if f2 != [] : 
            f2=f2[0]
            f_rsl = sub('.tif','_downsample.png', f)
            f2_rsl= sub('.tif','_downsample.png', f2)
            if not os.path.exists(f_rsl) :
                downsample( imageio.imread(f), step=step,  interp=2, subject_fn=f_rsl)

            if not os.path.exists(f2_rsl) :
                downsample( imageio.imread(f2), step=step,  interp=0, subject_fn=f2_rsl)

            new_train_list.append( f_rsl )
            new_label_list.append( f2_rsl )
        else : 
            continue

    return new_train_list, new_label_list

def safe_imread(fn) :
    print(fn)
    img = imageio.imread(fn)
    if len(img.shape) > 2 :
        img = np.mean(img,axis=2)
    return img

def fn_to_array(h5_fn, step,clobber=False, train_str='test/train/*tif', label_str='test/label/*tif') :
    x_list  = [ f for f in glob(train_str) if not 'downsample' in f ]
    y_list = [ f for f in glob(label_str) if not 'downsample' in f ]
   
    if not os.path.exists(h5_fn) or clobber :
        x_list, y_list = pair_train_and_labels(x_list, y_list, step)
        n=len(x_list)
        image = imageio.imread(x_list[0])
        ysize=image.shape[0]
        xsize=image.shape[1]
        data = h5.File(h5_fn, 'w')
        data.create_dataset("x", (n, ysize, xsize), dtype='float16')
        data.create_dataset("y", (n, ysize, xsize), dtype='float16')
        
        for i, (x_fn, y_fn)  in enumerate(zip(x_list, y_list)) :
            data['x'][i,:,:]=safe_imread(x_fn) #load_image(x_fn, step, clobber=clobber, interp=3)
            data['y'][i,:,:]=safe_imread(y_fn) #load_image(y_fn, step, clobber=clobber, interp=0)
        


import cv2
from utils.utils import *
def augment(images, deg_list, stretch_list, h_list,  binary=False) :
    y0=ydim = images.shape[1]
    x0=xdim = images.shape[2]

    n = len(h_list) * len(deg_list) * len(stretch_list) * images.shape[0]
    out = np.zeros([n,ydim,xdim,1])
    print(out.shape)
    j=0
    for i in range(images.shape[0]) :
        img = images[i, :].reshape(ydim,xdim)
        #Flip
        for deg in deg_list :
            img1 = rotate(img, deg) 
            #Stretch
            for stretch in stretch_list  :
                y1=int(round(stretch*ydim))
                x1=int(round(stretch*xdim))
                t = resize(img1, (y1,x1)  )
                if stretch < 1 :
                    xr = int((x0-x1)/2)
                    yr = int((y0-y1)/2) 
                    img2 = np.zeros([ydim,xdim])
                    img2[yr:(yr+y1),xr:(xr+x1)] = t
                elif stretch > 1 :
                    xr = int((x1-x0)/2)
                    yr = int((y1-y0)/2)
                    img2 = t[yr:(y0+yr),xr:(x0+xr) ]
                else :
                    img2=t
                #Histogram modify
                for h_func in h_list :
                    img3 = h_func(img2)
                    out[j] = img3.reshape(ydim,xdim,1)                
                    j += 1
    return out

def make_compile_model(masks) :
    image = Input(shape=( masks.shape[1], masks.shape[2], 1))
    IN = BatchNormalization()(image)
    #n_dil=[1,1,1,1,1,1,1] #[1,2,4,8,16,1]
    #n_dil=[0,0,0,0,0] #[1,2,4,8,16,1]

    nK=[8,8,16,16,32,32,64,64] 
    n_layers=int(len(nK))
    kDim=[3] * n_layers
    #nK=[10] * n_layers
    #for i in range(n_layers):
    #    OUT = Conv2D( nK[i] , kernel_size=[kDim[i],kDim[i]],activation='relu',padding='same')(OUT)
        #OUT = Conv2D( nK[i] , kernel_size=[kDim[i],kDim[i]], dilation_rate=(n_dil[i],n_dil[i]),activation='relu',padding='same')(OUT)
    #    OUT = Dropout(DO)(OUT)
    #OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(CONV8)
    DO=0.1
    N0=20
    N1=N0*2
    N2=N1*2
    N3=N2*2
    #LEVEL 1
    CONV1 = Conv2D( N0 , kernel_size=[3,3],activation='relu',padding='same')(IN)
    CONV1 = Dropout(DO)(CONV1)
    POOL1 = MaxPooling2D(pool_size=(2, 2))(CONV1)
    #LEVEL 2

    CONV2 = Conv2D( N1 , kernel_size=[3,3],activation='relu',padding='same')(POOL1)
    CONV2 = Dropout(DO)(CONV2)
    POOL2 = MaxPooling2D(pool_size=(2, 2))(CONV2)

    #LEVEL 3
    CONV3 = Conv2D( N2 , kernel_size=[3,3],activation='relu',padding='same')(POOL2)
    CONV3 = Dropout(DO)(CONV3)
    POOL3 = MaxPooling2D(pool_size=(2, 2))(CONV3)

    #LEVEL 4
    CONV4 = Conv2D( N3 , kernel_size=[3,3],activation='relu',padding='same')(POOL3)
    CONV4 = Dropout(DO)(CONV4)


    #LEVEL 3
    CONV4_UP = UpSampling2D(size=(2, 2))(CONV4)
    CONV4_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV4_UP)
    UP1 = Concatenate()([CONV4_PAD, CONV3])#, mode='concat', concat_axis=3)

    CONV5 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    CONV5 = Dropout(DO)(CONV5)

    #LEVEL 2
    CONV5_UP = UpSampling2D(size=(2, 2))(CONV5)
    CONV5_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV5_UP)
    UP2 = Concatenate()([CONV5_PAD, CONV2])#, mode='concat', concat_axis=3)
    CONV6 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(UP2)
    CONV6 = Dropout(DO)(CONV6)

    #Level 1
    CONV6_UP = UpSampling2D(size=(2, 2))(CONV6)
    CONV6_PAD = ZeroPadding2D( ((0,0),(1,0)) )(CONV6_UP)
    UP3 = Concatenate()([CONV6_PAD, CONV1])#, mode='concat', concat_axis=3)
    CONV7 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP3) #MERGE1)
    CONV7 = Dropout(DO)(CONV7)
    OUT = Conv2D(3, kernel_size=1,  padding='same', activation='sigmoid')(CONV7)

    model = Model(inputs=[image], outputs=OUT)

    #ada = keras.optimizers.Adam(0.0001)
    ada = tf.keras.optimizers.Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer=ada,metrics=[dice] )
    print(model.summary())
    return model


def predict_results(source_dir, output_dir, model, images_val, masks_val, _use_radon ):
    if not os.path.exists(source_dir+os.sep+'results') : os.makedirs(source_dir+os.sep+'results')
    ydim=images_val.shape[1]
    xdim=images_val.shape[2]
    for i in range(images_val.shape[0]) :
        img=images_val[i,:].reshape([1,ydim,xdim,1])
        X = model.predict(img, batch_size=1)
   

        plt.subplot(1,3,1)
        plt.imshow(images_val[i].reshape(ydim,xdim) )
        plt.subplot(1,3,2)
        plt.imshow(masks_val[i].reshape(ydim,xdim))
        plt.subplot(1,3,3)
        plt.imshow(X.reshape(ydim,xdim))
        plt.savefig(output_dir+os.sep+str(i)+'.tif')

def train_model(source_dir, output_dir, step, epochs, clobber) :
    train_dir=source_dir+os.sep+'train'
    label_dir=source_dir+os.sep+'labels'
    data_fn = output_dir +os.sep +'data.h5'
    #generate_models(source_dir, output_dir, step, epochs, clobber)
    
    deg_list = [0]
    stretch_list = [1]
    identity = lambda x : x
    equalize = lambda x : cv2.equalizeHist( x.astype(np.uint8) )
    h_list_0 = [ identity, equalize]
    h_list_1 = [ identity] * len(h_list_0)
    n_aug= len(deg_list) * len(stretch_list) * len(h_list_0)

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    if not os.path.exists(source_dir+os.sep+'train.h5') or not os.path.exists(source_dir+os.sep+'labels.h5') or clobber  :
        fn_to_array(data_fn, step, clobber=clobber)
   
    data = h5.File(data_fn,'r' )
    ratio=0.8
    n_images= data['x'].shape[0]
    n_train = int(round(ratio * n_images) )
    n_val = n_images - n_train
    batch_size=1
	
    model = make_compile_model(data['x']) 
    steps=int(np.floor(n_train/batch_size))
    val_steps=int(np.floor(n_val/batch_size))
    model_name=source_dir+os.sep+"model.hdf5"
    checkpoint_fn = os.path.splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{dice:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_dice', verbose=0, save_best_only=True, mode='max')

    steps=max(1, int(np.floor(n_train/batch_size)) )
    val_steps=max(1, int(np.floor( (n_images-n_train)/batch_size)) )

    history = model.fit_generator( gen(data, batch_size, idx_range=(0,n_train)), 
            steps_per_epoch=steps, 
            validation_data=gen(data, batch_size, idx_range=(n_train, n_images)), 
            validation_steps=val_steps, 
            epochs=epochs, 
            callbacks=[ checkpoint]
            )

    model.save(model_name)
    with open(source_dir+os.sep+'history.json', 'w+') as fp: json.dump(history.history, fp)

    
    predict_results(source_dir, output_dir, model, images_val, masks_val, False )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train-source',dest='train_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--train-output',dest='train_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--ext',dest='ext', default='.TIF',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--epochs',dest='epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, clobber=args.clobber)
    

