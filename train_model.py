from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D
from keras.engine.topology import Input
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, merge, ZeroPadding2D
from keras.activations import relu
from keras.callbacks import History, ModelCheckpoint
import numpy as np
import keras
from utils import *
from glob import glob
from sys import exit
import imageio
from utils.utils import downsample
import os
from keras import backend as K
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize 
import argparse
import json

def gen(X,Y,batch_size=1):
    i=0
    while True :
        for i in range(0,X.shape[0],batch_size) :
            x=X[i:(i+batch_size)]
            y=Y[i:(i+batch_size)]
            i+= batch_size
            yield x, y 

def dice(y_true, y_pred):
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return (2*overlap +1e-10) / (total + 1e-10)

def load_image(fn, step, output_dir,clobber,  interp='cubic') :
    
    fn_split = os.path.splitext(os.path.basename(fn))
    fn2 = output_dir + os.sep + fn_split[0] + "_downsample" + fn_split[1]

    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = np.mean(img, axis=2)
    if not os.path.exists(fn2) or clobber :
        #if interp == 'nearest' : 
        #    print(fn2)
        #    plt.subplot(1,3,1)
        #    plt.imshow(img)
        img = downsample(img, step=step, interp='cubic')
        if interp == 'nearest' : 
            #plt.subplot(1,3,2)
            #plt.imshow(img)
            idx = img > np.max(img) * 0.05
            img[ idx ] = 1
            img[ ~idx ] = 0
            #print(np.sum(idx), np.sum(~idx))
            #if np.max(img) == 0 :
            #plt.subplot(1,3,3)
                #plt.imshow(img)
                #plt.show()
        imageio.imsave(fn2, img)
    else :
        img = imageio.imread(fn2)
        if len(img.shape) == 3 : img = np.mean(img, axis=2)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return(img)


def fn_to_array(source_dir, output_dir, step,clobber=False) :
    train_str=source_dir+os.sep+'train/*tif'
    label_str=source_dir+os.sep+'label/*tif'
    
    train_list  = [ f for f in glob(train_str) if not 'downsample' in f ]
    labels_list = [ f for f in glob(label_str) if not 'downsample' in f ]

    n=len(train_list)
    images = np.array([])
    masks  = np.array([])
    for f, i  in zip(train_list, range(n)) :
        #print(f)
        f2_root=os.path.splitext(os.path.basename(f))[0]
        #f2 =  glob('test/label/' + f2_root + "*")
        f2 = [ f for f in glob(source_dir+os.sep+'label/' + f2_root + "*") if not 'downsample' in f ]
        if f2 != [] : f2=f2[0]
        #print(f2)
        image=load_image(f, step, output_dir, clobber)
        mask=load_image(f2, step, output_dir, clobber, interp='nearest')
        if i == 0 :
            ysize=image.shape[0]
            xsize=image.shape[1]
            images = np.zeros((n,ysize,xsize,1))     
            masks  = np.zeros((n,ysize,xsize,1))     
        
        images[i] = image
        masks[i] = mask
    return images, masks

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
    N0=21
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
    CONV4_PAD = CONV4_UP # ZeroPadding2D( ((1,0),(1,0)) )(CONV4_UP)
    #CONV3 = ZeroPadding2D( ((1,0),(0,0)) )(CONV3)
    UP1 = merge([CONV4_PAD, CONV3], mode='concat', concat_axis=3)

    CONV5 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    CONV5 = Dropout(DO)(CONV5)

    #LEVEL 2
    CONV5_UP = UpSampling2D(size=(2, 2))(CONV5)
    CONV5_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV5_UP)

    UP2 = merge([CONV5_PAD, CONV2], mode='concat', concat_axis=3)
    CONV6 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(UP2)
    CONV6 = Dropout(DO)(CONV6)

    #Level 1
    CONV6_UP = UpSampling2D(size=(2, 2))(CONV6)
    CONV6_PAD = ZeroPadding2D( ((0,0),(1,0)) )(CONV6_UP)
    UP3 = merge([CONV6_PAD, CONV1], mode='concat', concat_axis=3)
    CONV7 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP3) #MERGE1)
    CONV7 = Dropout(DO)(CONV7)
    OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(CONV7)

    model = Model(inputs=[image], outputs=OUT)



    ada = keras.optimizers.Adam(0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer=ada,metrics=[dice] )
    print(model.summary())
    return model


def predict_results(output_dir, model, images_val, masks_val, _use_radon ):
    if not os.path.exists(output_dir+os.sep+'results') : os.makedirs(output_dir+os.sep+'results')
    ydim=images_val.shape[1]
    xdim=images_val.shape[2]
    for i in range(images_val.shape[0]) :
        img=images_val[i,:].reshape([1,ydim,xdim,1])
        X = model.predict(img, batch_size=1)
   
        if _use_radon : X = radon_tfm(X, False)

        plt.subplot(1,3,1)
        plt.imshow(images_val[i].reshape(ydim,xdim) )
        plt.subplot(1,3,2)
        plt.imshow(masks_val[i].reshape(ydim,xdim))
        plt.subplot(1,3,3)
        plt.imshow(X.reshape(ydim,xdim))
        plt.savefig(output_dir+os.sep+str(i)+'.tif')

def train_model(source_dir, output_dir, step, epochs, clobber,ratio=0.8 ) :
    model_name=output_dir+os.sep+"model.hdf5"
    
    if not os.path.exists(model_name) or clobber : 
        deg_list = [0]
        stretch_list = [1]
        identity = lambda x : x
        equalize = lambda x : cv2.equalizeHist( x.astype(np.uint8) )
        h_list_0 = [ identity, equalize]
        h_list_1 = [ identity] * len(h_list_0)
        n_aug= len(deg_list) * len(stretch_list) * len(h_list_0)

        if not os.path.exists(output_dir+os.sep+'train.npy') or not os.path.exists(output_dir+os.sep+'labels.npy') or clobber  :
            images, masks = fn_to_array(source_dir, output_dir, step, clobber=clobber)
            print(images.shape)
            print(masks.shape)
            #images = augment(images, deg_list, stretch_list, h_list_0)
            #masks = augment(masks, deg_list, stretch_list, h_list_1, True)
            np.save(output_dir+os.sep+'train', images)
            np.save(output_dir+os.sep+'labels', masks)
        else : 
            images = np.load(output_dir+os.sep+'train.npy')
            masks = np.load(output_dir+os.sep+'labels.npy')
        
        n_images= images.shape[0] / n_aug
        n_train = int(round(ratio * n_images)*n_aug )
        
        images_train= images[:n_train,]
        images_val  = images[n_train:]

        masks_train = masks[:n_train,]
        masks_val   = masks[n_train:]

        n=images_val.shape[0]
        m=images_train.shape[0]
        #batch_size_list = [  i for i in range(1,min(n,m)+1) if n % i == 0 and m % i == 0 ]
        batch_size =1 #batch_size_list[ int(len(batch_size_list)/3) ] 
        
        model = make_compile_model(masks) 
        steps=int(np.floor(m/batch_size))
        val_steps=int(np.floor(n/batch_size))

        checkpoint_fn = os.path.splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{val_dice:.4f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_dice', verbose=0, save_best_only=True, mode='max')
        history = model.fit_generator( gen(images_train, masks_train, batch_size), 
                steps_per_epoch=steps, 
                validation_data=gen(images_val, masks_val, batch_size ), 
                validation_steps=val_steps, 
                epochs=epochs, 
                callbacks=[ checkpoint]
                )


        model.save(model_name)
        with open(output_dir+os.sep+'history.json','w+') as fp: json.dump(history.history, fp)

        predict_results(output_dir, model, images_val, masks_val, False )

    return 0
