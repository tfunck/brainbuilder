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
    #print(fn2)
    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = np.mean(img, axis=2)
    if not os.path.exists(fn2) or clobber :
        #if interp == 'nearest' : 
             #plt.subplot(1,3,1)
             #plt.imshow(img)
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
    train_label_pair=[]
    for f  in train_list :
        #print("Train:\t", f)
        f2_root=os.path.splitext(os.path.basename(f))[0]
        #f2 =  glob('test/label/' + f2_root + "*")
        f2 = [ f for f in glob(source_dir+os.sep+'label/' + f2_root + "*") if not 'downsample' in f ]
        if f2 != [] : f2=f2[0]
        if not type(f) == str or not type(f2) == str : 
            print("Warning: skipping", f, f2)
            continue
        #print("Test:\t",f2)
        train_label_pair = train_label_pair + [(f, f2)]


    n=len(train_label_pair)
    images = []
    masks  = []
    print("Number of training/label pairs found:", n)
    i=0
    for f, f2 in train_label_pair :
        if i % 100 == 0 : print("Percent Complete:", round(100*0.+i/n,3))
        image=load_image(f, step, output_dir, clobber=clobber)
        mask=load_image(f2, step, output_dir, clobber=clobber, interp='nearest')
        #print(f)
        #print(f2)

        #plt.subplot(2,1,1)
        #plt.imshow(image.reshape(image.shape[0:2]))
        #plt.subplot(2,1,2)
        #plt.imshow(mask.reshape(mask.shape[0:2]))
        #plt.show()
        if images == [] :
            ysize=image.shape[0]
            xsize=image.shape[1]
            images = np.zeros((n,ysize,xsize,1))     
            masks  = np.zeros((n,ysize,xsize,1))     
        
        images[i] = image
        masks[i] = mask
        i+=1
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
    DO=0.2
    N0=16
    N1=N0*2
    N2=N1*2
    N3=N2*2
    N4=N3*2

    #LEVEL 1
    CONV1 = Conv2D( N0 , kernel_size=[3,3],activation='relu',padding='same')(IN)
    CONV1 = Dropout(DO)(CONV1)
    CONV1B = Conv2D( N0 , kernel_size=[3,3],activation='relu',padding='same')(CONV1)
    CONV1B = Dropout(DO)(CONV1B)
    POOL1 = MaxPooling2D(pool_size=(2, 2))(CONV1B)
    #LEVEL 2

    CONV2 = Conv2D( N1 , kernel_size=[3,3],activation='relu',padding='same')(POOL1)
    CONV2 = Dropout(DO)(CONV2)
    CONV2B = Conv2D( N1 , kernel_size=[3,3],activation='relu',padding='same')(CONV2)
    CONV2B = Dropout(DO)(CONV2B)
    POOL2 = MaxPooling2D(pool_size=(2, 2))(CONV2B)

    #LEVEL 3
    CONV3 = Conv2D( N2 , kernel_size=[3,3],activation='relu',padding='same')(POOL2)
    CONV3 = Dropout(DO)(CONV3) 
    CONV3B = Conv2D( N2 , kernel_size=[3,3],activation='relu',padding='same')(CONV3)
    CONV3B = Dropout(DO)(CONV3B)
    POOL3 = MaxPooling2D(pool_size=(2, 2))(CONV3B)

    #LEVEL 4
    CONV4 = Conv2D( N3 , kernel_size=[3,3],activation='relu',padding='same')(POOL3)
    CONV4 = Dropout(DO)(CONV4)
    CONV4B = Conv2D( N3 , kernel_size=[3,3],activation='relu',padding='same')(CONV4)
    CONV4B = Dropout(DO)(CONV4B)
    #POOL4 = MaxPooling2D(pool_size=(2, 2))(CONV4B)
    
    #LEVEL 5
    #CONV5 = Conv2D( N4 , kernel_size=[3,3],activation='relu',padding='same')(POOL4)
    #CONV5 = Dropout(DO)(CONV5)
    #CONV5B = Conv2D( N4 , kernel_size=[3,3],activation='relu',padding='same')(CONV5)
    #CONV5B = Dropout(DO)(CONV5B)

    #LEVEL 4
    #CONV6_UP = UpSampling2D(size=(2, 2))(CONV5B)
    #CONV6_PAD = ZeroPadding2D( ((1,0),(0,0)) )(CONV6_UP)
    #UP1 = merge([CONV6_PAD, CONV4B], mode='concat', concat_axis=3)

    #CONV6 = Conv2D( N3, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    #CONV6 = Dropout(DO)(CONV6)
    #CONV6B = Conv2D( N3, kernel_size=[3,3],activation='relu',padding='same')(CONV6)
    #CONV6B = Dropout(DO)(CONV6B)

    #LEVEL 3
    CONV7_UP = UpSampling2D(size=(2, 2))(CONV4B)
    #CONV7_UP = UpSampling2D(size=(2, 2))(CONV6B)
    CONV7_PAD = CONV7_UP # ZeroPadding2D( ((1,0),(1,0)) )(CONV4_UP)
    UP1 = merge([CONV7_PAD, CONV3B], mode='concat', concat_axis=3)

    CONV7 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    CONV7 = Dropout(DO)(CONV7)
    CONV7B = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(CONV7)
    CONV7B = Dropout(DO)(CONV7B)

    #LEVEL 2
    CONV8_UP = UpSampling2D(size=(2, 2))(CONV7B)
    CONV8_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV8_UP)

    UP2 = merge([CONV8_PAD, CONV2B], mode='concat', concat_axis=3)
    CONV8 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(UP2)
    CONV8 = Dropout(DO)(CONV8)
    CONV8B = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(CONV8)
    CONV8B = Dropout(DO)(CONV8B)

    #Level 1
    CONV8_UP = UpSampling2D(size=(2, 2))(CONV8B)
    CONV8_PAD = ZeroPadding2D( ((0,0),(1,0)) )(CONV8_UP)
    UP3 = merge([CONV8_PAD, CONV1B], mode='concat', concat_axis=3)
    CONV8 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP3) #MERGE1)
    CONV8 = Dropout(DO)(CONV8) 
    CONV8B = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(CONV8) #MERGE1)
    CONV8B = Dropout(DO)(CONV8B)

    OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(CONV8B)

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

def train_model(source_dir, output_dir, step, epochs, clobber,ratio=0.9 ) :
    if not os.path.exists(output_dir) : os.makedirs(output_dir)

    model_name=output_dir+os.sep+"model.hdf5"
    
    if not os.path.exists(model_name) or clobber : 
        deg_list = [0]
        stretch_list = [1]
        identity = lambda x : x
        equalize = lambda x : cv2.equalizeHist( x.astype(np.uint8) )
        h_list_0 = [ identity, equalize]
        h_list_1 = [ identity] * len(h_list_0)
        n_aug= len(deg_list) * len(stretch_list) * len(h_list_0)


        train_str=source_dir+os.sep+'train/*tif'
        label_str=source_dir+os.sep+'label/*tif'
        train_list  = [ f for f in glob(train_str) if not 'downsample' in f ]
        labels_list = [ f for f in glob(label_str) if not 'downsample' in f ]

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
        
        n_images= int(images.shape[0] / n_aug)
        n_train = int(round(ratio * n_images)*n_aug )
       
        train_set = np.random.choice(n_images, n_train)
        val_set = [ i for i in range(n_images) if not i in train_set ]

        images_train= images[train_set]
        images_val  = images[val_set]

        masks_train = masks[train_set]
        masks_val   = masks[val_set]

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
