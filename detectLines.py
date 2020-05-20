import os
import json
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import h5py as h5
import tensorflow as tf
import time
import shutil
from scipy.ndimage.morphology import distance_transform_cdt as cdt
from re import sub
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.keras import metrics
from tensorflow.keras import losses 
from utils import *
from glob import glob
from utils.utils import downsample
from keras import backend as K
from skimage.transform import resize 
from scipy.ndimage import rotate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
global NDIM
NDIM=2

def weighted_categorical_accuracy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred) :
        func = lambda y_true, y_pred : math_ops.reduce_mean( math_ops.multiply( math_ops.reduce_mean( math_ops.cast(math_ops.equal(y_true, y_pred) , K.floatx()), axis=(1,2)) , weights), axis=0 )
        num = func(y_true, y_pred)
        den = func(y_true, y_true)
        loss_var = math_ops.reduce_mean( math_ops.div( num, den) )
        return loss_var
    return loss

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



def reflect(x,y):
    """ rotate and reflect random samples
    """
    #create vector to rotate
    to_reflect_ud=np.random.randint(2,size=x.shape[0] ).astype(bool)
    x[to_reflect_ud,:,:]=np.flipud(x[to_reflect_ud,:,:])
    y[to_reflect_ud,:,:]=np.flipud(y[to_reflect_ud,:,:])
    
    to_reflect_lr=np.random.randint(2,size=x.shape[0] ).astype(bool)
    x[to_reflect_lr,:,:]=np.fliplr(x[to_reflect_lr,:,:])
    y[to_reflect_lr,:,:]=np.fliplr(y[to_reflect_lr,:,:])
   
    to_rotate=np.random.randint(2,size=x.shape[0] ).astype(bool)
    angles = np.random.uniform(0,360, x.shape[0])
    for i, rotate_bool in enumerate(to_rotate) :
        if rotate_bool :
            x[i,:,:]=rotate(x[i,:,:].astype(np.float32), angle=angles[i], reshape=False, cval=np.min(x[i]), order=1)
            y[i,:,:]=rotate(y[i,:,:], angle=angles[i], reshape=False, cval=np.min(x[i]), order=1)

    return x,y
    
def random_mask(x, mask_scale=(0.,0.30) ):
    '''
        Remove random mask from x
    '''
    if mask_scale[0] < 0 or mask_scale[1] >= 1 or mask_scale[0] > mask_scale[1]  :
        print("Error mask_scale must be between 0 and 1")
        exit(1)

    dim = np.array( x.shape[1:3] )
    for section in range(x.shape[0]) :
        #multiply image dimension <dim> by randomly generated values between mask_scale [0] and [1]
        mask_dim = np.array( dim * np.random.uniform(*mask_scale) ).astype(int)
        
        #generate random location in image x. subtract dim by mask dim
        #to prevent mask from overlapping with the x image border
        i = np.random.randint(1,dim[0]-mask_dim[0])
        j = np.random.randint(1,dim[1]-mask_dim[1])
        x[section, i:(i+mask_dim[0]),j:(j+mask_dim[1])] = 0
    return x

def create_synthetic_data(data, batch_size) :
    i0 = np.random.randint(0, data['x_clean'].shape[0] - batch_size )
    i1 = np.random.randint(0, data['y'].shape[0] - batch_size )
    
    x = data['x_clean'][i0:(i0+batch_size), :, :]
    for k in range(batch_size) :
        x[k] = (x[k] - np.min(x[k])) /  (np.max(x[k])-np.min(x[k]) )

    x_source = data['x'][i1:(i1+batch_size), :, :]
    y = data['y'][i1:(i1+batch_size), :,:]
    x_source, y = reflect(x_source, y)
    x_out = np.copy(x)
    x_out[ y > 0 ] = x_source[ y > 0]
    return x_out, y

def gen_qc(model, batch_size, x, y, epoch, i, train_qc_dir ) :
    z = model.predict( x, batch_size=batch_size )
    for k in range(batch_size) :
        plt.clf()
        plt.figure(figsize=(9,12))
        plt.subplot(3,1,1)
        #plt.title( os.path.basename(x_fn[k]) )
        plt.imshow( x[k,:,:].reshape(x.shape[1],x.shape[2]).astype(float) )

        plt.subplot(3,1,2)
        #plt.title( os.path.basename(y_fn[k]) )
        plt.imshow( np.argmax(y[k,:,:],axis=2).reshape(x.shape[1],x.shape[2]).astype(float), vmin=0, vmax=2 )
        
        plt.subplot(3,1,3)
        #plt.title( os.path.basename(y_fn[k]) )
        plt.imshow( np.argmax(z[k,:,:],axis=2).reshape(x.shape[1],x.shape[2]).astype(float), vmin=0, vmax=2 )

        plt.tight_layout()
        out_fn='%s/epoch_qc_%s_%s_%s.png' % (train_qc_dir, str(i), str(k), str(epoch))
        plt.savefig(out_fn,dpi=200)

def gen(data,batch_size, idx, mask=True, rotate=True, validate=False, qc=False):
    i=0 #idx[0] 
    X=data['x'][:]
    Y=data['y'][:]
    X_FN=data['x_fn'][:]
    Y_FN=data['y_fn'][:]

    if not validate : np.random.shuffle(idx)
    while True :
        ### Get batch data
        if i + batch_size < X.shape[0] or validate :
            cur_idx = idx[i:(i+batch_size)]
            x=X[ cur_idx ]
            y=Y[ cur_idx ]
            x_fn=X_FN[ cur_idx  ] 
            y_fn=Y_FN[ cur_idx  ] 
            #print('Reading original data', idx[i:(i+batch_size)])
        #elif not validate :
        #    #print('Generating Synthetic Data')
        #    x, y = create_synthetic_data(data, batch_size)
        #    x_fn = ['synthetic'] * batch_size
        #    y_fn = ['synthetic'] * batch_size
        #else :
        #    print('Error: index spills over data array')
        #    exit(1)

        ### Augment data
        if rotate :
            x, y = reflect(x, y) 
        if mask :
            x=random_mask(x)

        x = x.reshape(*x.shape, 1)
        y = to_categorical(y.astype(np.uint16), NDIM).astype(np.uint16)
        
        i += batch_size 
        yield x.astype(np.float32), y.astype(np.float32)



def downsample_image_subset(base_in_dir, base_out_dir, image_str, step=0.1, ext='tif', clobber=0 ) : 
    in_dir_str = base_in_dir+'/'+image_str+'/*'+ext
    out_dir = base_out_dir + '/' + image_str + '/'
    image_list  = [ f for f in glob(in_dir_str)]
    print('Downsample',in_dir_str, len(image_list))
    
    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)
    n=len(image_list)
    for i, fn in enumerate(image_list) :
        fn_rsl = out_dir + os.sep + os.path.splitext( os.path.basename(fn))[0]+'.png'

        if not os.path.exists(fn_rsl) or clobber >= 3 :
            try :
                img = safe_imread(fn)
                if img.shape != (312,416) :
                    img = resize(img, (312,416), order=2)
                    #Rescale to 0-255
                    img = (img - np.min(img)) / ( np.max(img) - np.min(img)) * 255
                
                imageio.imsave(fn_rsl,img.astype(np.uint8))
                print('\t',np.round(i/n,2),'Dowsampled:',fn_rsl, img.shape, np.max(img))
            except IndexError :
                print('Error: could not read file',fn)
                continue
    print('Downsampled', image_str)

def pair_train_and_labels(output_dir, train_list, labels_list, step) :
    new_train_list=[]
    new_label_list=[]
    
    for i, f  in enumerate(train_list ):
        if i % 100 == 0 : print(1.*i/len(train_list))
        f2_root=os.path.splitext(os.path.basename(f))[0]
        f2_root=re.sub('#L_' ,'' , re.sub('downsample', '',  re.sub('#_downsample', '', f2_root)))
        if f2_root[-1] == '#' : f2_root = f2_root[0:-1]
        if f2_root[-2:] == '#L' : f2_root = f2_root[0:-2]

        f2_str = output_dir+'/label/' + f2_root + "*"
        f2 = [ f for f in glob(f2_str)  ]
        if f2 != [] : 
            f2=f2[0]
            new_train_list.append( f)
            new_label_list.append( f2 )
        else : 
            print('coult not pair:',f, f2_str)

            continue

    return new_train_list, new_label_list


def fn_to_array(h5_fn, output_dir, step=0.1,clobber=0) :
    train_str=output_dir+'/train/*'
    label_str=output_dir+'/label/*'
    train_clean_str=output_dir+'/clean/*'
    exit_on_completion=False

    print(train_str)
    print(label_str)
    x_list  = [ f for f in glob(train_str+"*")]
    #x_clean_list = [ f for f in glob(train_clean_str+"*")]
    y_list = [ f for f in glob(label_str+"*") ]
    
    if len(y_list) == 0  :
        print('y_list empty ')
        exit(0)

    if  len(x_list) == 0 :
        print('x_list empty ')
        exit(0)

    if not os.path.exists(h5_fn) or clobber >= 2 :
        x_list, y_list = pair_train_and_labels(output_dir, x_list, y_list, step)
        if len(y_list) == 0  :
            print('y_list empty after pairing')
            exit(0)

        if  len(x_list) == 0 :
            print('x_list empty after pairing')
            exit(0)

        x_list_0=[]
        y_list_0=[]
        
        for x_fn, y_fn in zip(x_list, y_list):
            y = safe_imread(y_fn)
            y[ y <= 90] = 0
            y[ (y > 90 ) & (y<200)] = 1
            #Don't include labels == 2
            y[ y>=200 ] =  1 #2
            if np.max(y) == 0 :
                print('Skipping',y_fn)
                os.remove(y_fn)
                pass
            else :
                x_list_0.append(x_fn)
                y_list_0.append(y_fn)
        x_list = x_list_0
        y_list = y_list_0
        n=len(x_list)
        #n_clean = len(x_clean_list)
        image = imageio.imread(x_list[0])
        ysize=image.shape[0]
        xsize=image.shape[1]
        data = h5.File(h5_fn, 'w')
        dt = h5.string_dtype(encoding='ascii')
        data.create_dataset("x", (n, ysize, xsize), dtype='float16')
        #data.create_dataset("x_clean", (n_clean, ysize, xsize), dtype='float16')
        data.create_dataset("y", (n, ysize, xsize), dtype='uint16')
        data.create_dataset("x_fn", (n, ), dtype=dt)
        data.create_dataset("y_fn", (n, ), dtype=dt)
        
        print('Clean')
        #for i, fn in enumerate(x_clean_list) :
        #    x = safe_imread(fn) 
        #    data['x_clean'][i, :, : ] =  x

        for i, (x_fn, y_fn)  in enumerate(zip(x_list, y_list)) :
            x = safe_imread(x_fn) 
            if x.shape != data['x'].shape[1:3] : 
                x = x.T
            x = (x - np.min(x))/( np.max(x) - np.min(x) )
            data['x'][i,:,:] =  x 
            data['x_fn'][i] = x_fn
            
            y=safe_imread(y_fn)
            if y.shape != data['y'].shape[1:3] : 
                y = y.T
            y[ y <= 90] = 0
            y[ (y > 90 ) & (y<200)] = 1
            y[ y>=200 ] = 1 #2
            
            try :
                data['y'][i,:,:]=y.astype(np.uint16) 
                data['y_fn'][i] = y_fn
            except TypeError :
                print(y.shape)
                print('Could not save to array:',y_fn)
                os.remove(y_fn)
                exit_on_completion=True

        if exit_on_completion : exit(1)

     
from utils.utils import *

def make_compile_model(masks,class_weights,batch_size) :
    image = Input(shape=( masks.shape[1], masks.shape[2], 1))
    IN = image #BatchNormalization()(image)

    DO=0.2
    N0=16
    N1=N0*2
    N2=N1*2
    N3=N2*2
    N4=N3*2

    #LEVEL 1
    CONV1 = Conv2D( N0 , kernel_size=[3,3],activation='relu',padding='same')(IN)
    CONV1 = Conv2D( N0 , kernel_size=[3,3],activation='relu',padding='same')(CONV1)
    CONV1 = Dropout(DO)(CONV1)
    POOL1 = MaxPooling2D(pool_size=(2, 2))(CONV1)
    #LEVEL 2

    CONV2 = Conv2D( N1 , kernel_size=[3,3],activation='relu',padding='same')(POOL1)
    CONV2 = Conv2D( N1 , kernel_size=[3,3],activation='relu',padding='same')(CONV2)
    CONV2 = Dropout(DO)(CONV2)
    POOL2 = MaxPooling2D(pool_size=(2, 2))(CONV2)

    #LEVEL 3
    CONV3 = Conv2D( N2 , kernel_size=[3,3],activation='relu',padding='same')(POOL2)
    CONV3 = Conv2D( N2 , kernel_size=[3,3],activation='relu',padding='same')(CONV3)
    CONV3 = Dropout(DO)(CONV3)
    POOL3 = MaxPooling2D(pool_size=(2, 2))(CONV3)

    #LEVEL 4
    CONV4 = Conv2D( N3 , kernel_size=[3,3],activation='relu',padding='same')(POOL3)
    CONV4 = Conv2D( N3 , kernel_size=[3,3],activation='relu',padding='same')(CONV4)
    CONV4 = Dropout(DO)(CONV4)
    POOL4 = MaxPooling2D(pool_size=(2, 2))(CONV4)

    #LEVEL 5
    CONV5 = Conv2D( N4 , kernel_size=[3,3],activation='relu',padding='same')(POOL4)
    CONV5 = Conv2D( N4 , kernel_size=[3,3],activation='relu',padding='same')(CONV5)
    CONV5 = Dropout(DO)(CONV5)

    #LEVEL 4
    CONV6_UP = UpSampling2D(size=(2, 2))(CONV5)
    CONV6_PAD = ZeroPadding2D( ((1,0),(0,0)) )(CONV6_UP)
    UP1 = Concatenate()([CONV6_PAD, CONV4])#, mode='concat', concat_axis=3)
    CONV6 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    CONV6 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(CONV6)
    CONV6 = Dropout(DO)(CONV6)

    #LEVEL 3
    CONV7_UP = UpSampling2D(size=(2, 2))(CONV6)
    CONV7_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV7_UP)
    UP2 = Concatenate()([CONV7_PAD, CONV3])#, mode='concat', concat_axis=3)

    CONV7 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP2)
    CONV7 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(CONV7)
    CONV7 = Dropout(DO)(CONV7)

    #LEVEL 2
    CONV8_UP = UpSampling2D(size=(2, 2))(CONV7)
    CONV8_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV8_UP)
    UP3 = Concatenate()([CONV8_PAD, CONV2])#, mode='concat', concat_axis=3)
    CONV8 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(UP3)
    CONV8 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(CONV8)
    CONV8 = Dropout(DO)(CONV8)

    #Level 1
    CONV9_UP = UpSampling2D(size=(2, 2))(CONV8)
    CONV9_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV9_UP)
    UP4 = Concatenate()([CONV9_PAD, CONV1])
    CONV9 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP4) 
    CONV9 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(CONV9) 
    CONV9 = Dropout(DO)(CONV9)
    OUT = Conv2D(NDIM, kernel_size=1,  padding='same', activation='softmax', name='cls')(CONV9)

    model = Model(inputs=[image], outputs=[OUT])
    ada = tf.keras.optimizers.Adam()

    #model.compile(loss=weighted_categorical_crossentropy(class_weights),  optimizer=ada, metrics=['CategoricalAccuracy'] )
    model.compile(loss='categorical_crossentropy',  optimizer=ada, metrics=['CategoricalAccuracy'] )
    print(model.summary())
    return model

def run_model(model, data, batch_size, idx, max_steps, epoch, train_qc_dir, mask=True, rotate=True, validate=False, qc_epoch=[],qc_batch=[]):
    epoch_loss=0
    epoch_metric=0
    
    for step, (x, y) in enumerate(gen(data, batch_size, idx, validate=validate, mask=mask, rotate=rotate)) :
        #if np.max(x) > 1 or np.min(x) < 0 : 
        #    print('Error: incorrect range for training data',np.max(x), np.min(x) )

        if epoch in qc_epoch and step in qc_batch :  gen_qc(model, batch_size, x, y, epoch, step, train_qc_dir )

        if step >= max_steps : break

        if not validate:
            batch_loss, batch_metric  = model.train_on_batch(x, y) 
        else :
            batch_loss, batch_metric  = model.evaluate(x, y, verbose=0) 

        epoch_loss   += batch_loss * 1/( max_steps)
        epoch_metric += batch_metric * 1/( max_steps)

    return epoch_loss, epoch_metric

def fit_model(data, model, model_name,  epochs, class_weights_npy, output_dir, mask=True, rotate=True,  ratio=0.8, batch_size=10, samples_per_image=2 ) :
    train_qc_dir = output_dir + os.sep + 'train_qc'
    if not os.path.exists(train_qc_dir) : os.makedirs(train_qc_dir)

    n_images= data['x'].shape[0]
    n_train = int(round(ratio * n_images) )
    n_val = n_images - n_train
    all_idx = np.arange(n_images)
    np.random.shuffle( all_idx )
    train_idx = all_idx[0:n_train]
    val_idx = all_idx[n_train:n_images]

    print('N Images:', n_images, "N Train:", n_train, "N Val:", n_images - n_train )
    train_steps=int(np.floor( (samples_per_image * n_train ) /batch_size) )
    val_steps=int(np.floor(n_val/batch_size))
    model_fn=''
    best_loss= np.inf
    best_metric = 0
    overfit_epoch_limit=epochs
    overfit_check=0
    print('Fitting model')
    print('Train:', n_train,'Validate:',n_val)
    train_loss_list=[]
    val_loss_list=[]

    ###Iterate over epochs and train/validate
    for epoch in range(epochs) :
        # Train
        train_loss, train_metric = run_model(model, data, batch_size, train_idx, train_steps, epoch, train_qc_dir, qc_epoch=[], qc_batch=[], mask=mask, rotate=rotate)
        # Validate
        val_loss, val_metric = run_model(model, data, batch_size, val_idx, val_steps, epoch, train_qc_dir)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # Print result for current epoch
        sig_dig=5 
        print('Epoch:',epoch,'\tLoss:',round(train_loss,sig_dig),'\tMetric:', round(train_metric,sig_dig),end='')
        print('\tVal Loss:', round(val_loss,sig_dig) , '\tVal Metric:', round(val_metric,sig_dig) ) 

        # Save best model
        if val_loss < best_loss  :
            if os.path.exists(model_fn) : os.remove(model_fn)
            model_fn = model_name % (rotate, round(val_loss,3))
            model.save( model_fn )
            best_loss = val_loss
            print('Saving model',best_loss)

        if train_loss < val_loss :
            overfit_check += 1
        else :
            overfit_check=0

        if overfit_check >= overfit_epoch_limit : 
            print('Warning: More than ', overfit_epoch_limit, 'where training loss is greater than validation loss. breaking early from training.' )
            break
   
    ### Plot training and validation loss
    plt.figure()
    line1 = plt.plot(range(len(train_loss_list)), train_loss_list, c='r', label='train loss')
    line2 = plt.plot(range(len(val_loss_list)), val_loss_list, c='b', label='val loss')
    plt.legend()
    plt.savefig(output_dir+os.sep+'training_plot.png')

def predict_results(output_dir, model, data):
    qc_dir = output_dir + os.sep + 'qc'
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    ydim,xdim = data['x'].shape[1:3]
    for i in range(data['x'].shape[0]) :
        img=data['x'][i,:].reshape([1,ydim,xdim,1])
        img_fn = data['x_fn'][i].decode("utf-8")

        img = (img-np.min(img)) / (np.max(img) - np.min(img))
        X = np.argmax( model.predict(img, batch_size=1), axis=3)
        X = X.reshape(ydim,xdim)
        plt.clf()
        plt.figure(figsize=(12,8), dpi=200, facecolor='b' ) 
        plt.subplot(1,2,1)
        plt.imshow( img.reshape(ydim,xdim).astype(float) )
        plt.subplot(1,2,2)
        plt.imshow(X.astype(float),vmin=0,vmax=2) #.astype(int), vmin=0, vmax=2 )
        plt.tight_layout()
        print(qc_dir+os.sep+os.path.splitext(os.path.basename(img_fn))[0]+'_qc.png' )
        plt.savefig( qc_dir+os.sep+os.path.splitext(os.path.basename(img_fn))[0]+'_qc.png', facecolor='black')
        plt.clf()

        if i >30 : break


def train_model(source_dir, output_dir, step, epochs,mask=True, rotate=True, ext='tif',batch_size=10,ratio=0.8,samples_per_image=2, clobber=0) :
    data_fn=output_dir+os.sep+"data.h5"

    # Downsample images
    for img_str in ['label', 'clean', 'train'] :
        downsample_image_subset(source_dir, output_dir, img_str, step=step, ext=ext, clobber=clobber)  

    # Put downsampled images into an hdf5
    fn_to_array(data_fn, output_dir, step,  clobber=clobber)

    
    data = h5.File(data_fn,'r' )
    class_weights_npy = np.array([1,2,3])

    # Output model name
    model_name=output_dir+os.sep+"model_rot-%s_loss-%s.hdf5"
    print('Data:', data['x'][:].shape)
    print('Model Name:', model_name%(rotate,'*') )
    model_name_list = glob(model_name%(rotate,'*'))
    if len(model_name_list) == 0 or clobber >= 1 :
        # Create model
        model = make_compile_model(data['x'], class_weights_npy, batch_size) 
        # Fit model
        fit_model(data, model, model_name,  epochs, class_weights_npy, output_dir, mask=mask, rotate=rotate, ratio=ratio, batch_size=batch_size, samples_per_image=samples_per_image)
    else :
        model_fn = model_name_list[0]
        print(model_fn)
        model = load_model(model_fn, custom_objects={"loss":weighted_categorical_crossentropy(class_weights_npy)})
   
    # Apply model to new data
    predict_results( output_dir, model,  data )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train-source',dest='train_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--train-output',dest='train_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--ext',dest='ext', default='.tif',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--batch-size',dest='batch_size', default=10, type=int, help='Size of training batches')
    parser.add_argument('--ratio',dest='ratio', default=0.8, type=float, help='Ratio of data to use for training')
    parser.add_argument('--epochs',dest='epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--samples-per-image',dest='samples_per_image', default=1., type=float, help='Number of images to train on')
    parser.add_argument('--no-mask',dest='mask', action='store_false', default=True, help='Mask out regions during data augmentation')
    parser.add_argument('--no-rotate',dest='rotate', action='store_false', default=True, help='Use rotatations for data augmentation')
    parser.add_argument('--clobber', dest='clobber', type=int, default=0, help='Clobber results')

    args = parser.parse_args()
    train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, batch_size=args.batch_size, ratio=args.ratio, mask=args.mask, rotate=args.rotate,  samples_per_image=args.samples_per_image, ext=args.ext, clobber=args.clobber)
    

