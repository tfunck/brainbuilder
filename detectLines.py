import os
import json
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import h5py as h5
import tensorflow as tf
import time
from scipy.ndimage.morphology import distance_transform_cdt as cdt
from re import sub
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage import exposure
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
from skimage.transform import rotate, resize 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K

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
    
    return x,y
    
def random_mask(x, mask_scale=(0.,0.20) ):
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

def gen(data,batch_size, idx, validate=False, qc=False):
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
        elif not validate :
            #print('Generating Synthetic Data')
            x, y = create_synthetic_data(data, batch_size)
            x_fn = ['synthetic'] * batch_size
            y_fn = ['synthetic'] * batch_size
        else :
            print('Error: index spills over data array')
            exit(1)

        ### Augment data
        if not validate :
            x, y = reflect(x, y) 
            x=random_mask(x)

        x = x.reshape(*x.shape, 1)
        y = to_categorical(y.astype(np.uint16), 3).astype(np.uint16)
        
        i += batch_size 
        yield x.astype(np.float32), y.astype(np.float32)



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

def downsample_image_subset(base_in_dir, base_out_dir, image_str, step=0.1, ext='.tif', clobber=False ) : 
    in_dir_str = base_in_dir+'/'+image_str+'/*'+ext
    out_dir = base_out_dir + '/' + image_str + '/'
    print(in_dir_str)
    image_list  = [ f for f in glob(in_dir_str)]
    
    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)

    for fn in image_list :
        fn_rsl = out_dir + os.sep + sub('.'+ext,'_downsample.png', os.path.basename(fn))
        if not os.path.exists(fn_rsl) or clobber :
            img = safe_imread(fn)
            downsample( img, step=step,  interp=2, subject_fn=fn_rsl)
    print('Downsampled', image_str)

def pair_train_and_labels(output_dir, train_list, labels_list, step) :
    new_train_list=[]
    new_label_list=[]
    
    for i, f  in enumerate(train_list ):
        if i % 100 == 0 : print(1.*i/len(train_list))
        f2_root=os.path.splitext(os.path.basename(f))[0]
        f2_root=re.sub('#L' ,'' , re.sub('_downsample', '',  re.sub('#_downsample', '', f2_root)))
        f2_str = output_dir+'/label/' + f2_root + "*"
        f2 = [ f for f in glob(f2_str)  ]
        if f2 != [] : 
            f2=f2[0]
            new_train_list.append( f)
            new_label_list.append( f2 )
        else : 
            continue

    return new_train_list, new_label_list

def safe_imread(fn) :
    img = imageio.imread(fn)
    if len(img.shape) > 2 :
        img = np.mean(img,axis=2)
    return img


def fn_to_array(h5_fn, output_dir, step=0.1,clobber=False) :
    train_no_label_str = output_dir + '/train_no_labels/*'
    train_str=output_dir+'/train/*'
    label_str=output_dir+'/label/*'
    train_clean_str=output_dir+'/clean/*'

    x_list  = [ f for f in glob(train_str+"downsample*")]
    x_no_label_list  = [ f for f in glob(train_no_label_str+"downsample*")]
    x_clean_list = [ f for f in glob(train_clean_str+"downsample*")]
    y_list = [ f for f in glob(label_str+"downsample*") ]
    
    if not os.path.exists(h5_fn) or clobber :
        x_list, y_list = pair_train_and_labels(output_dir, x_list, y_list, step)
        
        x_list_0=[]
        y_list_0=[]
        for x_fn, y_fn in zip(x_list, y_list):
            y = safe_imread(y_fn)
            y[ y <= 90] = 0
            y[ (y > 90 ) & (y<200)] = 1
            y[ y>=200 ] = 2
            if np.max(y) == 0 :
                pass
            else :
                x_list_0.append(x_fn)
                y_list_0.append(y_fn)
        x_list = x_list_0
        y_list = y_list_0
        n=len(x_list)
        n_clean = len(x_clean_list)
        n_no_label = len(x_no_label_list)
        image = imageio.imread(x_list[0])
        ysize=image.shape[0]
        xsize=image.shape[1]
        data = h5.File(h5_fn, 'w')
        dt = h5.string_dtype(encoding='ascii')
        data.create_dataset("x", (n, ysize, xsize), dtype='float16')
        data.create_dataset("x_fn", (n, ), dtype=dt)
        data.create_dataset("x_clean", (n_clean, ysize, xsize), dtype='float16')
        data.create_dataset("x_no_label", (n_no_label, ysize, xsize), dtype='float16')
        data.create_dataset("y", (n, ysize, xsize), dtype='uint16')
        data.create_dataset("y_fn", (n, ), dtype=dt)
        
        print('Clean')
        for i, fn in enumerate(x_clean_list) :
            data['x_clean'][i, :, : ] =  safe_imread(fn)

        print('No Lines')
        for i, fn  in enumerate(x_no_label_list) :
            imageio.imread(fn)
            data['x_no_label'][i, :, : ] =  safe_imread(fn)

        for i, (x_fn, y_fn)  in enumerate(zip(x_list, y_list)) :
            x = safe_imread(x_fn) #load_image(x_fn, step, clobber=clobber, interp=3)
            x = (x - np.min(x))/( np.max(x) - np.min(x) )
            data['x'][i,:,:] = x
            data['x_fn'][i] = x_fn
            
            y=safe_imread(y_fn)
            y[ y <= 90] = 0
            y[ (y > 90 ) & (y<200)] = 1
            y[ y>=200 ] = 2
            data['y'][i,:,:]=y.astype(np.uint16) #load_image(y_fn, step, clobber=clobber, interp=0)
            data['y_fn'][i] = y_fn
            #print(i,np.max(data['y'][i,:,:]),np.max(y), y_fn)
       
            #if not os.path.exists('qc_'+os.path.basename(x_fn)) :
            #plt.clf()
            #print('qc_'+os.path.basename(x_fn)+'.png')
            #plt.subplot(3,1,1)
            #plt.imshow( data['x'][i,:,:] )
            #plt.subplot(3,1,2)
            #plt.imshow( data['y'][i,:,:] , vmin=0, vmax=2)
            #plt.savefig('qc_'+os.path.basename(x_fn)+'.png', dpi=200)
     
from utils.utils import *

def make_compile_model(masks,class_weights,batch_size) :
    image = Input(shape=( masks.shape[1], masks.shape[2], 1))
    IN = image #BatchNormalization()(image)

    DO=0.25
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
    CONV6_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV6_UP)
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
    CONV9_PAD = ZeroPadding2D( ((0,0),(0,1)) )(CONV9_UP)
    UP4 = Concatenate()([CONV9_PAD, CONV1])
    CONV9 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP4) 
    CONV9 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(CONV9) 
    CONV9 = Dropout(DO)(CONV9)
    OUT = Conv2D(3, kernel_size=1,  padding='same', activation='softmax', name='cls')(CONV9)

    model = Model(inputs=[image], outputs=[OUT])
    ada = tf.keras.optimizers.Adam()

    #model.compile(loss=weighted_categorical_crossentropy(class_weights),  optimizer=ada, metrics=['CategoricalAccuracy'] )
    model.compile(loss='categorical_crossentropy',  optimizer=ada, metrics=['CategoricalAccuracy'] )
    print(model.summary())
    return model

def run_model(model, data, batch_size, idx, max_steps, epoch, train_qc_dir, validate=False, qc_epoch=[],qc_batch=[]):
    epoch_loss=0
    epoch_metric=0
    
    for step, (x, y) in enumerate(gen(data, batch_size, idx, validate=validate)) :
        if epoch in qc_epoch and step in qc_batch :  gen_qc(model, batch_size, x, y, epoch, step, train_qc_dir )

        if step >= max_steps : break

        if not validate:
            batch_loss, batch_metric  = model.train_on_batch(x, y) 
        else :
            batch_loss, batch_metric  = model.evaluate(x, y, verbose=0) 

        epoch_loss   += batch_loss * 1/( max_steps)
        epoch_metric += batch_metric * 1/( max_steps)

    return epoch_loss, epoch_metric

def fit_model(data, model, model_name,  epochs, class_weights_npy, output_dir, ratio=0.8, batch_size=10 ) :
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
    samples_per_image=1.5
    train_steps=int(np.floor( (samples_per_image * n_train ) /batch_size) )
    val_steps=int(np.floor(n_val/batch_size))

    best_loss= np.inf
    best_metric = 0
    print('Fitting model')
    print('Train:', n_train,'Validate:',n_val)
    train_loss_list=[]
    val_loss_list=[]

    ###Iterate over epochs and train/validate
    for epoch in range(epochs) :
        # Train
        train_loss, train_metric = run_model(model, data, batch_size, train_idx, train_steps, epoch, train_qc_dir, qc_epoch=[], qc_batch=[])
        # Validate
        val_loss, val_metric = run_model(model, data, batch_size, val_idx, val_steps, epoch, train_qc_dir, validate=True)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # Print result for current epoch
        sig_dig=5 
        print('Epoch:',epoch,'\tLoss:',round(train_loss,sig_dig),'\tMetric:', round(train_metric,sig_dig),end='')
        print('\tVal Loss:', round(val_loss,sig_dig) , '\tVal Metric:', round(val_metric,sig_dig) ) 

        # Save best model
        if val_loss < best_loss :
            print('Saving model')
            model.save(model_name)
            best_loss = val_loss
   
    ### Plot training and validation loss
    plt.figure()
    line1 = plt.plot(range(epochs), train_loss_list, c='r')
    line2 = plt.plot(range(epochs), val_loss_list, c='b')
    plt.legend((line1,line2),('Train Loss', 'Val Loss'))
    plt.savefig(output_dir+os.sep+'training_plot.png')

def predict_results(output_dir, model, data):
    ydim=data['x_no_label'].shape[1]
    xdim=data['x_no_label'].shape[2]
    qc_dir = output_dir + os.sep + 'qc'
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

    for i in range(data['x_no_label'].shape[0]) :
        img=data['x_no_label'][i,:].reshape([1,ydim,xdim,1])
        img = (img-np.min(img)) / (np.max(img) - np.min(img))
        X = np.argmax( model.predict(img, batch_size=1), axis=3)
        plt.figure(figsize=(12,8), dpi=200, facecolor='b' ) 
        plt.subplot(1,2,1)
        plt.imshow( img.reshape(ydim,xdim).astype(float) )
        plt.subplot(1,2,2)
        plt.imshow(X.reshape(ydim,xdim).astype(float),vmin=0,vmax=2) #.astype(int), vmin=0, vmax=2 )
        print(qc_dir+os.sep+str(i)+'.png')
        plt.tight_layout()
        plt.savefig(qc_dir+os.sep+str(i)+'.png', facecolor='black')
        plt.clf()
        if i > 30 : break


def train_model(source_dir, output_dir, step, epochs, ext='tif',batch_size=10,ratio=0.8, clobber=False) :
    data_fn=output_dir+os.sep+"data.h5"

    # Downsample images
    for img_str in ['label', 'clean', 'train', 'train_no_labels'] :
        downsample_image_subset(source_dir, output_dir, img_str, step=step, ext=ext, clobber=clobber)  

    # Put downsampled images into an hdf5
    if not os.path.exists(source_dir+os.sep+'train.h5') or not os.path.exists(source_dir+os.sep+'labels.h5') or clobber:
        fn_to_array(data_fn, output_dir, step, clobber=clobber)

    
    data = h5.File(data_fn,'r' )
    class_weights_npy = np.array([1,1,1])

    # Output model name
    model_name=output_dir+os.sep+"model.hdf5"

    if not os.path.exists(model_name) or clobber :
        # Create model
        model = make_compile_model(data['x'], class_weights_npy, batch_size) 
        # Fit model
        fit_model(data, model, model_name,  epochs, class_weights_npy, output_dir, ratio=ratio, batch_size=batch_size)
    else :
        print(model_name)
        model = load_model(model_name, custom_objects={"loss":weighted_categorical_crossentropy(class_weights_npy)})
   
    # Apply model to new data
    predict_results( output_dir, model,  data )

    return 0

def apply_model(train_output_dir, raw_file, lin_file, raw_output_dir, step, ext='.tif', clobber=False):
    max_model=get_max_model(train_output_dir)
    
    #print("Got raw file names.")
    downsample_file = downsample_raw([raw_file], raw_output_dir, step, clobber)
    print("Got downsampled files.")
    line_files = get_lines(downsample_files, raw_files,max_model, raw_output_dir,  clobber)
    print("Loaded line files.")
    #remove_lines(line_files, lin_files, raw_output_dir, clobber)
    remove_lines(line_files, raw_files, raw_output_dir, clobber)
    print("Removed lines from raw files.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train-source',dest='train_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--train-output',dest='train_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--ext',dest='ext', default='.tif',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--batch-size',dest='batch_size', default=10, type=float, help='Size of training batches')
    parser.add_argument('--ratio',dest='ratio', default=0.8, type=float, help='Ratio of data to use for training')
    parser.add_argument('--epochs',dest='epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, batch_size=args.batch_size, ratio=args.ratio, ext=args.ext, clobber=args.clobber)
    

