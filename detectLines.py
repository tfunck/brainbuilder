import os
import json
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import h5py as h5
from re import sub
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import History, ModelCheckpoint
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


def dice(y_true, y_pred):
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return (2*overlap +1e-10) / (total + 1e-10)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def rotate_and_reflect(x,y):
    """ rotate and reflect random samples
    """
    #create vector to rotate
    to_reflect_ud=np.random.randint(2,size=x.shape[0] ).astype(bool)
    x[to_reflect_ud,:,:]=np.flipud(x[to_reflect_ud,:,:])
    y[to_reflect_ud,:,:]=np.flipud(y[to_reflect_ud,:,:])
    
    to_reflect_lr=np.random.randint(2,size=x.shape[0] ).astype(bool)
    x[to_reflect_lr,:,:]=np.fliplr(x[to_reflect_lr,:,:])
    y[to_reflect_lr,:,:]=np.fliplr(y[to_reflect_lr,:,:])
    
    #torotate=np.random.randint(2,size=x.shape[0] ).astype(bool)
    #x[torotate,:,:]=np.rot90(x[torotate,:,:],axes=(1,2))
    #y[torotate,:,:]=np.rot90(y[torotate,:,:],axes=(1,2))
    return x,y

    
def random_mask(x, dim=256, mask_scale=(0.,0.40) ):
    '''
        Remove random mask from x
    '''
    if mask_scale[0] < 0 or mask_scale[1] >= 1 or mask_scale[0] > mask_scale[1]  :
        print("Error mask_scale must be between 0 and 1")
        exit(1)

    for section in range(x.shape[0]) :
        #multiply image dimension <dim> by randomly generated values between mask_scale [0] and [1]
        mask_dim = np.array(dim * np.random.uniform(*mask_scale)).astype(int)

        #generate random location in image x. subtract dim by mask dim
        #to prevent mask from overlapping with the x image border
        i = np.random.randint(1,dim-mask_dim)
        j = np.random.randint(1,dim-mask_dim)
        x[section, i:(i+mask_dim),j:(j+mask_dim)] = 0
    return x

def gen(data,batch_size, idx, step=128):
    i=idx[0] 
    while True :
        x=data['x'][i:(i+batch_size)]
        y=data['y'][i:(i+batch_size)]
        dim = list(x.shape[0:3])+[1]
      
        rand_dim1 = np.random.randint(0, dim[1]-step, dim[0]).astype(int) 
        rand_dim2 = np.random.randint(0, dim[2]-step, dim[0]).astype(int)
        x_patch = np.zeros([dim[0], step, step, 1])
        y_patch = np.zeros([dim[0], step, step, 1])
        for i in range(x.shape[0]) :
            a=rand_dim1[i]
            b=rand_dim1[i]+step
            c=rand_dim2[i]
            d=rand_dim2[i]+step
            x_patch[i,:,:,0] = x[i, a:b, c:d ]
            y_patch[i,:,:,0] = y[i, a:b, c:d ]
        #    print(np.unique(y))
        #    plt.clf()
        #    plt.subplot(2,1,1)
        #    plt.imshow( x[j,:,:].reshape(624,833).astype(float) )
        #    plt.subplot(2,1,2)
        #    plt.imshow( y[j,:,:].reshape(624,833).astype(float), vmin=0, vmax=2 )
        #    plt.savefig('qc_'+str(i+j)+'.png')

        #print(' xmax', np.max(x), 'ymax', np.max(y))
        #x = x.reshape( dim )
        x,y=rotate_and_reflect(x,y ) 
        #x=random_mask(x)
        y_patch = to_categorical(y_patch.astype(np.uint16), 3).astype(np.uint16)
        #y = y.reshape( dim ).astype(np.float16)
        #y[ y >= 1 ] = 1.
        i = i + batch_size % idx.shape[0] 

        #for j in range(0,x.shape[0]):
        #    print(i+j, np.max(y), np.mean(y), np.sum(y), np.sum(np.isnan(y)))
        yield x_patch.astype(np.float32), y_patch.astype(np.float32)
        #yield y, y 
        #yield x, y 



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

def downsample_images(source_dir,output_dir, step=1, clobber=False):
    
    train_str=source_dir+'/train/*tif'
    label_str=source_dir+'/label/*tif'
    train_dir = output_dir +os.sep +"train"
    label_dir = output_dir +os.sep +"label"

    x_list  = [ f for f in glob(train_str)]
    y_list = [ f for f in glob(label_str) ]
    for fn in x_list :
        fn_rsl = train_dir + os.sep + sub('.tif','_downsample.png', os.path.basename(fn))
        if not os.path.exists(fn_rsl) or clobber :
            img = imageio.imread(fn)
            downsample( img, step=step,  interp=2, subject_fn=fn_rsl)

    for fn in y_list :
        fn_rsl = label_dir + os.sep + sub('.tif','_downsample.png', os.path.basename(fn))
        if not os.path.exists(fn_rsl) or clobber :
            img = safe_imread(fn)
            #print(fn_rsl)
            #print('a', np.unique(img))
            img[ img == 127.5 ] = 0
            img[ img == 255 ] = 128
            #print( np.unique(img))
            downsample( img, step=step,  interp=2, subject_fn=fn_rsl)


def pair_train_and_labels(output_dir, train_list, labels_list, step) :
    new_train_list=[]
    new_label_list=[]
    
    for i, f  in enumerate(train_list ):
        if i % 100 == 0 : print(1.*i/len(train_list))
        f2_root=os.path.splitext(os.path.basename(f))[0]
        f2 = [ f for f in glob(output_dir+'/label/' + f2_root + "*")  ]
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
    train_str=output_dir+'/train/*'
    label_str=output_dir+'/label/*'
    x_list  = [ f for f in glob(train_str+"downsample*")]
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
            #if 'QF#HG#MR1S1#L#rx82#9434#01' in y_fn :
            #    print('-->',y_fn, np.max(y), np.max(y)==0)

            if np.max(y) == 0 :
                print('Skip', y_fn)
            else :
                x_list_0.append(x_fn)
                y_list_0.append(y_fn)
        x_list = x_list_0
        y_list = y_list_0
        
        n=len(x_list)
        image = imageio.imread(x_list[0])
        ysize=image.shape[0]
        xsize=image.shape[1]
        data = h5.File(h5_fn, 'w')
        data.create_dataset("x", (n, ysize, xsize), dtype='float16')
        data.create_dataset("y", (n, ysize, xsize), dtype='uint16')
        
        for i, (x_fn, y_fn)  in enumerate(zip(x_list, y_list)) :
            x = safe_imread(x_fn) #load_image(x_fn, step, clobber=clobber, interp=3)
            data['x'][i,:,:] = x
            
            y=safe_imread(y_fn)
            y[ y <= 90] = 0
            y[ (y > 90 ) & (y<200)] = 1
            y[ y>=200 ] = 2
            data['y'][i,:,:]=y.astype(np.uint16) #load_image(y_fn, step, clobber=clobber, interp=0)
            #print(i,np.max(data['y'][i,:,:]),np.max(y), y_fn)
       
            #if not os.path.exists('qc_'+os.path.basename(x_fn)+'.png') :
                #plt.clf()
                #print('qc_'+os.path.basename(x_fn)+'.png')
                #plt.subplot(2,1,1)
                #plt.imshow( x )
                #plt.subplot(2,1,2)
                #plt.imshow( y , vmin=0, vmax=2)
                #plt.savefig('qc_'+os.path.basename(x_fn)+'.png')
     
from utils.utils import *

def make_compile_model(masks,class_weights,step=100) :
    image = Input(shape=( step, step, 1))
    IN = image #BatchNormalization()(image)

    DO=0.1
    N0=20
    N1=N0#*2
    N2=N1#*2
    N3=N2#*2
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
    CONV6_PAD = ZeroPadding2D( ((0,0),(0,0)) )(CONV6_UP)
    UP3 = Concatenate()([CONV6_PAD, CONV1])#, mode='concat', concat_axis=3)
    CONV7 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP3) #MERGE1)
    CONV7 = Dropout(DO)(CONV7)
    OUT = Conv2D(3, kernel_size=1,  padding='same', activation='softmax')(CONV7)

    model = Model(inputs=[image], outputs=OUT)
    #ada = keras.optimizers.Adam(0.0001)
    ada = tf.keras.optimizers.Adam()
    #model.compile(loss = 'categorical_crossentropy', optimizer=ada, metrics=[f1_m ] ) #f1_m
    model.compile(loss = weighted_categorical_crossentropy(class_weights) , optimizer=ada, metrics=[f1_m ] ) #f1_m

    return model

def add_padding(img, zmax, xmax):
    z=img.shape[0]
    x=img.shape[1]
    dz = zmax - z
    dx = xmax - x
    z_pad = pad_size(dz)
    x_pad = pad_size(dx)
    img_pad = np.pad(img, (z_pad,x_pad), 'minimum')
    return img_pad

def get_subset_dim(shape, i, patch_size):
    #print('a',shape, i, patch_size)
    #print('b',shape, i+patch_size)
    #print(i,end='-->')
    if shape < i +patch_size :
        i = shape - patch_size
        #print(i)
    else :
        pass
        #print()
    return i

def predict_results(source_dir, output_dir, model,patch_size, data, n_train, n_images, _use_radon ):
    if not os.path.exists(source_dir+os.sep+'results') : os.makedirs(source_dir+os.sep+'results')
    ydim=data['x'].shape[1]
    xdim=data['y'].shape[2]
    qc_dir = output_dir + os.sep + 'qc'
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

    for j, i in enumerate( range(n_train, n_images) ) :
        img=data['x'][i,:].reshape([1,ydim,xdim,1])
        seg=data['y'][i,:].reshape([1,ydim,xdim,1])
        
        itr=0
        X = np.zeros_like(img)
        N = np.zeros_like(img)
        stride = int(patch_size/2)
        for x in range(0, img.shape[1], stride):
            for y in range(0, img.shape[2], stride ):
                
                xi = get_subset_dim(img.shape[1], x, patch_size)
                yi = get_subset_dim(img.shape[2], y, patch_size)
                
                subset = img[0,xi:xi+patch_size, yi:yi+patch_size,0 ]
                
                subset = subset.reshape([1,patch_size,patch_size,1])
                xx = np.argmax( model.predict(subset, batch_size=1), axis=3)
                X[ 0,xi:xi+patch_size, yi:yi+patch_size ,0 ] +=  xx[0,:, : ]
                N[ 0,xi:xi+patch_size, yi:yi+patch_size ,0 ] +=  1
        X = X/ N
        plt.figure(figsize=(12,8), dpi=200, facecolor='b' ) 
        plt.subplot(1,3,1)
        plt.imshow( img.reshape(ydim,xdim).astype(float) )
        plt.subplot(1,3,2)
        plt.imshow(seg.reshape(ydim,xdim).astype(float), vmin=0, vmax=2  )
        plt.subplot(1,3,3)
        plt.imshow(X.reshape(ydim,xdim).astype(float),vmin=0,vmax=2) #.astype(int), vmin=0, vmax=2 )
        print(qc_dir+os.sep+str(i)+'.png')
        plt.tight_layout()
        plt.savefig(qc_dir+os.sep+str(i)+'.png', facecolor='black')
        plt.clf()
        if j > 15 : break

def get_class_weights(data, class_weights_fn, new_class_weights ):
    print('Calculating class weights')
    if not os.path.exists(class_weights_fn) or new_class_weights :
        y=np.array(data['y'][:]).flatten()
        class_weights = compute_class_weight("balanced", np.array([0,1,2]), y)
        
        class_weights_dict = {  "0":str(class_weights[0]), 
                                "1":str(class_weights[1]),
                                "2":str(class_weights[2]) }

        json.dump(class_weights_dict, open(class_weights_fn, 'w'))
    else :
        class_weights_str = json.load(open(class_weights_fn, 'r'))
        class_weights={ 0:float(class_weights_str["0"]), 
                        1:float(class_weights_str["1"]), 
                        2:float(class_weights_str["2"])} 

    print("Class Weights:", class_weights)
    return class_weights

def train_model(source_dir, output_dir, step, epochs, clobber) :
    train_dir=source_dir+os.sep+'train'
    label_dir=source_dir+os.sep+'labels'
    data_fn = output_dir +os.sep +'data.h5'
    

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    downsample_images(source_dir, output_dir, step=step, clobber=True)

    if not os.path.exists(source_dir+os.sep+'train.h5') or not os.path.exists(source_dir+os.sep+'labels.h5') or clobber  :
        fn_to_array(data_fn, output_dir, step, clobber=clobber)
   
    data = h5.File(data_fn,'r' )


    ratio=0.8
    n_images= data['x'].shape[0]
    n_train = int(round(ratio * n_images) )
    n_val = n_images - n_train
    all_idx = np.arange(n_images)
    np.random.shuffle( all_idx )
    train_idx = all_idx[0:n_train]
    val_idx = all_idx[n_train:n_images]

    batch_size=10
    print('N Images:', n_images, "N Train:", n_train, "N Val:", n_images - n_train )
    patch_size=256
    max_steps=int(np.floor(n_train/batch_size))
    val_steps=int(np.floor(n_val/batch_size))
    model_name=output_dir+os.sep+"model.hdf5"
    #checkpoint_fn = os.path.splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{f1_m:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(checkpoint_fn, monitor='loss', verbose=0, save_best_only=True, mode='min')

    steps=int(n_train/batch_size)
    val_steps=int((n_images-n_train)/batch_size)
    best_loss= np.inf
    if not os.path.exists(model_name) or clobber :
        #class_weights = get_class_weights(data, output_dir+os.sep+'class_weights.json', new_class_weights=False )
        #ar = np.array([ np.sum(data['y'][:] == 0),  np.sum(data['y'][:] == 1), np.sum(data['y'][:] == 2) ])
        #ar = ar/ np.product(data['y'].shape[0] * data['y'].shape[1] * data['y'].shape[2] )
        #ar = 1/ ar
        #class_weights_npy = np.array( list(class_weights.values()) )
        class_weights_npy = np.array([1,3,6])
        model = make_compile_model(data['x'],class_weights_npy , step=patch_size) 
        print('Fitting model')
        for epoch in range(epochs) :
            train_loss=0
            train_metric=0
            val_loss=0
            val_metric=0
            
            for step, (x, y) in enumerate(gen(data, batch_size, train_idx,step=patch_size)) :
                if step >= max_steps : break
                loss, metric = model.train_on_batch(x, y) 
                train_loss   += loss * 1/( max_steps)
                train_metric += metric * 1/( max_steps)
            
            for step, (x, y) in enumerate(gen(data, batch_size, val_idx,step=patch_size)) :
                if step >= val_steps : break
                loss, metric = model.evaluate(x, y,verbose=0)
                val_loss   += loss * 1/( val_steps)
                val_metric += metric * 1/( val_steps)
            
            sig_dig=5 
            print('Epoch:',epoch,'\tLoss:',round(train_loss,sig_dig),'\tMetric:', round(train_metric,sig_dig),end='')
            print('\tVal Loss:', round(val_loss,sig_dig) , '\tVal Metric:', round(val_metric,sig_dig) ) 
            if val_loss < best_loss :
                print('Saving model')
                model.save(model_name)
                best_loss = val_loss

        #history = model.fit_generator( 
        #        gen(data, batch_size, idx=train_idx), 
        #        steps_per_epoch=steps, 
        #        validation_data=gen(data, batch_size, idx=val_idx), 
        #        validation_steps=val_steps, 
        #        epochs=epochs, 
        #        #callbacks=[ checkpoint],
        #        verbose=1
        #        #class_weight=class_weights
        #        )
        #model.save(model_name)
    else :
        model = load_model(model_name, custom_objects={"dice":dice, "f1_m":f1_m})
    #print(history.history)
    #with open(output_dir+os.sep+'history.json', 'w+') as fp: json.dump(history.history, fp)
    predict_results(output_dir, output_dir, model, patch_size, data, n_train, n_images, False )

    return 0

def apply_model(train_output_dir, raw_file, lin_file, raw_output_dir, step, ext='.TIF', clobber=False):
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
    parser.add_argument('--ext',dest='ext', default='.TIF',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--epochs',dest='epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, clobber=args.clobber)
    

