import os
import argparse
import numpy as np
import keras
import imageio
import matplotlib.pyplot as plt
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
from utils import *
from glob import glob
from utils.utils import downsample
from keras import backend as K
from skimage.transform import rotate, resize 
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from utils.utils import downsample
from train_model import dice
from scipy.misc import imresize 
from scipy.ndimage.filters import gaussian_filter

def get_max_model(train_output_dir):
    model_list_str = train_output_dir+os.sep+"model_checkpoint-*hdf5"
    model_list = glob(model_list_str)
    if model_list == [] :
        print("Could not find any saved models in:", model_list_str )
        exit(1)
    models = [ os.path.basename(f).split('.')[-2] for f in model_list]
    max_model_idx = models.index(max(models))
    max_model = model_list[max_model_idx]
    return max_model

def get_raw_files(raw_source_dir, ext):
    raw_files_str = raw_source_dir +  os.sep + "*"+ext
    raw_files = glob(raw_files_str)
    if raw_files == [] :
        print("Could not find any files in: ", raw_files_str)
        exit(1)
    return(raw_files)

def downsample_raw(raw_files, output_dir, step, clobber) :
    downsample_files=[]

    dwn_dir=output_dir + os.sep + 'downsampled'
    if not os.path.exists(dwn_dir) : os.makedirs(dwn_dir)

    for f in raw_files :
        f2_split = os.path.splitext(os.path.basename(f))
        f2=dwn_dir +os.sep+f2_split[0]+'.png'
        if not os.path.exists(f2) or clobber :
            img = imageio.imread(f)
            if len(img.shape) == 3 : img = np.mean(img, axis=2)
            img = downsample(img, step=step, interp='cubic')
            imageio.imsave(f2,img)
        downsample_files += [f2]

    return downsample_files

def get_lines(downsample_files,raw_files, max_model,output_dir, clobber) :

    print("Using model located in:", max_model) 
   
    line_dir=output_dir + os.sep + 'lines'
    if not os.path.exists(line_dir) : os.makedirs(line_dir)
    line_files=[]
    img0 = imageio.imread(raw_files[0])
    if len(img0.shape) == 3 : img0 = np.mean(img0, axis=2)
    y0=img0.shape[0]
    x0=img0.shape[1]

    for f in downsample_files :
        line_fn_split = os.path.splitext(os.path.basename(f))
        line_fn=line_dir +os.sep+line_fn_split[0]+'.png'

        if not os.path.exists(line_fn) or clobber:

            img=imageio.imread(f)
            ydim=img.shape[0]
            xdim=img.shape[1]
            if ydim > xdim : 
                img = img.T
            try : #Load keras if has not been loaded yet 
                keras
            except NameError :
                from keras.models import load_model
                import keras.metrics
                keras.metrics.dice = dice
                model = load_model(max_model )

            # Apply neural network model to downsampled image
            img=img.reshape([1,img.shape[0],img.shape[1],1])
            X = model.predict(img, batch_size=1)
            idx = X > np.max(X) * 0.5
            X[ idx ]  = 1
            X[ ~idx ] = 0
            
            # Resample to full resolution
            X=X.reshape(ydim,xdim)
            X2 = imresize(X, (y0,x0), interp='nearest')
            X2=X2.reshape(y0,x0)
            
            # Save output image
            imageio.imsave(line_fn, X2)
        line_files += [line_fn]

    return line_files


def fill(iraw, iline, it=10):
    iraw_temp = np.copy(iraw)
    iline_dil = binary_dilation(iline, iterations=it).astype(int)
    
    iline_dil = iline_dil - iline
    #plt.subplot(2,1,1)
    #plt.imshow(iline_dil)

    border = iraw[iline_dil == 1]
    border = border.flatten()

    replacement_values = border[np.random.randint(0, len(border) , np.prod(iraw.shape))].reshape(iraw.shape)
    iraw_temp[ iline == 1 ] = replacement_values[ iline == 1 ] 
    iraw_temp = gaussian_filter(iraw_temp, 5)
    iraw[ iline==1 ] = iraw_temp[iline == 1]

    #plt.subplot(2,1,2)
    #plt.imshow(iraw)
    #plt.show()
    return iraw 

    


def fill0(iraw, iline) :

    m=np.max(iline)
    xx, yy = np.meshgrid(range(iraw.shape[1]), range(iraw.shape[0]))
    xx=xx.reshape(-1)
    yy=yy.reshape(-1)
    iraw_vtr = iraw.reshape(-1)
    iline_vtr = iline.reshape(-1)

    idx0=np.zeros(xx.shape).astype(bool)
    idx1=np.zeros(xx.shape).astype(bool)
    idx2=np.zeros(xx.shape).astype(bool)
    idx3=np.zeros(xx.shape).astype(bool)
    span=8
    while np.sum(iline) > 0 : 


        print(np.sum(iline))
        y0 = yy - span
        y0[y0 < 0] = 0

        x0 = xx - span
        x0[x0 < 0] = 0
        
        y1=yy+span
        y1[y1 >= iraw.shape[0]] = iraw.shape[0]-1
        
        x1=xx+span
        x1[x1 >= iraw.shape[1]] = iraw.shape[1]-1
        
        idx0[:]=False
        idx1[:]=False
        idx2[:]=False
        idx3[:]=False
        inside=iline[yy,xx] == m
        idx0[(iline[ y0, xx ]  < m ) & inside  ] =True
        idx1[(iline[ y1, xx ]  < m ) & inside ] =True
        idx2[(iline[ yy, x0 ]  < m ) & inside ] =True
        idx3[(iline[ yy, x1 ]  < m ) & inside ] =True

        n = idx0.astype(int) +  idx1.astype(int) + idx2.astype(int) + idx3.astype(int)
        
        i = n == 0
        
        iline[~i.reshape(iraw.shape)]=0
        
        n[~i]= n[~i].astype(float)
        n[i]=1
        iraw[yy[~i],xx[~i]]=0
        temp=np.copy(iraw)

        temp[yy[idx0],xx[idx0]] += iraw[y0[idx0], xx[idx0]]   
        temp[yy[idx1],xx[idx1]] += iraw[y1[idx1], xx[idx1]]   
        temp[yy[idx2],xx[idx2]] += iraw[yy[idx2], x0[idx2]]   
        temp[yy[idx3],xx[idx3]] += iraw[yy[idx3], x1[idx3]] 

        temp /= n.reshape(iraw.shape)
        iraw=temp
        span *= 2

    return iraw


from scipy.ndimage.morphology import binary_dilation, binary_erosion
def remove_lines(line_files, raw_files, raw_output_dir, clobber) :
    final_dir=raw_output_dir + os.sep + 'final'
    if not os.path.exists(final_dir) : os.makedirs(final_dir)

    for raw in raw_files :
        base =re.sub('#L', '', os.path.splitext(os.path.basename(raw))[0])

        fout = final_dir + os.sep + base + '.TIF'
        if not os.path.exists(fout) or clobber : 

            iraw = imageio.imread(raw)
            if len(iraw.shape) == 3 : iraw = np.mean(iraw, axis=2)
            if iraw.shape[0] > iraw.shape[1] : 
                iraw = iraw.T
            lines= [ f for f in line_files if base in f ]
            if lines != [] : lines=lines[0]
            else : 
                print("Failed at remove_lines for :", raw); 
                exit(1)
            iline = imageio.imread(lines)
            iline=binary_dilation(iline,iterations=3).astype(int)
            iraw=fill(iraw, iline)
            imageio.imsave(fout, iraw)

    return 0

def apply_model(train_output_dir, raw_source_dir, lin_source_dir, raw_output_dir, step, ext='.TIF', clobber=False):
    max_model=get_max_model(train_output_dir)
    raw_files=get_raw_files(raw_source_dir, '.tif')  
    lin_files=get_raw_files(lin_source_dir, ext)  
    #print("Got raw file names.")
    downsample_files = downsample_raw(raw_files, raw_output_dir, step, clobber)
    print("Got downsampled files.")
    line_files = get_lines(downsample_files, raw_files,max_model, raw_output_dir,  clobber)
    print("Loaded line files.")
    remove_lines(line_files, lin_files, raw_output_dir, clobber)
    print("Removed lines from raw files.")


    return 0
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


def load_image(fn, step, clobber, interp='cubic') :
    
    fn_split = os.path.splitext(fn)
    fn2 = fn_split[0] + "_downsample" + fn_split[1]

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


def fn_to_array(step,clobber=False, train_str='test/train/*tif', label_str='test/label/*tif') :
    train_list  = [ f for f in glob(train_str) if not 'downsample' in f ]
    labels_list = [ f for f in glob(label_str) if not 'downsample' in f ]

    n=len(train_list)
    images = np.array([])
    masks  = np.array([])
    for f, i  in zip(train_list, range(n)) :
        f2_root=os.path.splitext(os.path.basename(f))[0]
        f2 = [ f for f in glob('test/label/' + f2_root + "*") if not 'downsample' in f ]
        if f2 != [] : f2=f2[0]
        image=load_image(f, step, clobber)
        mask=load_image(f2, step, clobber, interp='nearest')
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
    CONV4_PAD = ZeroPadding2D( ((1,0),(1,0)) )(CONV4_UP)
    UP1 = merge([CONV4_PAD, CONV3], mode='concat', concat_axis=3)

    CONV5 = Conv2D( N2, kernel_size=[3,3],activation='relu',padding='same')(UP1)
    CONV5 = Dropout(DO)(CONV5)

    #LEVEL 2
    CONV5_UP = UpSampling2D(size=(2, 2))(CONV5)
    CONV5_PAD = ZeroPadding2D( ((0,0),(1,0)) )(CONV5_UP)
    UP2 = merge([CONV5_PAD, CONV2], mode='concat', concat_axis=3)
    CONV6 = Conv2D( N1, kernel_size=[3,3],activation='relu',padding='same')(UP2)
    CONV6 = Dropout(DO)(CONV6)

    #Level 1
    CONV6_UP = UpSampling2D(size=(2, 2))(CONV6)
    CONV6_PAD = ZeroPadding2D( ((1,0),(1,0)) )(CONV6_UP)
    UP3 = merge([CONV6_PAD, CONV1], mode='concat', concat_axis=3)
    CONV7 = Conv2D( N0, kernel_size=[3,3],activation='relu',padding='same')(UP3) #MERGE1)
    CONV7 = Dropout(DO)(CONV7)
    OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(CONV7)

    model = Model(inputs=[image], outputs=OUT)



    ada = keras.optimizers.Adam(0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer=ada,metrics=[dice] )
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
    #generate_models(source_dir, output_dir, step, epochs, clobber)
    
    deg_list = [0]
    stretch_list = [1]
    identity = lambda x : x
    equalize = lambda x : cv2.equalizeHist( x.astype(np.uint8) )
    h_list_0 = [ identity, equalize]
    h_list_1 = [ identity] * len(h_list_0)
    n_aug= len(deg_list) * len(stretch_list) * len(h_list_0)

    if not os.path.exists(source_dir+os.sep+'train.npy') or not os.path.exists(source_dir+os.sep+'labels.npy') or clobber  :
        images, masks = fn_to_array(step, clobber=clobber)
        print(images.shape)
        print(masks.shape)
        np.save(train_dir, images)
        np.save(label_dir, masks)
    else : 
        images = np.load(source_dir+os.sep+'train.npy')
        masks = np.load(source_dir+os.sep+'labels.npy')
    
    ratio=0.8
    n_images= images.shape[0] / n_aug
    n_train = int(round(ratio * n_images)*n_aug )
    
    images_train= images[:n_train,]
    images_val  = images[n_train:]

    masks_train = masks[:n_train,]
    masks_val   = masks[n_train:]

    n=images_val.shape[0]
    m=images_train.shape[0]
    batch_size_list = [  i for i in range(1,min(n,m)+1) if n % i == 0 and m % i == 0 ]
	
    model = make_compile_model(masks) 
    steps=int(np.floor(m/batch_size))
    val_steps=int(np.floor(n/batch_size))
    model_name=source_dir+os.sep+"model.hdf5"
    checkpoint_fn = os.path.splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{dice:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_dice', verbose=0, save_best_only=True, mode='max')

    batch_size = 16

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
                    'data/train',  # this is the target directory
                    target_size=(150, 150),  # all images will be resized to 150x150
                    batch_size=batch_size,
                    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
                    'data/validation',
                    target_size=(150, 150),
                    batch_size=batch_size,
                    class_mode='binary')


    history = model.fit_generator( gen(images_train, masks_train, batch_size), 
            steps_per_epoch=steps, 
            validation_data=gen(images_val, masks_val, batch_size ), 
            validation_steps=val_steps, 
            epochs=epochs, 
            callbacks=[ checkpoint]
            )

    model.save(model_name)
    with open(source_dir+os.sep+'history.json', 'w+') as fp: json.dump(history.history, fp)

    
    predict_results(source_dir, output_dir, model, images_val, masks_val, False )

    return 0


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Process some integers.')
#    parser.add_argument('--source',dest='source_dir', default='test/', help='Directory with raw images')
#    parser.add_argument('--output',dest='output_dir', default='test/results',  help='Directory name for outputs')
#    parser.add_argument('--step',dest='step', default=0.5, type=float, help='File extension for input files (default=.tif)')
#    parser.add_argument('--epochs',dest='epochs', default=1, type=int, help='Number of epochs')
#    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

#    args = parser.parse_args()
#    main(args.source_dir, args.output_dir, step=args.step, epochs=args.epochs, clobber=args.clobber)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train-source',dest='train_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--train-output',dest='train_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--raw-source',dest='raw_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--lin-source',dest='lin_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--raw-output',dest='raw_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--ext',dest='ext', default='.TIF',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--epochs',dest='epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if args.train_source_dir != '' and args.train_output_dir != '' :
        train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, clobber=args.clobber)
    else :
        print("Skipping train_model because either --train-source or --train-output is not set")
    
    if args.raw_source_dir != '' and args.raw_output_dir != '' and args.train_output_dir != '' :
        apply_model(args.train_output_dir,args.raw_source_dir,args.lin_source_dir,args.raw_output_dir,args.step,args.ext, args.clobber)
    else :
        print("Skipping apply_model because either --train-output, --raw-source, or --raw-output are not set")

