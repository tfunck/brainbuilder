import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import keras
import keras.backend as K
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from imageio import imread, imwrite
from glob import glob
from skimage.transform import rescale
from skimage.exposure import  equalize_hist
from keras.layers.convolutional import Conv1D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
from keras.models import Model
from keras.layers import BatchNormalization, ZeroPadding2D, Concatenate
from keras.layers.core import Dropout, Dense, Flatten
from keras.layers import LeakyReLU, MaxPooling2D, concatenate
from keras.activations import relu
from keras.callbacks import History, ModelCheckpoint
from keras.models import load_model

def gen_rand_locations(dim,step,n_samples):
    values=np.arange(0,dim,step,dtype=int)
    np.random.shuffle(values)
    return values[0:n_samples]

def identifyBrainTissue(source_dir,output_dir,n_samples=10, xstep=300, zstep=300, total=3000, clobber=False, model=None) :
    qc_dir=output_dir + os.sep + 'qc/'
    if not os.path.exists(qc_dir) :
        os.makedirs(qc_dir)
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    source_files=glob(source_dir+"/*")
    
    #for t in ["oxot", "epib", "ampa", "uk14", "mk80", "pire" ] :
    #    source_files += glob(source_dir+"/*"+t+"*")
    np.random.shuffle(source_files)
    idx=0
    for f in source_files:
        im = imread(f)
    
        z_values = gen_rand_locations(im.shape[1],zstep,n_samples)
        x_values = gen_rand_locations(im.shape[0],xstep,n_samples)

        for x, z in zip(x_values, z_values) :
            splitf = os.path.splitext(os.path.basename(f))
            out_fn = output_dir + os.sep + splitf[0] +'_'+str(x)+'_'+str(z)+splitf[1]
            qc_fn = qc_dir + os.sep + splitf[0] +'_'+str(x)+'_'+str(z)+'.png'

            if not os.path.exists(out_fn) or clobber :
                im2=np.copy(im)
                if True in [ True for i in  ["oxot", "epib", "ampa", "uk14", "mk80", "pire" ] if i in f ] :
                    im2 = equalize_hist(im2)

                im2[x:(x+xstep), z:(z+zstep)] = im2[x:(x+xstep), z:(z+zstep)] + 1.5 * np.std(im2)
                imsubset = im[x:(x+xstep), z:(z+zstep) ]
                if imsubset.shape != (xstep, zstep) : continue

                #If a model is provided, apply it to the image subset to determine probability of tissue
                # and determine whether or not image should be saved
                write_img=True
                if model != None :
                    tissue_prob = model.predict(imsubset.reshape(1,*imsubset.shape,1), batch_size=1)[0][0]
                    
                    out_fn = output_dir + os.sep + splitf[0] +'_'+str(x)+'_'+str(z)+'_'+str(round(tissue_prob,2))+splitf[1]
                    qc_fn = qc_dir + os.sep + splitf[0] +'_'+str(x)+'_'+str(z)+'_'+str(round(tissue_prob,2))+'.png'
                    if tissue_prob > 0.6 or tissue_prob < 0.4 : 
                        write_img=False


                if imsubset.shape == (xstep,zstep) and write_img :
                    imwrite(out_fn, imsubset)
                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.imshow(im2)
                    plt.subplot(1,2,2)
                    plt.imshow(imsubset)
                    plt.savefig(qc_fn)
                    idx += 1
                    print(idx, os.path.basename(f) ) 
                    if idx >= total : return 0
    
import shutil
import re
def replace_qc_img(unsorted_dir, tissue_dir, not_tissue_dir):
    tissue_files = glob(tissue_dir+"/*png")
    not_tissue_files = glob(not_tissue_dir+"/*png")

    for f in tissue_files + not_tissue_files :
        #Name of tissue image that corresponds to qc image, f
        patch_fn = glob(unsorted_dir+"/"+os.path.splitext(os.path.basename(f))[0]+".TIF" )
        #If there is no TIF file in the directory of unsorted files, skip this qc image, f
        if len(patch_fn) > 0 :
            patch_fn=patch_fn[0]
        else :
            continue
        #Set output file name
        target_fn = re.sub('.png', '.TIF', f)
        shutil.copy(patch_fn, target_fn)
        shutil.remove(patch_fn)
        #print(patch_fn, target_fn)


    return 0

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
def get_batch_size(train_x, val_x, limit=10) :
    n=val_x.shape[0]
    m=train_x.shape[0]
    batch_size_list = np.array([  i for i in range(1,min(n,m)+1) if n % i == 0 and m % i == 0 ])
    batch_size = max(batch_size_list[batch_size_list < limit])
    steps=int(np.floor(m/batch_size))
    val_steps=int(np.floor(n/batch_size))
    return batch_size, steps, val_steps

def get_image_filenames(tissue_dir, not_tissue_dir):
    ###Load Images
    tissue=glob(tissue_dir+"/*.TIF")
    tissue_label=[1.]*len(tissue)

    not_tissue=glob(not_tissue_dir+'/*.TIF')
    not_tissue_label=[0.]*len(not_tissue)

    x = np.array(tissue + not_tissue)
    y = np.array(tissue_label + not_tissue_label)

    idx = np.random.choice(range(len(x)), len(x), replace=False)
    x = x[idx]
    y = y[idx]
    return x, y

def read_format_images(start_idx, end_idx, xstep, zstep, x,y  ):
    train_x=np.zeros([end_idx-start_idx,xstep,zstep])
    for i, f in enumerate(x[start_idx:end_idx]):
        img= imread(f)
        #img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if img.shape != (300,300) : continue
        if y[i] == 1 : 
            pass
            #img[100:200,100:200]=np.ones((100,100))
            #img = np.ones((300,300))
        else : 
            pass 
            #img = np.zeros((300,300))
            #img *= 0
        train_x[i,:,:] = img
    return train_x

def create_train_val_data(x, y, train_ratio, val_ratio, xstep=300, zstep=300):
    ### Create Training and Validation sets
    train_idx = int(len(x) * train_ratio)
    val_idx = int(train_idx + len(x) * val_ratio)
    print(train_idx, val_idx, len(x))

    train_x = read_format_images(0, train_idx, xstep, zstep, x,y  )
    train_y = np.array(y[0:train_idx])    
   

    val_x = read_format_images(train_idx, val_idx, xstep, zstep, x ,y  )
    val_y = np.array(y[train_idx:val_idx])    

    return train_x, train_y, val_x, val_y

def gen(X,Y,batch_size=1,xdim=300,zdim=300):
    while True :
        for i in range(0,X.shape[0],batch_size) :
            x=X[i:(i+batch_size)]
            y=Y[i:(i+batch_size)]
            #plt.clf()
            #plt.imshow(x[0,:,:])
            #plt.title(str(y[0]))
            #plt.show()
            x=x.reshape( [*x.shape, 1])
            #i+= batch_size
            yield x, y 

def inception(IN, nk,ks,do=0.1):

    CONVA1 = Dropout(do)(Conv2D( nk,strides=[2,2], kernel_size=[ks,ks],activation='relu',padding='same')(IN))
    CONVB1 = Dropout(do)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(IN))
    CONVB2 = Dropout(do)(Conv2D( nk, strides=[2,2], kernel_size=[ks,ks],activation='relu',padding='same')(CONVB1))
    CONVC1 = Dropout(do)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(IN))
    CONVC2 = Dropout(do)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONVC1))
    CONVC3 = Dropout(do)(Conv2D( nk,strides=[2,2], kernel_size=[ks,ks],activation='relu',padding='same')(CONVC2))
    CONVD1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2],padding='same')(IN)

    CONCAT = Concatenate(axis=3)([CONVA1, CONVB2, CONVC3, CONVD1])


    return CONCAT

def make_compile_model( xdim=300,zdim=300 ) :
    IN = Input(shape=( xdim, zdim, 1))
    DO=0.1

    #LEVEL 1
    ks=3
    nk=8
    CONCAT1=inception(IN, nk, ks)
    CONCAT2=inception(CONCAT1, nk, ks)
    CONCAT3=inception(CONCAT2, nk, ks)

    #CONV1=MaxPooling2D(pool_size=(2, 2))(Dropout(0.1)(Conv2D( nk*3, kernel_size=[ks,ks],activation='relu',padding='same')(IN)))
    #CONV2=MaxPooling2D(pool_size=(2, 2))(Dropout(0.1)(Conv2D( nk*3, kernel_size=[ks,ks],activation='relu',padding='same')(CONV1)))
    #CONV3=MaxPooling2D(pool_size=(2, 2))(Dropout(0.1)(Conv2D( nk*2, kernel_size=[ks,ks],activation='relu',padding='same')(CONV2)))
    #CONV4=MaxPooling2D(pool_size=(2, 2))(Dropout(0.1)(Conv2D( nk*2, kernel_size=[ks,ks],activation='relu',padding='same')(CONV3)))
    #CONV5=MaxPooling2D(pool_size=(2, 2))(Dropout(0.1)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV4)))
    #CONV6=Dropout(0.1)(Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONV5))

    CONVE = Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONCAT3)
    
    CONVF =  Conv2D( nk, kernel_size=[ks,ks],activation='relu',padding='same')(CONVE)

    
    FLAT = Flatten()(CONVF)
 
    DENSE1 = Dense(8, activation='relu')(FLAT)
    DENSE2 = Dense(16, activation='relu')(DENSE1)
    DENSE3 = Dense(8, activation='relu')(DENSE2)
    
    OUT = Dense(1, activation='sigmoid')(DENSE3)

    model = Model(inputs=[IN], outputs=OUT)

    ada = keras.optimizers.Adam(0.0001)
    model.compile(loss = 'binary_crossentropy',  optimizer=ada,metrics=['acc'] )
    print(model.summary())
    return model

def train_model( model_fn, tissue_dir,not_tissue_dir, train_ratio=0.75,val_ratio=0.25, epochs=50):

    x, y = get_image_filenames(tissue_dir, not_tissue_dir)
    train_x, train_y, val_x, val_y = create_train_val_data(x, y, train_ratio, val_ratio)
   
    ### Calculate batch size
    batch_size, steps, val_steps = get_batch_size(train_x, train_y, 20)
    checkpoint_fn = "tissue_subset/checkpoint-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='acc', verbose=0, save_best_only=True, mode='max')
    
    ### Make Model
    model = make_compile_model( xdim=300,zdim=300)
    print("Tissue:", round(100.*np.sum(np.array(y) == 1.)/len(y),2), '%' )
    print("Not Tissue:", round(100.*np.sum(np.array(y) == 0.)/len(y),2), '%' )
    print("Batch Size:", batch_size)
    print('Data Points:', len(train_x), len(val_x), len(y))

    ### Fit Model
    test = gen(train_x, train_y, batch_size=batch_size)

    history = model.fit_generator( gen(train_x, train_y, batch_size=batch_size), 
                steps_per_epoch=steps, 
                validation_data=gen(val_x, val_y ), 
                validation_steps=val_steps, 
                epochs=epochs, 
                callbacks=[ checkpoint]
                )
    model.save(model_fn)
    return model



def apply_model_to_images(model, in_dir, out_dir, xstep=300, zstep=300, batch_size=40):
    if not os.path.exists(out_dir) :
        os.makedirs(out_dir)
    
    source_files = glob(in_dir+"/*.TIF")
    stride=50

    for f in source_files :
        print(f)
        img = imread(f)
        out = np.zeros([ int(img.shape[0]/stride)+1, int(img.shape[1]/stride)+1] )
        points=[]
        subsetList=[]
        for x in range(0, img.shape[0], stride) :
            for z in range(0,img.shape[1], stride) :
                subset = img[x:(x+xstep),z:z+zstep]
                if subset.shape[0] < xstep or subset.shape[1] < zstep :
                    subset = add_padding(subset, xstep, zstep)
                subset = subset.reshape(1,xstep,zstep)
                subsetList.append( subset )
                #xi = int(x+xstep/2.) if (x+xstep/2.) < img.shape[0] else img.shape[0]-1
                xi = int(x/stride) if int(x/stride) < img.shape[0] else img.shape[0]-1

                #zi = int(z+zstep/2.) if (z+zstep/2.) < img.shape[1] else img.shape[1]-1
                zi = int(z/stride) if int(z/stride) < img.shape[1] else img.shape[1]-1


                points.append( [xi, zi] )     

        subsetVolume = np.concatenate(subsetList, axis=0)
        subsetVolume = subsetVolume.reshape(*subsetVolume.shape,1)

        tissue_predicted = np.zeros((subsetVolume.shape[0],1))
        for i in range(0,subsetVolume.shape[0],batch_size):
            tissue_predicted[i:(i+batch_size)] = model.predict(subsetVolume[i:(i+batch_size),:], batch_size=batch_size )
        pts_array = np.zeros(img.shape)

        for i, pts in enumerate(points) :
            x=pts[0]
            z=pts[1]
            #pts_array[int(x*stride),int(z*stride)] = 1
            out[x,z]=tissue_predicted[i]
        
        #pts_array = binary_dilation(pts_array,iterations=10).astype(int)
        #pts_array *= np.max(img)
        pts_array += img
        xscale = img.shape[0] / out.shape[0]
        zscale = img.shape[1] / out.shape[1]

        out_rsl = rescale(out,[xscale,zscale],order=3)
        
        plt.subplot(3,1,1)
        plt.imshow(img)
        #plt.subplot(2,2,2)
        #plt.imshow(pts_array)
        plt.subplot(3,1,2)
        plt.imshow(out)
        plt.subplot(3,1,3)
        plt.imshow(out_rsl)
        plt.savefig(out_dir+os.sep+os.path.splitext(os.path.basename(f))[0]+'.png' )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--apply-model', dest='apply_model', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--create-init-data', dest='create_init_data', type=int,  default=0, help='Clobber results')
    parser.add_argument('--create-new-data', dest='create_new_data', type=int, default=0, help='Clobber results')
    
    args = parser.parse_args()

    create_init_data= args.create_init_data
    create_new_data = args.create_new_data
    clobber= args.clobber
    
    #Base Options
    tissue_dir='tissue_subset/tissue/'
    not_tissue_dir='tissue_subset/not_tissue/'
    unsorted_dir='tissue_subset/unsorted'

    if create_init_data :
        identifyBrainTissue("lin/R_slab_1/", unsorted_dir, total=create_init_data)

    replace_qc_img(unsorted_dir, tissue_dir, not_tissue_dir)
    #Get Model
    model_fn = "tissue_subset/model.h5"
    if not os.path.exists(model_fn) or clobber :
        model=train_model(model_fn, tissue_dir, not_tissue_dir,  epochs=7)
    else :
        model=load_model(model_fn)

    if create_new_data > 0 :
        identifyBrainTissue("lin/R_slab_*/", unsorted_dir, model=model, total=create_new_data)
    
    if args.apply_model :
        #Apply model to images
        apply_model_to_images(model, "lin/R_slab_1", "tissue_subset/predict")

