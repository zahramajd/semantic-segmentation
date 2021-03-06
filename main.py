## Semantic Segmentation
#TODO:
# test parameters

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image
from pylab import *
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute 
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import resnet50, vgg19



def _read_to_tensor(fname, output_height=224, output_width=224, normalize_data=False):
    
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)
    
    output = tf.image.resize(imgs_decoded, [output_height, output_width])
    
    if normalize_data:
        output = (output - 128) / 128

    return output

def read_images(img_dir):
    
    file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    
    frames_list = [file for file in file_list if ('_L' not in file) and ('txt' not in file)]
    frames_list.remove('.DS_Store')

    masks_list = [file for file in file_list if ('_L' in file) and ('txt' not in file)]
    
    frames_list.sort()
    masks_list.sort()
    
    # print('{} frame files found in the provided directory.'.format(len(frames_list)))
    # print('{} mask files found in the provided directory.'.format(len(masks_list)))
    
    frames_paths = [os.path.join(img_dir, fname) for fname in frames_list]
    masks_paths = [os.path.join(img_dir, fname) for fname in masks_list]
    
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)
    
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)
    
    # print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    # print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list

def generate_image_folder_structure(DATA_PATH, frames, masks, frames_list, masks_list):
    
    folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val']

    # for folder in folders:
    #     try:
    #         os.makedirs(DATA_PATH + folder)
    #     except Exception as e: print(e)

    frame_batches = tf.compat.v1.data.make_one_shot_iterator(frames) 
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)
    
    dir_name='train'
    for file in zip(frames_list[:-round(0.2*len(frames_list))],masks_list[:-round(0.2*len(masks_list))]):
        
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)
        
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)
        
        frame.save(DATA_PATH+'{}_frames/{}'.format(dir_name,dir_name)+'/'+file[0])
        mask.save(DATA_PATH+'{}_masks/{}'.format(dir_name,dir_name)+'/'+file[1])
    
    dir_name='val'
    for file in zip(frames_list[-round(0.2*len(frames_list)):],masks_list[-round(0.2*len(masks_list)):]):
        
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)
        
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)
        
        frame.save(DATA_PATH+'{}_frames/{}'.format(dir_name,dir_name)+'/'+file[0])
        mask.save(DATA_PATH+'{}_masks/{}'.format(dir_name,dir_name)+'/'+file[1])
    
    # print("Saved {} frames to directory {}".format(len(frames_list),DATA_PATH))
    # print("Saved {} masks to directory {}".format(len(masks_list),DATA_PATH))
  
def parse_code(l):

    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

def convert_labels(img_dir):

    label_codes, label_names = zip(*[parse_code(l) for l in open(img_dir+"label_colors.txt")])
    label_codes, label_names = list(label_codes), list(label_names)

    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}

    name2id = {v:k for k,v in enumerate(label_names)}
    id2name = {k:v for k,v in enumerate(label_names)}

    return id2code

def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape((-1,3)) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

def onehot_to_rgb(onehot, colormap):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def TrainAugmentGenerator(DATA_PATH, id2code, seed = 1, batch_size = 5):

    train_image_generator = train_frames_datagen.flow_from_directory(
    DATA_PATH + 'train_frames/',
    batch_size = batch_size, seed = seed, target_size = (224, 224))

    train_mask_generator = train_masks_datagen.flow_from_directory(
    DATA_PATH + 'train_masks/',
    batch_size = batch_size, seed = seed, target_size = (224, 224))

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)

def TrainAugmentGenerator2(DATA_PATH, id2code, seed = 1, batch_size = 5):
    
    train_image_generator = train_frames_datagen.flow_from_directory(
    DATA_PATH + 'train_frames/',
    batch_size = batch_size, seed = seed, target_size = (227, 227))

    train_mask_generator = train_masks_datagen.flow_from_directory(
    DATA_PATH + 'train_masks/',
    batch_size = batch_size, seed = seed, target_size = (227, 227))

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)

def ValAugmentGenerator(DATA_PATH, id2code, seed = 1, batch_size = 5):

    val_image_generator = val_frames_datagen.flow_from_directory(
    DATA_PATH + 'val_frames/',
    batch_size = batch_size, seed = seed, target_size = (224, 224))


    val_mask_generator = val_masks_datagen.flow_from_directory(
    DATA_PATH + 'val_masks/',
    batch_size = batch_size, seed = seed, target_size = (224, 224))


    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)

def ValAugmentGenerator2(DATA_PATH, id2code, seed = 1, batch_size = 5):
    
    val_image_generator = val_frames_datagen.flow_from_directory(
    DATA_PATH + 'val_frames/',
    batch_size = batch_size, seed = seed, target_size = (227, 227))


    val_mask_generator = val_masks_datagen.flow_from_directory(
    DATA_PATH + 'val_masks/',
    batch_size = batch_size, seed = seed, target_size = (227, 227))


    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)

def VGGSegnet(n_classes, input_height=224, input_width=224, vgg_level=3):
    
    VGG_Weights_path = "pretrained_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

    img_input = Input(shape=(input_height, input_width, 3))

    # block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last')(x)
    f1 = x

    # block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last')(x)
    f2 = x

    # block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last')(x)
    f3 = x

    # block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last')(x)
    f4 = x

    # block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last')(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg  = Model(img_input, x)
    vgg.load_weights(VGG_Weights_path)

    levels = [f1, f2, f3, f4, f5]

    o = levels[vgg_level]

    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last', activation='elu'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last',activation='elu'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last',activation='elu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last',activation='elu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last',activation='elu'))(o)
    o = (BatchNormalization())(o)
    
    
    o = Conv2D(n_classes, (3, 3), padding='same', data_format='channels_last')(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model

def alexNetSegmentation(n_classes, input_height=227, input_width=227):

    img_input = Input(shape=(input_height, input_width, 3))

    # block 1
    x = Conv2D(96, (11, 11), activation='relu', padding='same', strides=(4, 4), name='block1_conv1', data_format='channels_last')(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool', data_format='channels_last')(x)
    x = (BatchNormalization())(x)
    f1 = x

    # block 2
    x = Conv2D(256, (5, 5), activation='relu', padding='same', strides=(1, 1), name='block2_conv1', data_format='channels_last')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool', data_format='channels_last')(x)
    x = (BatchNormalization())(x)
    f2 = x

    # block 3
    x = Conv2D(384, (3, 3), activation='relu', padding='same', strides=(1, 1), name='block3_conv1', data_format='channels_last')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same', strides=(1, 1), name='block3_conv2', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1), name='block3_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool', data_format='channels_last')(x)
    f3 = x

    # x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)

    x = Conv2D(21, (1, 1), activation='relu', padding='same', strides=(1, 1), name='dec_1', data_format='channels_last')(x)
    x = Conv2DTranspose(32 ,(63,63), strides=(32, 32),data_format='channels_last')(x)

    x = ZeroPadding2D((2, 2))(x)
    o = x

    model = Model(img_input, o)
    print(model.summary())
    return model

def ResNetSegmentation(n_classes, input_height=224, input_width=224):

    img_input = Input(shape=(input_height, input_width, 3))

    resnet_model = resnet50.ResNet50(input_tensor=img_input ,weights="imagenet")

    o = resnet_model.get_layer('activation_39').output

    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last', kernel_initializer='he_uniform'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last', kernel_initializer='he_uniform'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last', kernel_initializer='he_uniform'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last', kernel_initializer='he_uniform'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last', kernel_initializer='he_uniform'))(o)
    o = (BatchNormalization())(o)
    
    
    o = Conv2D(n_classes, (3, 3), padding='same', data_format='channels_last', kernel_initializer='he_uniform')(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model

def vgg19Segmentation(n_classes, input_height=224, input_width=224):

    img_input = Input(shape=(input_height, input_width, 3))

    vgg19_model = vgg19.VGG19(input_tensor=img_input ,weights="imagenet")

    o = vgg19_model.get_layer('block4_pool').output

    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)
    
    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2,2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1,1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)
    
    
    o = Conv2D(n_classes, (3, 3), padding='same', data_format='channels_last')(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


img_dir = 'data/CamSeq01/'

# just once at beggining
frame_tensors, masks_tensors, frames_list, masks_list = read_images(img_dir)
generate_image_folder_structure(img_dir, frame_tensors, masks_tensors, frames_list, masks_list)

id2code = convert_labels(img_dir)

# normalizing
data_gen_args = dict(rescale=1./255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)

# model = alexNetSegmentation(32)
model = VGGSegnet(32, vgg_level=3)
# model = ResNetSegmentation(32)
# model = vgg19Segmentation(32)


# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tversky_loss,dice_coef,'accuracy'])
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[tversky_loss,dice_coef,'accuracy'])


tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='camvid_model_vgg16_segnet_checkpoint.h5', monitor='accuracy', save_best_only='True', save_weights_only='True', verbose=1)
es = EarlyStopping(mode='min', monitor='val_loss', patience=50, verbose=1)
callbacks = [tb, mc, es]

batch_size = 32
steps_per_epoch = 15.0
validation_steps = 4.0
num_epochs = 30

result = model.fit_generator(TrainAugmentGenerator(DATA_PATH=img_dir, id2code=id2code), steps_per_epoch=steps_per_epoch,
                validation_data = ValAugmentGenerator(DATA_PATH=img_dir, id2code=id2code), 
                validation_steps = validation_steps, epochs=num_epochs, callbacks=callbacks, verbose=1)

model.save_weights("camvid_model_vgg16_segnet.h5", overwrite=True)


# Get actual number of epochs model was trained for
N = len(result.history['loss'])

#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))

fig.add_subplot(1,2,1)
plt.title("Training Loss")
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.ylim(0, 1)

fig.add_subplot(1,2,2)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), result.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), result.history["val_accuracy"], label="val_accuracy")
plt.ylim(0, 1)

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

training_gen = TrainAugmentGenerator(DATA_PATH=img_dir, id2code=id2code)
testing_gen = ValAugmentGenerator(DATA_PATH=img_dir, id2code=id2code)

batch_img,batch_mask = next(testing_gen)
pred_all= model.predict(batch_img)
np.shape(pred_all)

for i in range(0,np.shape(pred_all)[0]):
    
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(batch_img[i])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
    
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(onehot_to_rgb(batch_mask[i],id2code))
    ax2.grid(b=None)
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Predicted labels')
    ax3.imshow(onehot_to_rgb(pred_all[i],id2code))
    ax3.grid(b=None)
    
    plt.show()