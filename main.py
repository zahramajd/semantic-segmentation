## Semantic Segmentation

#TODO:
# AlexNet
# uNet
# VGG16
# ResNet

import tensorflow as tf
import glob

def read_image(path):
    img_raw = tf.read_file(path)
    img_tensor = tf.image.decode_png(img_raw, channels=3)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    return img_tensor


path = 'DataSet/701_StillsRaw_full'
file_paths = glob.glob(path+'/*.png')


image_list = [read_image(file_path) for file_path in file_paths]
image_batch = tf.stack(image_list)

with tf.Session() as sess:
    image_batch = sess.run(image_batch)
    print(image_batch.shape)
