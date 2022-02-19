import os
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
import matplotlib.pyplot as plt


def crop_center(image):

    shape     = image.shape
    new_shape = min(shape[1], shape[2])

    offset_y  = max(shape[1] - shape[2], 0) // 2
    offset_x  = max(shape[2] - shape[1], 0) // 2

    image     = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
    return image 


def resize_image_to_square(image_np, image_size=(256, 256), preserve_aspect_ratio=True):

    image_np_extra = image_np.astype(np.float32)[np.newaxis, ...]

    if image_np_extra.max() > 1.0:
        image_np_extra = image_np_extra / 255
    
    if len(image_np_extra.shape) == 3:
        image_np_extra = tf.stack([image_np_extra, image_np_extra, image_np_extra], axis=-1)

        
    image_np_extra = crop_center(image_np_extra)
    image_np_extra = tf.image.resize(image_np_extra, image_size, preserve_aspect_ratio=True)
    return image_np_extra


def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    img        = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]

    if img.max() > 1.0:
        img = img / 255

    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=1)

    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_n(images, titles=('',)):

    n           = len(images)
    image_sizes = [image.shape[1] for image in images]
    w           = (image_sizes[0] * 6) // 320

    plt.figure(figsize=(w * n, w))
    gs          = gridspec.GridSpec(1, n, width_ratios=image_sizes)

    for i in range(n):

        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()