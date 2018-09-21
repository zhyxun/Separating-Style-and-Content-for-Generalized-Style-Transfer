__author__ = 'yxzhang'
import argparse
import collections
from ops import *
import os
import numpy as np
import pickle
import scipy.misc
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=500, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--dim", type=float, default=100, help="the dimension of the difference encode")
parser.add_argument("--num_parallel_prefetch", type=int, default=30, help="the number of prefetch")
parser.add_argument("--adam_lr", type=float, default=0.0002, help="initial learning rate for adam")
# parser.add_argument("--decay_rate", type=float, default=0.8, help="decay for learning rate")
# parser.add_argument("--decay_steps", type=float, default=12000, help="decay frequency for learning rate")

parser.add_argument("--style_num", type=float, default=50, help="the number of styles in each batch")
parser.add_argument("--content_num", type=float, default=50, help="the number of characters in each batch")
parser.add_argument("--style_sample_n", type=float, default=10, help="the number of samples for each style")
parser.add_argument("--content_sample_n", type=float, default=10, help="the number of samples for each content")
parser.add_argument("--target_batch_size", type=int, default=50, help="number of target images in batch")

parser.add_argument("--image_channel", type=int, default=1, help="channel of images")
parser.add_argument("--loss_func", type=str, default='l1_loss', help="loss function used")

# export options
parser.add_argument("--output_filetype", default="jpg", choices=["png", "jpg"])
a = parser.parse_args()
Model = collections.namedtuple("Model", "model_loss, outputs, model_grads_and_vars, model_train, mse")

def process(input):
    image_string = tf.read_file(input)
    try:
        decode = tf.image.decode_jpeg
        images = decode(image_string,channels=1)
    except:
        decode = tf.image.decode_png
        images = decode(image_string,channels=1)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    return images

def encoder_block(inputs, kernel_size, stride, channel_out, name, mode):
    with tf.variable_scope(name):
        convolved = conv(inputs, channel_out, stride=stride,kernel_size=kernel_size)
        normalized = batchnorm(convolved, mode)
        rectified = lrelu(normalized, 0.2)
        return rectified

def Content_encoder(Content_inputs, mode):
    layers = [Content_inputs]
    kernel_sizes = [5, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 2, 2, 2, 2, 2, 2, 2]
    out_channels = [a.ndf, a.ndf*2, a.ndf*4, a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*8]

    with tf.variable_scope("content_encoder"):#80*80 80*80

        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            out_channel = out_channels[i]
            output = encoder_block(layers[-1], kernel_size, stride, out_channel, "encoder_"+str(i+1), mode)
            layers.append(output)
        return layers

def Style_encoder(Content_inputs, mode):
    layers = [Content_inputs]
    kernel_sizes = [5, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 2, 2, 2, 2, 2, 2, 2]
    out_channels = [a.ndf, a.ndf*2, a.ndf*4, a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*8]

    with tf.variable_scope("style_encoder"):#80*80 80*80

        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            out_channel = out_channels[i]
            output = encoder_block(layers[-1], kernel_size, stride, out_channel, "encoder_"+str(i+1), mode)
            layers.append(output)

        return layers


def decoder_block(inputs, kernel_size, stride, channel_out, name, mode, add):
    with tf.variable_scope(name):
        output = deconv(inputs, channel_out, kernel_size=kernel_size,stride=stride,add=add)
        output = batchnorm(output, mode)
        rectified = relu(output)
        return rectified

def Decoder(input, contents_rec, generator_outputs_channels, mode):
    layers = [input]
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 5]
    strides = [2, 2, 2, 2, 2, 2, 2, 1]
    adds = [0, -1, -1, 0, 0, 0, 0, 0]
    out_channels = [a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*8, a.ndf*4, a.ndf*2, a.ndf, 1]

    for i in range(len(kernel_sizes)-1):
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        out_channel = out_channels[i]
        add = adds[i]

        inputs = tf.concat([layers[-1],contents_rec[-i-1]],3)
        output = decoder_block(inputs, kernel_size, stride, out_channel, 'decoder_'+str(i+1), mode, add)
        layers.append(output)

    with tf.variable_scope("decoder_8"):#80*80 80*80
        input = tf.concat([layers[-1],contents_rec[-8]],3)
        output = deconv(input,generator_outputs_channels,kernel_size=5,stride=1,add=0)
        layers.append(output)
    return layers

def trans_style(styles,decoder_input_dim,content_dim):
    shape = styles.get_shape().as_list()

    with tf.variable_scope("trans"):
        matrix1 = tf.get_variable("Matrix1",[shape[-1],content_dim,decoder_input_dim],tf.float32,
                                 tf.random_normal_initializer(0, 0.02))
        style_weighted = tf.einsum('bi,ijk->bjk',styles,matrix1)

        return style_weighted

def create_generator(inputsS, inputsC, targets, zero_nt, mean_pixel_valuet):
    #process the pictures for content*************************************************
    with tf.variable_scope("Content_encode"):
        for j in range(a.content_num):
            temp = tf.slice(inputsC,[j*a.content_sample_n,0,0,0],[1,-1,-1,-1])
            for i in range(1,a.content_sample_n):
                temp = tf.concat([temp,tf.slice(inputsC,[j*a.content_sample_n+i,0,0,0],[1,-1,-1,-1])],3)
            if j == 0:
                temps = tf.zeros(temp.get_shape().as_list())

            temps = tf.concat([temps,temp],0)
        contents_rec = Content_encoder(tf.slice(temps,[1,0,0,0],[-1,-1,-1,-1]), a.mode)
        contents = contents_rec[-1]

    #process the pictures for style***************************************************
    with tf.variable_scope("Style_encode"):
        for i in range(a.style_num):
            temp = tf.slice(inputsS,[i*a.style_sample_n,0,0,0],[1,-1,-1,-1])
            for j in range(1,a.style_sample_n):
                temp = tf.concat([temp,tf.slice(inputsS,[i*a.style_sample_n+j,0,0,0],[1,-1,-1,-1])],3)
            if i == 0:
                temps = tf.zeros(temp.get_shape().as_list())

            temps = tf.concat([temps,temp],0)
        styles = Style_encoder(tf.slice(temps,[1,0,0,0],[-1,-1,-1,-1]), a.mode)[-1]

    #combine the one style with one content as the input of the decoder***************
    with tf.variable_scope("Combine"):
        content_dim = np.prod(tf.slice(contents,[0,0,0,0],[1,-1,-1,-1]).get_shape().as_list())
        decoder_input_dim = a.ndf*8
        shape = tf.slice(styles,[0,0,0,0],[1,-1,-1,-1]).get_shape().as_list()
        styles = tf.reshape(styles,[a.style_num,-1])
        styles_transed = trans_style(styles,decoder_input_dim,content_dim)
        generator_outputs_channels = inputsC.get_shape().as_list()[-1]

        for i in range(a.style_num):
            stylei = styles_transed[i,:,:]
            j = i
            contentj = tf.slice(contents,[j,0,0,0],[1,-1,-1,-1])
            contentj = tf.reshape(contentj,[1,-1])

            decoder_inputij = tf.matmul(contentj,stylei)
            decoder_inputij = tf.reshape(decoder_inputij,shape)
            if i == 0:
                decoder_inputs = tf.zeros(decoder_inputij.get_shape().as_list())

            decoder_inputs = tf.concat([decoder_inputs,decoder_inputij],0)

        pictures = tf.slice(decoder_inputs,[1,0,0,0],[-1,-1,-1,-1])
        pictures_decode = Decoder(pictures, contents_rec, generator_outputs_channels, a.mode)[-1]
        pictures_decode_s = tf.sigmoid(pictures_decode)
        mse = tf.reduce_mean(tf.square(pictures_decode_s - targets))

        # weight loss
        if a.loss_func == 'l1_loss':
            pictures_decode_one = tf.reshape(pictures_decode_s,[a.target_batch_size,-1])
            pictures_origin_one = tf.reshape(targets,[a.target_batch_size,-1])
            weight = 1.0/tf.to_float(zero_nt)*tf.nn.softmax(mean_pixel_valuet)*a.target_batch_size
            loss_for_decoder = tf.reduce_mean(tf.reduce_sum(tf.abs(pictures_decode_one-pictures_origin_one),1)*weight)
        elif a.loss_func == 'ce_loss':
            loss_for_decoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=pictures_decode))

    return pictures_decode_s, loss_for_decoder, mse


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    image_saved = merge(images,size)
    image_saved = image_saved[:,:,0]
    return scipy.misc.imsave(path, image_saved)

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, step, size,kind):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filename = str(step) + "-" + str(kind) + ".png"
    out_path = os.path.join(image_dir, filename)
    imsave(images, size, out_path)

def append_index(fileset):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>outputs</th><th>targets</th></tr>")
        index.write("<tr>")

    index.write("<td>%s</td>" % fileset["name"])
    for kind in ["outputs","targets"]:
        index.write("<td><img src='images/%s'></td>" % fileset[kind])
    index.write("</tr>")
    return index_path

