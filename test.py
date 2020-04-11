from config import config, log_config
from utils import *
from model import *

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from random import shuffle
import matplotlib
import datetime
import time
import shutil
from tqdm import tqdm

batch_size = config.TRAIN.batch_size
batch_size_init = config.TRAIN.batch_size_init
lr_init = config.TRAIN.lr_init
lr_init_init = config.TRAIN.lr_init_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
n_epoch_init = config.TRAIN.n_epoch_init
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

lambda_adv = config.TRAIN.lambda_adv
lambda_lr_d = config.TRAIN.lambda_lr_d
lambda_binary = config.TRAIN.lambda_binary
lambda_perceptual = config.TRAIN.lambda_perceptual

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(np.ceil(np.sqrt(batch_size)))


def test():
    print('Test Start')
    mode='DMENet_BDCS'
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    mode_dir = config.TRAIN.root_dir + '{}'.format(mode)
    ckpt_dir = mode_dir + '/checkpoint'
    # sample_dir = mode_dir + '/samples/1_test/{}'.format(date)
    sample_dir='./author_dir'

    # cuhk_img_path = './data/'
    cuhk_img_path = './author/'
    test_blur_img_list = np.array(sorted(tl.files.load_file_list(path = cuhk_img_path, regx = '.jpg|.JPG|.png', printable = True)))
    test_blur_imgs = read_all_imgs(test_blur_img_list, path = cuhk_img_path, mode = 'RGB')
    
    avg_time = 0.
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    with tf.variable_scope('input'):
        patches_blurred = tf.placeholder('float32', [1, None, None, 3], name = 'input_patches')
        # labels = tf.placeholder('float32', [1, None, None, 1], name = 'labels')

    with tf.variable_scope('main_net') as scope:
        with tf.variable_scope('defocus_net') as scope:
            with tf.variable_scope('encoder') as scope:
                feats_down = VGG19_down(patches_blurred, reuse = False, scope = scope, is_test = True)
            with tf.variable_scope('decoder') as scope:
                output_defocus, feats_up, _, refine_lists = UNet_up(patches_blurred, feats_down, is_train = False, reuse = False, scope = scope)

    # init vars
    sess.run(tf.global_variables_initializer())

    # load checkpoint
    tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(mode), sess = sess)
    print('num of images '+str(len(test_blur_img_list)))

    for i in tqdm(np.arange(len(test_blur_imgs))):
        test_blur_img = np.copy(test_blur_imgs[i])
        test_blur_img = refine_image(test_blur_img)

        # run network
        print('processing {} ...'.format(test_blur_img_list[i]))
        tic = time.time()
        # feed_dict = {patches_blurred: np.expand_dims(test_blur_img, axis = 0), labels: np.expand_dims(test_gt_img, axis = 0)}
        feed_dict = {patches_blurred: np.expand_dims(test_blur_img, axis = 0)}
        defocus_map, feats_down_out, feats_up_out, refine_lists_out = sess.run([output_defocus, feats_down, feats_up, refine_lists], feed_dict)

        toc = time.time()
        defocus_map = np.squeeze(defocus_map)
        defocus_map_norm = defocus_map - defocus_map.min()
        defocus_map_norm = defocus_map_norm / defocus_map_norm.max()
        # print(sample_dir)

        print('processing {} ... Done [{:.3f}s]'.format(test_blur_img_list[i], toc - tic))
        avg_time = avg_time + (toc - tic)

        tl.files.exists_or_mkdir(sample_dir, verbose = False)
        tl.files.exists_or_mkdir(sample_dir + '/image')
        tl.files.exists_or_mkdir(sample_dir + '/out')
        tl.files.exists_or_mkdir(sample_dir + '/out_norm')
        # tl.files.exists_or_mkdir(sample_dir + '/gt')
        # print(defocus_map)
        # print(defocus_map_norm)

        scipy.misc.toimage(test_blur_img, cmin = 0., cmax = 1.).save(sample_dir + '/{0:04d}_1_input.png'.format(i))
        scipy.misc.toimage(test_blur_img, cmin = 0., cmax = 1.).save(sample_dir + '/image/{0:04d}.png'.format(i))
        scipy.misc.toimage(defocus_map, cmin = 0., cmax = 1.).save(sample_dir + '/{0:04d}_2_defocus_map_out.png'.format(i))
        scipy.misc.toimage(defocus_map, cmin = 0., cmax = 1.).save(sample_dir + '/out/{0:04d}.png'.format(i))
        scipy.misc.toimage(defocus_map_norm, cmin = 0., cmax = 1.).save(sample_dir + '/{0:04d}_3_defocus_map_norm_out.png'.format(i))
        scipy.misc.toimage(defocus_map_norm, cmin = 0., cmax = 1.).save(sample_dir + '/out_norm/{0:04d}.png'.format(i))

    avg_time = avg_time / len(test_blur_imgs)
    print('averge time: {:.3f}s'.format(avg_time))

if __name__ == '__main__':
    test()




