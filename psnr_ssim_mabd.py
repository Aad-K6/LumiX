import os
import glob
import time
import tensorflow as tf
import numpy as np
from config import TEST_LIST, RESULT_DIR

directory = 'test_set_results/'
TEST_RESULT_DIR = RESULT_DIR + directory
MAX_VAL = 255

sess = tf.Session()
t_vid1 = tf.placeholder(tf.float32, [None, None, None, None])
t_vid2 = tf.placeholder(tf.float32, [None, None, None, None])

def psnr_metric(x, y, max_val):
    mse = tf.reduce_mean(tf.square(x - y), axis=(1, 2, 3))
    psnr = 10.0 * tf.log(max_val ** 2 / mse) / tf.log(10.0)
    return tf.reduce_mean(psnr)

def ssim_metric(x, y, max_val):
    def _ssim(x, y, max_val):
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        mu_x = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        mu_y = tf.reduce_mean(y, axis=(1, 2, 3), keepdims=True)
        sigma_x = tf.reduce_mean(tf.square(x - mu_x), axis=(1, 2, 3), keepdims=True)
        sigma_y = tf.reduce_mean(tf.square(y - mu_y), axis=(1, 2, 3), keepdims=True)
        sigma_xy = tf.reduce_mean((x - mu_x) * (y - mu_y), axis=(1, 2, 3), keepdims=True)

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return tf.reduce_mean(ssim, axis=(1, 2, 3))

    return _ssim(x, y, max_val)

def get_psnr_ssim(sess, vid1, vid2):
    psnr_val = sess.run(psnr_metric(t_vid1, t_vid2, MAX_VAL), feed_dict={t_vid1: vid1, t_vid2: vid2})
    ssim_val = sess.run(ssim_metric(t_vid1, t_vid2, MAX_VAL), feed_dict={t_vid1: vid1, t_vid2: vid2})
    return psnr_val, ssim_val

def brightness(vid):
    R, G, B = vid[:, :, :, 0], vid[:, :, :, 1], vid[:, :, :, 2]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

def get_mse_mabd(vid1, vid2):
    b_vid1 = brightness(vid1)
    b_vid2 = brightness(vid2)
    mabd1 = np.diff(b_vid1).mean(axis=(1, 2))
    mabd2 = np.diff(b_vid2).mean(axis=(1, 2))
    return ((mabd1 - mabd2) ** 2).mean()

output_files = glob.glob(TEST_RESULT_DIR + '*.npy')
gt_files = [os.path.basename(file)[:-4] for file in output_files]

if 'psnr_ssim_mabd' in os.listdir('.'):
    os.rename('psnr_ssim_mabd', 'psnr_ssim_mabd' + '_' + str(time.localtime().tm_mon).zfill(2) + str(time.localtime().tm_mday).zfill(2) + '-' + str(time.localtime().tm_hour).zfill(2) + str(time.localtime().tm_min).zfill(2))

with open('psnr_ssim_mabd', 'w') as f:
    pass

all_psnr = 0
all_ssim = 0
all_mabd = 0

if len(gt_files) == 0:
    print("No files found in directory:", TEST_RESULT_DIR)
else:
    for output_file in output_files:
        out_vid = np.load(output_file)
        gt_file = os.path.basename(output_file)
        gt_vid = np.load('0_data/gt_he/' + gt_file)
        t0 = time.time()
        psnr, ssim = get_psnr_ssim(sess, out_vid, gt_vid)
        t1 = time.time()
        mabd = get_mse_mabd(out_vid, gt_vid)
        t2 = time.time()
        print('Done.\t{}s\t{}s'.format(t1 - t0, t2 - t1))
        with open('psnr_ssim_mabd', 'a') as f:
            f.write(os.path.basename(output_file)[:-4] + ' ' + str(psnr) + ' ' + str(ssim) + ' ' + str(mabd) + '\n')
        all_psnr += psnr
        all_ssim += ssim
        all_mabd += mabd

    with open('psnr_ssim_mabd', 'a') as f:
        if len(gt_files) > 0:
            f.write('\n' * 3 + 'overall_average ' + str(all_psnr / len(gt_files)) + ' ' + str(all_ssim / len(gt_files)) + ' ' + str(all_mabd / len(gt_files)) + '\n')
