# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import util.metrics as metrics2
from dltk.utils import sliding_window_segmentation_inference
from tensorflow.contrib import predictor

from dataloader.reader_ibsr import read_fn
from skimage import measure
from scipy import ndimage
import heapq
import copy


READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = 3


def predict(args):
    # Read in the csv with the file names you would want to predict on
    file_names_list = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).values

    # We trained on the last 15 subjects, so we predict on the rest
    file_names = file_names_list[:3]

    # From the model_path, parse the latest saved model and restore a predictor from it
    export_dir = [os.path.join(args.model_path, o) for o in os.listdir(args.model_path)
                  if os.path.isdir(os.path.join(args.model_path, o)) and
                  o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Fetch the output probability op of the trained network
    y_prob = my_predictor._fetch_tensors['y_prob']
    num_classes = y_prob.get_shape().as_list()[-1]

    # Iterate through the files, predict on the full volumes and compute a Dice coefficient
    from collections import defaultdict
    total_dice = defaultdict(list)
    total_hd = defaultdict(list)

    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)
        lbl = np.expand_dims(output['labels']['y'], axis=0)

        # Do a sliding window inference with our DLTK wrapper
        pred = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[y_prob],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=1)[0]

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        # Save the file as .nii.gz using the header information from the original sitk image
        output_fn = os.path.join(args.model_path, '{}_seg.nii.gz'.format(output['subject_id']))
        new_sitk = sitk.GetImageFromArray(pred[0, :, :, :].astype(np.int32))
        sitk.WriteImage(new_sitk, output_fn)

        # Calculate the AVG Dice coefficient for one image
        dsc = np.nanmean(metrics2.dice(pred, lbl, num_classes)[1:15])
        hd = np.nanmean(metrics2.hd(pred[0], lbl[0], num_classes)[1:15])

        # Calculate and Print each Dice coefficient for one image
        for idx, i in enumerate([14, 13, 6, 5, 12, 11, 10, 9, 8, 7, 4, 3, 2, 1]):
            dsc_tmp = metrics2.dice(pred, lbl, num_classes)[i]
            total_dice.setdefault("dsc_{}".format(idx), []).append(dsc_tmp)
            print('Id={}; Dice_{}={:0.4f}; time={:0.2} secs;'.format(output['subject_id'], idx, dsc_tmp, time.time() - t0))

        total_dice.setdefault("total_mean_dsc", []).append(dsc)
        print('Id={}; AVG Dice={:0.4f}; time={:0.2} secs; output_path={};'.format(output['subject_id'], dsc, time.time() - t0, output_fn))

        for idx, i in enumerate([14, 13, 6, 5, 12, 11, 10, 9, 8, 7, 4, 3, 2, 1]):
            hd_tmp = metrics2.hd(pred[0], lbl[0], num_classes)[i]
            total_hd.setdefault("hd_{}".format(idx), []).append(hd_tmp)
            print('Id={}; hd_{}={:0.4f}; time={:0.2} secs;'.format(output['subject_id'], idx, hd_tmp, time.time() - t0))
        total_hd.setdefault("total_mean_hd", []).append(hd)
        print('Id={}; AVG HD={:0.4f}; time={:0.2} secs; output_path={};'.format(output['subject_id'], hd, time.time() - t0, output_fn))

    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~ Dice Results on All Test Cases ~~~~~~~~~~~~~~~~~~~~~~")

    all_dice = []
    for k, v in total_dice.items():
        all_dice.append(np.mean(v))
        print(k, "%.3f" % (np.mean(v)), "±", "%.3f" % (np.std(v)))


    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~ HD Results (mean, std) on All Test Cases ~~~~~~~~~~~~~~~~~~~~~~")

    all_hd = []
    for k, v in total_hd.items():
        v = [i for i in v if i != 0]
        print(k, "%.2f" % (np.mean(v)), "±", "%.2f" % (np.std(v)))
        all_hd.append(np.mean(v))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MRBrainS18 example segmentation deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default = './checkpoint/ibsr_1'

    )
    parser.add_argument('--csv', default='./conf/ibsr_stripped.csv')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Call training
    predict(args)
