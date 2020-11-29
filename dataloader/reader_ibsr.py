from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import tensorflow as tf
import numpy as np

from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array
# from dltk.io.preprocessing import whitening
from skimage import exposure

_mean = 51.874330275992
_std = 22.917319692490114

labels = sorted([10, 11, 12 ,13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58])
# x = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]
# label_kv = {10: "Left-Thalamus",
#             49: "Right-Thalamus",
#             11: "Left-Caudate",
#             50: "Right-Caudate ",
#             12 : "Left-Putamen",
#             51: "Right-Putamen",
#             13: "Left-Pallidum",
#             52: "Right-Pallidum",
#             17: "Left-Hippocampus",
#             53: "Right-Hippocampus",
#             18: "Left-Amygdala",
#             54: "Right-Amygdala",
#             26: "Left-Accumbens",
#             58: "Right-Accumbens"}


# len(labels)
NUM_CLASSES = len(labels) + 1


def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.

    Args:
        file_references (list): A list of lists containing file references, such
            as [['id_0', 'image_filename_0', target_value_0], ...,
            ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parameterise read_fn ouputs
            (e.g. reader_params = {'n_examples': 10, 'example_size':
            [64, 64, 64], 'extract_examples': True}, etc.).

    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """

    def _augment(img, lbl):
        """An image augmentation function"""
        img = add_gaussian_noise(img, sigma=0.1)
        # [img, lbl] = flip([img, lbl], axis=0)

        return img, lbl

    for f in file_references:
        subject_id = f[0]
        img_fn = f[1]

        # Read the image nii with sitk and keep the pointer to the sitk.Image of an input
        t1_sitk = sitk.ReadImage(str(img_fn).replace("/IBSR/", "/IBSR_preprocessed/").replace("_ana_strip.nii.gz", "_ana_strip_1mm_center_cropped.nii.gz"))
        t1 = ((np.clip(sitk.GetArrayFromImage(t1_sitk), 0., 100.) - 0.) / 100.).swapaxes(0, 1)

        lbl_sitk = sitk.ReadImage(str(img_fn).replace("/IBSR/", "/IBSR_preprocessed/").replace("_ana_strip.nii.gz", "_seg_ana_1mm_center_cropped.nii.gz"))
        lbl = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32).swapaxes(0, 1)

        # Create a 4D multi-sequence image (i.e. [channels, x, y, z])
        images = np.stack([t1], axis=-1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images},
                   'labels': None,
                   'sitk': t1_sitk,
                   'subject_id': subject_id}

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)

        # Check if the reader is supposed to return training examples or full
        #  images
        if params['extract_examples']:
            n_examples = params['n_examples']
            example_size = params['example_size']
            class_weights = params['class_weights'] if "class_weights" in params else None

            images, lbl = extract_class_balanced_example_array(
                image=images,
                label=lbl,
                example_size=example_size,
                n_examples=n_examples,
                classes=NUM_CLASSES,
                class_weights=class_weights
            )

            for e in range(len(images)):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': lbl[e].astype(np.int32)},
                       'subject_id': subject_id}
        else:
            yield {'features': {'x': images},
                   'labels': {'y': lbl},
                   'sitk': t1_sitk,
                   'subject_id': subject_id}

    return
