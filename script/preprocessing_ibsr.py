import os, glob
import subprocess
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.stats as stats

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def generate_real_lbl(tmp_lbl):
    labels = sorted([10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58])

    real_lbl = np.zeros(tmp_lbl.shape)

    for i in range(1, len(labels)+1):
        seg_one = (tmp_lbl == labels[i-1])
        real_lbl[seg_one] = i

    return real_lbl


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]


def _get_percentiles(percentiles_cutoff):
    quartiles = np.arange(25, 100, 25).tolist()
    deciles = np.arange(10, 100, 10).tolist()
    all_percentiles = list(percentiles_cutoff) + quartiles + deciles
    percentiles = sorted(set(all_percentiles))
    return np.array(percentiles)


def _get_average_mapping(percentiles_database):
    """Map the landmarks of the database to the chosen range.

    Args:
        percentiles_database: Percentiles database over which to perform the
            averaging.
    """
    # Assuming percentiles_database.shape == (num_data_points, num_percentiles)
    pc1 = percentiles_database[:, 0]
    pc2 = percentiles_database[:, -1]
    s1, s2 = STANDARD_RANGE
    slopes = (s2 - s1) / (pc2 - pc1)
    slopes = np.nan_to_num(slopes)
    intercepts = np.mean(s1 - slopes * pc1)
    num_images = len(percentiles_database)
    final_map = slopes.dot(percentiles_database) / num_images + intercepts
    return final_map


def _standardize_cutoff(cutoff):
    """Standardize the cutoff values given in the configuration.

    Computes percentile landmark normalization by default.

    """
    cutoff = np.asarray(cutoff)
    cutoff[0] = max(0., cutoff[0])
    cutoff[1] = min(1., cutoff[1])
    cutoff[0] = np.min([cutoff[0], 0.09])
    cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff


def normalize(array, landmarks, mask=None, cutoff=None, epsilon=1e-5):
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    mapping = landmarks

    data = array
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)

    if mask is None:
        mask = np.ones_like(data, np.bool)
    mask = mask.reshape(-1)

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

    quantiles_cutoff = _standardize_cutoff(cutoff_)
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles = _get_percentiles(percentiles_cutoff)
    percentile_values = np.percentile(data[mask], percentiles)

    # Apply linear histogram standardization
    range_mapping = mapping[range_to_use]
    range_perc = percentile_values[range_to_use]
    diff_mapping = np.diff(range_mapping)
    diff_perc = np.diff(range_perc)

    # Handling the case where two landmarks are the same
    # for a given input image. This usually happens when
    # image background is not removed from the image.
    diff_perc[diff_perc < epsilon] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])

    # Compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc

    # Compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(data, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * data + aff_img
    new_img = new_img.reshape(shape)
    new_img = new_img.astype(np.float32)

    return new_img


def calculate_landmarks(image_path='../dataset/IBSR/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip.nii.gz'):
    quantiles_cutoff = DEFAULT_CUTOFF
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles_database = []
    percentiles = _get_percentiles(percentiles_cutoff)

    count = 1
    t1_fn = glob.glob(image_path)
    for img_path in t1_fn:
        print(img_path)
        atlas_path = img_path.replace("_ana_strip.nii.gz", "_ana_brainmask.nii.gz")

        if os.path.exists(img_path) and os.path.exists(atlas_path):
            img_sitk = sitk.ReadImage(str(img_path))
            img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

            mask_sitk = sitk.ReadImage(str(atlas_path))
            mask_np = sitk.GetArrayFromImage(mask_sitk).swapaxes(0, 2)
            mask = (mask_np != 0)

            percentile_values = np.percentile(img_np[mask], percentiles)
            percentiles_database.append(percentile_values)
            count += 1
        else:
            raise FileNotFoundError

    percentiles_database = np.vstack(percentiles_database)
    mapping = _get_average_mapping(percentiles_database)
    print(mapping)

    np.save('../dataset/IBSR/mapping.npy', mapping)

    return mapping


def histogram_stardardization_resample_center_crop(mapping,
                                                   input_path='../dataset/IBSR/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip.nii.gz'):
    t1_fn = glob.glob(input_path)
    for img_path in t1_fn:
        mask_path = img_path.replace("_ana_strip.nii.gz", "_ana_brainmask.nii")
        atlas_path = img_path.replace("_ana_strip.nii.gz", "_seg_ana.nii")

        # ~~~~~~~~~~~~~~~ images and masks ~~~~~~~~~~~~~~~
        img_sitk = sitk.ReadImage(str(img_path))
        img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

        mask_sitk = sitk.ReadImage(str(mask_path))
        mask_np = sitk.GetArrayFromImage(mask_sitk).swapaxes(0, 2)
        mask = mask_np != 0

        # 1. histogram_stardardization
        img_np_hs = normalize(img_np, mapping, mask)

        # 2. resample
        img_sitk_hs = sitk.GetImageFromArray(img_np_hs.swapaxes(0, 2))
        img_sitk_hs.SetSpacing(img_sitk.GetSpacing())
        img_sitk_hs.SetDirection(img_sitk.GetDirection())
        img_sitk_hs.SetOrigin(img_sitk.GetOrigin())

        img_sitk_hs_resampled = resample_image(img_sitk_hs)
        img_np_hs_resampled = sitk.GetArrayFromImage(img_sitk_hs_resampled).swapaxes(0, 2)

        # ~~~~~~~~~~~~~~~ atlas ~~~~~~~~~~~~~~~
        atlas = sitk.ReadImage(str(atlas_path))

        # 1. resample
        atlas_resampled = resample_image(atlas, is_label=True)
        atlas_np_resampled = sitk.GetArrayFromImage(atlas_resampled).swapaxes(0, 2)

        # ~~~~~~~~~~~~~~~ center_crop ~~~~~~~~~~~~~~~
        lbl = generate_real_lbl(atlas_np_resampled)
        box = bbox2_3D(lbl)
        center = [(box[1] + box[0]) / 2, (box[3] + box[2]) / 2, (box[5] + box[4]) / 2]
        cropped_size = [80, 80, 80]

        atlas_cropped = lbl[int(center[0] - cropped_size[0] / 2):int(center[0] + cropped_size[0] / 2),
              int(center[1] - cropped_size[1] / 2):int(center[1] + cropped_size[1] / 2),
              int(center[2] - cropped_size[2] / 2):int(center[2] + cropped_size[2] / 2)].swapaxes(0, 2)
        img_np_hs_resampled_cropped = img_np_hs_resampled[int(center[0] - cropped_size[0] / 2):int(center[0] + cropped_size[0] / 2),
             int(center[1] - cropped_size[1] / 2):int(center[1] + cropped_size[1] / 2),
             int(center[2] - cropped_size[2] / 2):int(center[2] + cropped_size[2] / 2)].swapaxes(0, 2)

        new_img = sitk.GetImageFromArray(img_np_hs_resampled_cropped)
        new_img.SetSpacing(img_sitk_hs_resampled.GetSpacing())
        new_img.SetDirection(img_sitk_hs_resampled.GetDirection())
        new_img.SetOrigin(img_sitk_hs_resampled.GetOrigin())

        output_image_path = img_path.replace("/IBSR/", "/IBSR_preprocessed/").replace("_ana_strip.nii.gz", "_ana_strip_1mm_center_cropped.nii.gz")
        output_image_folder = '/'.join(output_image_path.split('/')[:-1])
        if not os.path.exists(str(output_image_folder)):
            os.makedirs(str(output_image_folder))
        sitk.WriteImage(new_img, str(output_image_path))

        new_atlas = sitk.GetImageFromArray(atlas_cropped)
        new_atlas.SetSpacing(atlas_resampled.GetSpacing())
        new_atlas.SetDirection(atlas_resampled.GetDirection())
        new_atlas.SetOrigin(atlas_resampled.GetOrigin())

        output_atlas_path = img_path.replace("/IBSR/", "/IBSR_preprocessed/").replace("_ana_strip.nii.gz", "_seg_ana_1mm_center_cropped.nii.gz")
        sitk.WriteImage(new_atlas, str(output_atlas_path))


def plot_hist(input_path='../dataset/IBSR_preprocessed/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip_1mm_center_cropped.nii.gz'):
    t1_fn = glob.glob(input_path)
    for volpath in t1_fn:
            img_sitk = sitk.ReadImage(str(volpath))
            img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)
            data = np.reshape(np.clip(img_np, 0, 100), -1)
            density = stats.gaussian_kde(data)
            xs = np.linspace(0, 100, 100)
            density.covariance_factor = lambda: .25
            density._compute_covariance()

            plt.plot(xs, density(xs))
    plt.show()


if __name__ == '__main__':
    mapping = calculate_landmarks(image_path='../dataset/IBSR/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip.nii.gz')
    # mapping = np.asarray([-3.55271368e-15, 2.57710517e+01, 3.88517641e+01, 4.35466129e+01, 4.71340347e+01, 5.33819476e+01, 5.94418704e+01, 6.61089981e+01, 7.37854501e+01, 7.78200114e+01, 8.16028501e+01, 8.96790844e+01, 1.00000000e+02])

    histogram_stardardization_resample_center_crop(mapping=mapping,
                                                   input_path='../dataset/IBSR/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip.nii.gz')

    plot_hist(input_path='../dataset/IBSR_preprocessed/IBSR_nifti_stripped/IBSR_*/IBSR_*_ana_strip_1mm_center_cropped.nii.gz')
