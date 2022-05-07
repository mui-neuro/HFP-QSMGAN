import os
import shlex
import shutil
import ants

import numpy as np

from os.path import isfile, join
from subprocess import Popen, PIPE
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.util.shape import view_as_windows
from itertools import product

from transforms3d.affines import compose
from transforms3d.taitbryan import euler2mat


def assert_dir(dir_path):
    full_path = os.path.abspath(dir_path)
    if not os.path.isdir(full_path):
        print('Creating %s' % full_path)
        os.makedirs(full_path)


def move(src, dest):
    print('Moving %s to %s' % (src, dest))
    shutil.move(src, dest)


def copy(src, dest):
    print('Copying %s to %s' % (src, dest))
    shutil.copy(src, dest)


def remake_dir(dir_path):
    full_path = os.path.abspath(dir_path)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path)


def run(cmd, live_verbose=False):
    print('\n' + cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    if output:
        print(output.decode('latin_1'))
    if error:
        print(error.decode('latin_1'))


def dice_(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    return (2. * intersection + smooth) /  \
        (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def adjust_shape(img, target_shape):

    """ Crop or pad image to target shape """

    # Crop
    data = img.numpy()
    low = np.floor((np.array(img.shape) - target_shape)/2.).astype(int)
    high = np.ceil((np.array(img.shape) - target_shape)/2.).astype(int)
    low_ = low.copy()
    high_ = high.copy()
    low_[[l_ < 0 for l_ in low_]] = 0
    high_[[h_ < 0 for h_ in high_]] = 0
    high_ = np.array(img.shape) - high_
    data = data[low_[0]:high_[0],
                low_[1]:high_[1],
                low_[2]:high_[2]]

    # Pad
    low_ = -low.copy()
    high_ = -high.copy()
    low_[[l_ < 0 for l_ in low_]] = 0
    high_[[h_ < 0 for h_ in high_]] = 0
    pad_width = [(low_[0], high_[0]),
                 (low_[1], high_[1]),
                 (low_[2], high_[2])]
    data = np.pad(data, pad_width, mode='constant', constant_values=0)

    # Set ants params
    spacing = img.spacing
    direction = img.direction
    has_components = img.has_components
    is_rgb = img.is_rgb

    aff = np.zeros([4, 4])
    aff[:3, :3] = img.direction*spacing
    aff[:3, 3] = img.origin
    aff[3, 3] = 1.
    inv_aff = np.linalg.inv(aff)
    inv_aff[:3, 3] -= low
    origin = list(np.linalg.inv(inv_aff)[:3, 3])

    # This creates temporary files... has to be avoided
    # img = ants_ref.to_nibabel()
    # inv_aff = np.linalg.inv(img.affine)
    # inv_aff[:3,3] -= low
    # aff = np.linalg.inv(inv_aff)
    # print(aff)
    # return ants.from_nibabel(nib.Nifti1Image(d, aff))

    return ants.from_numpy(data,
                           origin=origin,
                           spacing=spacing,
                           direction=direction,
                           has_components=has_components,
                           is_rgb=is_rgb)


def ants_rigid(fin, ref, fout, prefix):

    """ Wrapper for ANTs rigid registration """

    cmd = 'antsRegistration -d 3 \
            -n BSpline \
            -r [ %s,%s,1] \
            -t Rigid[0.1] \
            -m MI[ %s,%s,1,32,Regular,0.25] \
            -c [1000x500x250x100,1e-6,10] \
            -f 8x4x2x1 \
            -s 3x2x1x0vox \
            -o [%s,%s] -v' % (ref, fin, ref, fin, prefix, fout)
    run(cmd)


def create_brainmask(fin, fout):

    brain_file = '%s_brain.nii.gz' % fout
    if not isfile(brain_file):
        cmd = 'bet %s %s -m -R' % (fin, brain_file)
        run(cmd)


def random_affine(rng=None):

    if rng is None:
        rng = np.random.default_rng()

    trans = [0, 0, 0]
    rots = rng.uniform(-5.0, 5.0, size=3)/180 * np.pi
    # zooms = np.array([1, 1, 1])
    zooms = rng.uniform(0.9, 1.1, size=3)
    # shears = [0, 0, 0]
    shears = rng.uniform(0.0, 5.0, size=3)/180 * np.pi
    Rmat = euler2mat(*rots)
    M = compose(trans, Rmat, zooms, shears)

    return M


def random_elastic_transform(
        image_shape,
        alpha, sigma,
        rng=None):

    """
        Elastic deformation of images as described in [Simard2003] (with modifications).
        [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    if rng is None:
        rng = np.random.default_rng()

    dx = gaussian_filter((rng.random(size=image_shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random(size=image_shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((rng.random(size=image_shape) * 2 - 1), sigma) * alpha

    elastic = [dx, dy, dz]

    return elastic


def apply_transforms(
            image,
            affine=None,
            elastic=None,
            LR=False,
            order=3,  # cubic
            mode='constant'
        ):

    # Assume constant shape for all inputs
    if isinstance(image, list):
        image_shape = image[0].shape
    else:
        image_shape = image.shape

    x, y, z = np.meshgrid(np.arange(image_shape[1]),
                          np.arange(image_shape[0]),
                          np.arange(image_shape[2]))

    if affine is not None:

        coords = np.vstack([x.flatten(),
                            y.flatten(),
                            z.flatten(),
                            np.ones(np.prod(image_shape))])

        # Center image at origin
        coords[0, :] -= image_shape[1] / 2
        coords[1, :] -= image_shape[0] / 2
        coords[2, :] -= image_shape[2] / 2

        coords = np.dot(affine, coords)

        # Put image back in place
        coords[0, :] += image_shape[1] / 2
        coords[1, :] += image_shape[0] / 2
        coords[2, :] += image_shape[2] / 2

        x = np.reshape(coords[0, ...], image_shape)
        y = np.reshape(coords[1, ...], image_shape)
        z = np.reshape(coords[2, ...], image_shape)

    if elastic is not None:
        x += elastic[0]
        y += elastic[1]
        z += elastic[2]

    coords = [y, x, z]  # inverting x,y is necessary for map_coordinates

    if isinstance(image, list):
        if not isinstance(order, list):
            order = [order]*len(image)
        if not isinstance(mode, list):
            mode = [mode]*len(image)
        deformed = []
        for image_, order_, mode_ in zip(image, order, mode):
            deformed.append(map_coordinates(
                                image_,
                                coords,
                                order=order_,
                                mode=mode_).reshape(image_shape))
    else:
        deformed = map_coordinates(image,
                                   coords,
                                   order=order,
                                   mode=mode).reshape(image_shape)

    if LR:  # assumes RAS coordinates
        deformed = [data[::-1, ...] for data in deformed]

    return deformed


def extract_views(data, patch_shape, step, full_image_coverage=True):

    """ Extract views on data """
    """ data: (channels), x, y, z """

    if data.ndim == 3:
        data_shape = data.shape
        window_shape = patch_shape
    elif data.ndim == 4:  # Assumes channels first
        data_shape = data.shape[1:4]
        window_shape = np.hstack((data.shape[0], patch_shape))
    else:
        raise ValueError('Not implemented.')

    views = view_as_windows(data, window_shape)

    # Define patches
    x_range = list(range(0, data_shape[0] - patch_shape[0] + 1, step[0]))
    y_range = list(range(0, data_shape[1] - patch_shape[1] + 1, step[1]))
    z_range = list(range(0, data_shape[2] - patch_shape[2] + 1, step[2]))

    # Make sure we cover the whole image
    if full_image_coverage:
        if x_range[-1] + patch_shape[0] + 1 < data_shape[0]:
            x_range += [data_shape[0] - patch_shape[0] - 1]
        if y_range[-1] + patch_shape[1] + 1 < data_shape[1]:
            y_range += [data_shape[1] - patch_shape[1] - 1]
        if z_range[-1] + patch_shape[2] + 1 < data_shape[2]:
            z_range += [data_shape[2] - patch_shape[2] - 1]

    index = list(product(x_range, y_range, z_range))

    return views, index
