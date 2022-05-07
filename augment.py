import ants
import faulthandler

import numpy as np

from multiprocessing import Pool
from itertools import product
from os.path import isfile, join
from importlib import reload

try:
    import utils as ut
except:
    import src.utils as ut

reload(ut)

faulthandler.enable()


def augment_data(args, n_augment=None, elastic=True, affine=True):

    """ Augment image using nonlinear warping and left/right flip """

    subjects, data_dirs = args

    for ns, subject in enumerate(subjects):

        # Figure out which image to load
        fout = [join(data_dir, 'augment', subject, '%i.nii.gz' % n_augment)
                for data_dir in data_dirs]

        if not np.all([isfile(fout_) for fout_ in fout]):

            # Load data
            img = [ants.image_read(join(data_dir, subject + '.nii.gz'))
                   for data_dir in data_dirs]
            data = [data_.numpy() for data_ in img]

            if n_augment is None:
                data_out = data
            else:

                seed = n_augment*len(subjects) + ns
                rng = np.random.default_rng(seed)
                M = ut.random_affine(rng=rng)
                E = ut.random_elastic_transform(
                        data[0].shape, 500, 10, rng=rng)
                LR = rng.choice([True, False])

                order = [3]*len(data)  # cubic interpolation

                data_out = ut.apply_transforms(
                                    data,
                                    affine=M,
                                    elastic=E,
                                    LR=LR,
                                    order=order
                                )

            # Save
            for data_, img_, fout_ \
                    in zip(data_out, img, fout):
                img_out = img_.new_image_like(data_)
                print(fout_)
                ants.image_write(img_out, fout_)


def augment_dataset(subjects,
                    data_dirs,
                    n_augment=None,
                    n_jobs=1):

    """ Wrapper for augment_data """

    print('Augmenting dataset')

    # Assert output directories
    for data_dir in data_dirs:
        for subject in subjects:
            ut.assert_dir(join(data_dir, 'augment', subject))

    if n_augment is None:
        augment_data([subjects, data_dirs, None])
    else:
        params = list(product([[subjects, data_dirs]],
                              range(n_augment)))

        with Pool(processes=n_jobs) as pool:
            pool.starmap(augment_data, params)
