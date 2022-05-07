#! /usr/bin/env python

import os
import argparse
import inspect
import ants

import numpy as np

from os.path import join, abspath, isfile
from importlib import reload
from sklearn.model_selection import KFold

import utils as ut
import augment as aug
from unet_module import UNet3dModule
from unet_gan_module import UNetGan3dModule
from training_module import get_lr_scheduler


reload(ut)
reload(aug)


if __name__ == '__main__':

    main_dir = os.path.dirname(abspath(join(
                               inspect.getfile(inspect.currentframe()),
                               os.pardir)))

    parser = argparse.ArgumentParser(description="HFP-QSMGAN")
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("-cwd", "--cwd", type=str, default=main_dir,
                       help="Current (Root) Working directory.")
    mutex.add_argument("-s", "--subject", type=str, default=None,
                       help="Subject to be processed.")
    mutex.add_argument("-sl", "--subjects_list", type=str, default=None,
                       help="List of all subject to be processed.")
    parser.add_argument("-pi", "--preproc_inputs", action="store_true",
                        help='Preprocess input SW images.')
    parser.add_argument("-res", "--resolution", type=float, default=0.6875,
                        help='Isotropic resolution for preprocessing.')
    parser.add_argument("-po", "--preproc_output", action="store_true",
                        help='Preprocess outputs images (align and normalize).')
    parser.add_argument("-md", "--mask_data", action="store_true",
                        help='Mask images aligned to SWI magnitude.')
    parser.add_argument("-aug", "--augment", action="store_true",
                        help='Augment SWI and QSM data.')
    parser.add_argument("-train_unet", "--train_unet", action="store_true",
                        help='Train vanilla UNET model.')
    parser.add_argument("-train_unet_gan", "--train_unet_gan", action="store_true",
                        help='Train UNet with GAN.')
    parser.add_argument("-predict", "--predict", action="store_true",
                        help='Predict QSM maps from SW phase images.')
    parser.add_argument("-resume", "--resume", type=str, default=None,
                        help='Resume checkpoint (\'latest\' or epoch number)')
    parser.add_argument("-swi_phase_dir", "--swi_phase_dir", type=str, default='swi_phase',
                        help='Path to directory containing SW phase images.')
    parser.add_argument("-swi_mag_dir", "--swi_mag_dir", type=str, default='swi_mag',
                        help='Path to directory containing SW magnitude images.')
    parser.add_argument("-qsm_dir", "--qsm_dir", type=str, default='tgv_qsm',
                        help='Path to directory containing target QSM images.')
    parser.add_argument("-gre_mag_dir", "--gre_mag_dir", type=str, default='gre_mag',
                        help='Path to directory containing magnitude images matching the target QSM images.')
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1,
                        help="Number of parallel jobs. Default 1.")
    parser.add_argument("-n_epochs", "--n_epochs", type=int, default=50,
                        help="Number of epochs. Default 50.")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Level of verbosity. 0: silent, 1: minimal (default), 2: detailed.")
    args = parser.parse_args()


# Post process sujetcs arguments
if args.subject:
    subjects = np.array([args.subject])

if args.subjects_list:
    with open(args.subjects_list, 'r') as f:
        subjects = np.array([subject.strip() for subject in f.readlines()])

# Convert resume option to None, 'latest' or int
if args.resume is None or args.resume == 'latest':
    resume = args.resume
else:
    resume = int(args.resume)


if args.preproc_inputs:

    resolution = [args.resolution,
                  args.resolution,
                  args.resolution]

    print('Preprocessing SW phase images')

    target_shape = (240, 320, 223)

    in_dir = args.swi_phase_dir
    out_dir = join(in_dir, 'preproc')
    ut.assert_dir(out_dir)

    for subject in subjects:

        fin = join(in_dir, subject + '.nii.gz')
        fout = join(out_dir, subject + '.nii.gz')

        if isfile(fin) and not isfile(fout):

            # Load data
            img = ants.image_read(fin)
            data = img.numpy()

            # Scale data
            data /= (4096./100)  # Transform to radians times 100
            img_scaled = img.new_image_like(data)

            # Resample to isotropic resolution
            img_iso = ants.resample_image(
                            img_scaled, resolution, interp_type=4)

            # Adjust final shape across all images
            img_out = ut.adjust_shape(img_iso, target_shape)

            print(fout)
            ants.image_write(img_out, fout)

    print('Preprocessing SW images & magnitude')

    in_dirs = ['swi', args.swi_mag_dir]
    for in_dir in in_dirs:

        out_dir = join(in_dir, 'preproc')
        ut.assert_dir(out_dir)

        for subject in subjects:

            fin = join(in_dir, subject + '.nii.gz')
            fout = join(out_dir, subject + '.nii.gz')

            if isfile(fin) and not isfile(fout):
                img = ants.image_read(fin)
                img_iso = ants.resample_image(
                                img, resolution, interp_type=4)
                img_out = ut.adjust_shape(img_iso, target_shape)
                print(fout)
                ants.image_write(img_out, fout)


if args.preproc_output:

    for TE in [3, 4, 5]:

        qsm_aligned_dir = join(args.qsm_dir, 'TE%i' % TE, 'aligned')
        qsm_scaled_dir = join(qsm_aligned_dir, 'scaled')
        ut.assert_dir(qsm_aligned_dir)
        ut.assert_dir(qsm_scaled_dir)

        for subject in subjects:

            f_qsm_aligned = join(qsm_aligned_dir, subject + '.nii.gz')
            if not isfile(f_qsm_aligned):

                # Align data to GRE magnitude image
                swi_mag = ants.image_read(join(args.swi_mag_dir, 'preproc', subject + '.nii.gz'))
                gre_mag = ants.image_read(join(args.gre_mag_dir, subject, '%i.nii.gz' % TE))

                dct = ants.registration(
                        swi_mag,
                        gre_mag,
                        type_of_transform='Rigid',
                        verbose=True,
                        interp_type=4
                      )

                qsm = ants.image_read(join(args.qsm_dir, 'TE%i' % TE, subject + '.nii.gz'))
                qsm_aligned = ants.apply_transforms(
                                swi_mag,
                                qsm,
                                dct['fwdtransforms'],
                                interpolator='bSpline'
                            )

                print(f_qsm_aligned)
                ants.image_write(qsm_aligned, f_qsm_aligned)

                # Scale data
                f_qsm_scaled = join(qsm_scaled_dir, subject + '.nii.gz')
                qsm_scaled = qsm_aligned.new_image_like(
                                np.tanh(10*qsm_aligned.numpy())
                            )
                print(f_qsm_scaled)
            ants.image_write(qsm_scaled, f_qsm_scaled)


if args.mask_data:

    mask_dir = join(args.swi_mag_dir, 'preproc', 'mask')
    ut.assert_dir(mask_dir)

    for subject in subjects:

        # Create brainmask
        mask_file = join(mask_dir, subject + '_brain_mask.nii.gz')
        if not isfile(mask_file):
            mag_file = join(args.swi_mag_dir, 'preproc', subject + '.nii.gz')
            brain_file = join(mask_dir, subject + '_brain.nii.gz')
            cmd = 'bet %s %s -n -m -R -f 0.1' % (mag_file, brain_file)
            ut.run(cmd)

        mask = ants.image_read(mask_file)
        mask = ants.utils.morphology(
                    mask, 'erode', radius=5, mtype='binary'
               ).numpy()

        in_dirs = [
                join(args.swi_phase_dir, 'preproc'),
                join(args.swi_mag_dir, 'preproc'),
                join('tgv_qsm', 'TE3', 'aligned', 'scaled'),
                join('tgv_qsm', 'TE4', 'aligned', 'scaled'),
                join('tgv_qsm', 'TE5', 'aligned', 'scaled')
            ]

        for in_dir in in_dirs:

            out_dir = join(in_dir, 'masked')
            ut.assert_dir(out_dir)

            fin = join(in_dir, subject + '.nii.gz')
            fout = join(out_dir, subject + '.nii.gz')

            if isfile(fin) and not isfile(fout):
                data = ants.image_read(fin)
                data_masked = data.new_image_like(data.numpy() * mask)
                print(fout)
                ants.image_write(data_masked, fout)


if args.augment:

    data_dirs = [join(args.swi_phase_dir, 'preproc', 'masked'),
                 join('tgv_qsm', 'TE3', 'aligned', 'scaled', 'masked'),
                 join('tgv_qsm', 'TE4', 'aligned', 'scaled', 'masked'),
                 join('tgv_qsm', 'TE5', 'aligned', 'scaled', 'masked')]

    aug.augment_dataset(
        subjects,
        data_dirs,
        n_augment=args.n_epochs,
        n_jobs=args.n_jobs
    )
    

if args.train_unet:

    n_folds = 5
    batch_size = 8

    patch_shape = (96, 96, 96)
    step_size = (32, 32, 32)
    crop_size = (32, 32, 32)

    initial_learning_rate = 1e-5

    lr_scheduler = get_lr_scheduler(
                mode='constant',
                initial_learning_rate=initial_learning_rate,
            )

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    TEs = [3, 4, 5]

    for TE in TEs:

        model = 'unet_TE%i_96_32_1e-5' % TE
        batches_x_dir = [join(args.swi_phase_dir, 'preproc', 'masked')]
        batches_y_dir = join('tgv_qsm', 'TE%i' % TE, 'aligned', 'scaled', 'masked')

        for n_fold, (training, test) in enumerate(folds):

            training_subjects = subjects[training]
            test_subjects = subjects[test]

            print('Training data')
            print(subjects[training])
            print('Test data')
            print(subjects[test])

            # Train models
            model_path = join(main_dir, 'models', model, 'n_fold-%i' % n_fold)
            ut.assert_dir(model_path)

            checkpoint = join(model_path, 'checkpoint_%i.pt' % args.n_epochs)
            if not isfile(checkpoint):

                mdl = UNet3dModule(
                            patch_shape,
                            step_size,
                            crop_size=crop_size,
                            initial_learning_rate=initial_learning_rate,
                            lr_scheduler=lr_scheduler
                        )

                mdl.train(
                        model_path,
                        training_subjects,
                        test_subjects,
                        batches_x_dir,
                        batches_y_dir,
                        n_epochs=args.n_epochs,
                        batch_size=batch_size,
                        resume=resume
                    )

        # Train final model using all available subjects
        model_path = join(main_dir, 'models', model, 'final')
        ut.assert_dir(model_path)

        checkpoint = join(model_path, 'checkpoint_%i.pt' % args.n_epochs)
        if not isfile(checkpoint):

            mdl = UNet3dModule(
                        patch_shape,
                        step_size,
                        crop_size=crop_size,
                        initial_learning_rate=initial_learning_rate,
                        lr_scheduler=lr_scheduler
                    )

            mdl.train(
                    model_path,
                    subjects,
                    None,
                    batches_x_dir,
                    batches_y_dir,
                    n_epochs=args.n_epochs,
                    batch_size=batch_size,
                    resume=resume
                )


if args.train_unet_gan:

    n_folds = 5
    batch_size = 8

    patch_shape = (96, 96, 96)
    step_size = (32, 32, 32)
    crop_size = (32, 32, 32)

    initial_learning_rate = 1e-5

    lr_scheduler = get_lr_scheduler(
                mode='constant',
                initial_learning_rate=initial_learning_rate,
            )

    # GAN parameters
    gan_mode = 'rsgan'
    unet_upsampling = 'trilinear'
    discriminator_batch_norm = False

    # Generate splits for CV
    skf = KFold(n_splits=n_folds, shuffle=False)
    folds = [folds for folds in skf.split(subjects)]

    TEs = [3, 4, 5]

    for TE in TEs:

        model = 'unet_gan_TE%i_96_32_1e-5_rsgan_trilinear' % TE
        batches_x_dir = [join(args.swi_phase_dir, 'preproc', 'masked')]
        batches_y_dir = join('tgv_qsm', 'TE%i' % TE, 'aligned', 'scaled', 'masked')

        for n_fold, (training, test) in enumerate(folds):

            training_subjects = subjects[training]
            test_subjects = subjects[test]

            print('Training data')
            print(subjects[training])
            print('Test data')
            print(subjects[test])

            # Train models
            model_path = join(main_dir, 'models', model, 'n_fold-%i' % n_fold)
            ut.assert_dir(model_path)

            checkpoint = join(model_path, 'checkpoint_%i.pt' % args.n_epochs)
            if not isfile(checkpoint):

                mdl = UNetGan3dModule(
                        patch_shape,
                        step_size,
                        crop_size=crop_size,
                        batch_size=batch_size,
                        gan_mode=gan_mode,
                        lr_scheduler=lr_scheduler,
                        unet_upsampling=unet_upsampling,
                        discriminator_batch_norm=discriminator_batch_norm
                    )


                mdl.train(
                        model_path,
                        training_subjects,
                        test_subjects,
                        batches_x_dir,
                        batches_y_dir,
                        n_epochs=args.n_epochs,
                        batch_size=batch_size,
                        patch_shape=patch_shape,
                        resume=resume
                    )

        # Train final model using all available subjects
        model_path = join(main_dir, 'models', model, 'final')
        ut.assert_dir(model_path)

        checkpoint = join(model_path, 'checkpoint_%i.pt' % args.n_epochs)
        if not isfile(checkpoint):

            mdl = UNetGan3dModule(
                    patch_shape,
                    step_size,
                    crop_size=crop_size,
                    batch_size=batch_size,
                    gan_mode=gan_mode,
                    lr_scheduler=lr_scheduler,
                    unet_upsampling=unet_upsampling,
                    discriminator_batch_norm=discriminator_batch_norm
                )

            mdl.train(
                    model_path,
                    subjects,
                    None,
                    batches_x_dir,
                    batches_y_dir,
                    n_epochs=args.n_epochs,
                    batch_size=batch_size,
                    patch_shape=patch_shape,
                    resume=resume
                )


if args.predict:

    # Model
    for TE in [3, 4, 5]:
        for model in ['unet', 'unet_gan']:

            # Patch parameters
            patch_shape = (96, 96, 96)
            step_size = (32, 32, 32)
            crop_size = (32, 32, 32)

            if model == 'unet':
                model_str = 'unet_TE%i_96_32_1e-5' % TE
                mdl = UNet3dModule(
                        patch_shape,
                        step_size,
                        crop_size=crop_size
                    )

            elif model == 'unet_gan':
                model_str = 'unet_gan_TE%i_96_32_1e-5_rsgan_trilinear' % TE
                mdl = UNetGan3dModule(
                        patch_shape,
                        step_size,
                        crop_size=crop_size
                    )

            model_path = join(main_dir, 'models', model_str, 'final')
            checkpoint = join(model_path, 'checkpoint_%i.pt' % args.n_epochs)

            out_dir = join(args.swi_phase_dir, 'prediction', model_str)
            ut.assert_dir(out_dir)

            def inv_transform(x): return np.arctanh(x)/10

            mdl.predict(
                    checkpoint,
                    subjects,
                    join(args.swi_phase_dir, 'preproc', 'masked'),
                    out_dir,
                    transform=inv_transform
                )
