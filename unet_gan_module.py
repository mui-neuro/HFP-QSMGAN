import ants
import torch
import os
import time

import numpy as np
import torch.nn as nn

from kornia.augmentation import CenterCrop3D
from os.path import join, isfile

from gan_module import GANModule
from unet_module import UNet3d
from training_module import TrainingdModule


try:
    import utils as ut
except:
    import src.utils as ut


class Discriminator(nn.Module):

    
    def __init__(
                self,
                output_kernel,
                depth=4,
                n_base_filters=8,
                batch_norm=False
            ):

        super().__init__()

        blocks = []
        for d in range(depth):
            n_filters = n_base_filters*(2**d)
            if d == 0:
                in_filters = 1
                batch_norm_ = False
            else:
                in_filters = n_filters//2
                batch_norm_ = batch_norm

            blocks += [nn.Conv3d(
                            in_filters, n_filters,
                            kernel_size=4,
                            stride=2,
                            padding=1
                        )]
            if batch_norm_:
                blocks += [nn.BatchNorm3d(num_features=n_filters)]
            blocks += [nn.LeakyReLU(0.2)]

        blocks += [nn.Conv3d(
                        n_filters, 1,
                        kernel_size=output_kernel,
                        padding=0
                    )]

        self.model = nn.Sequential(*blocks)


    def forward(self, input):
        return self.model(input)


class UNetGan3dModule(TrainingdModule):

    
    def __init__(
                self,
                patch_shape,
                step_size,
                crop_size=None,
                batch_size=8,
                initial_learning_rate=2e-4,
                lr_scheduler=None,
                gan_mode='rsgan',
                lambda_adv=0.01,
                discriminator_depth=4,
                unet_upsampling='trilinear',
                discriminator_batch_norm=False,
                generator='unet'
            ):

        super().__init__(
                patch_shape,
                step_size,
                crop_size=crop_size,
                initial_learning_rate=initial_learning_rate,
                lr_scheduler=lr_scheduler
            )

        if crop_size is None:
            discriminator_output_kernel = \
                tuple(np.array(patch_shape) // 2**(discriminator_depth))
        else:
            discriminator_output_kernel = \
                tuple(np.array(crop_size) // 2**(discriminator_depth))

        # Select generator
        if generator == 'unet':
            G = UNet3d(crop_size=self.crop_size,
                       upsampling=unet_upsampling)
        else:
            raise ValueError('Invalid generator ' + generator)

        self.model = GANModule(
                generator=G,
                discriminator=Discriminator(
                            depth=discriminator_depth,
                            output_kernel=discriminator_output_kernel,
                            batch_norm=discriminator_batch_norm),
                gan_mode=gan_mode,
                lambda_adv=lambda_adv,
                batch_size=batch_size,
                device=self.device
            )

        print(self.model.G)
        print(self.model.D)

        if self.crop_size is not None:
            self.crop_fn = CenterCrop3D(tuple(self.crop_size))


    def train(
                self,
                model_path,
                training_subjects,
                test_subjects,
                batches_x_dir,
                batches_y_dir,
                n_epochs=1,
                batch_size=1,
                patch_shape=None,
                n_pre_train=None,
                pre_train_lr=1e-4,
                pre_trained_G=None,
                resume=None,
            ):

        if pre_trained_G is not None:
            print('Loading pre-trained generator from ' + pre_trained_G)
            checkpoint = torch.load(pre_trained_G)
            self.model.G.load_state_dict(checkpoint['model'])
            self.model.G_opt.load_state_dict(checkpoint['optimizer'])

        self.loss_dict = {}

        # Pre-train discriminator
        pre_trained_D = join(model_path, 'pre_trained_D.pt')
        if isfile(pre_trained_D):
            print('\nLoading pre-trained discriminator')
            checkpoint = torch.load(pre_trained_D)
            self.model.D.load_state_dict(checkpoint['D'])
            self.model.D_opt.load_state_dict(checkpoint['D_opt'])
        elif n_pre_train is not None:

            print('\nPre-training discriminator')
            print('Learning rate: %f' % pre_train_lr)

            self.set_lr(self.model.D_opt, pre_train_lr)

            for ne in range(n_pre_train):

                # Load training batches
                train_x, train_y = self.fetch_batch(
                                        batches_x_dir,
                                        batches_y_dir,
                                        training_subjects,
                                        n_epoch=ne
                                    )

                n_train = len(train_x)

                # Randomize patches & cast to tensors
                index = self.rng.permutation(n_train)
                train_x = train_x[index, ...]
                train_y = train_y[index, ...]

                # Train
                for nb in range(0, n_train, batch_size):

                    # Load batches on GPU
                    index = np.arange(nb, nb + batch_size)
                    index = list(index[index < n_train])
                    batch_x = train_x[index, ...]
                    batch_y = train_y[index, ...]

                    # Update batch and optimize model
                    self.model.update_data_pool(batch_x, batch_y)
                    self.model.train_discriminator_iter(randomize=False)

                    # Record losses, print & plot them
                    self.loss_dict['d'] = self.model.d_loss_iter
                    self.loss_dict['d_real'] = self.model.d_real_iter
                    self.loss_dict['d_fake'] = self.model.d_fake_iter
                    if self.model.gan_mode != 'rsgan':
                        self.loss_dict['gp'] = self.model.gp_loss_iter
                    self.plot_loss(out_dir=model_path)
                    self.print_loss(ne + 1)

            # Save out pre-trained discriminator
            torch.save({
                    'D': self.model.D.state_dict(),
                    'D_opt': self.model.D_opt.state_dict(),
                    'loss_dict': self.loss_dict
                }, pre_trained_D)

        # Start or resume training

        start_epoch = None

        if resume is not None:
            fin = None
            if isinstance(resume, int):
                fin = join(model_path, 'checkpoint_%i.pt' % resume)
            elif resume == 'latest':
                files = np.array(os.listdir(model_path))
                files = files[[f.startswith('checkpoint') for f in files]]
                if len(files) != 0:
                    id = np.max([int(f[11:-3]) for f in files])
                    if id < n_epochs:
                        fin = join(model_path, 'checkpoint_%i.pt' % id)
                    else:
                        return
            else:
                print(resume)
                raise ValueError('Invalid resume mode.')
            if fin is not None:
                print('Resume training from ' + fin)
                checkpoint = torch.load(fin)
                start_epoch = checkpoint['n_train']
                self.model.G.load_state_dict(checkpoint['G'])
                self.model.D.load_state_dict(checkpoint['D'])
                self.model.G_opt.load_state_dict(checkpoint['G_opt'])
                self.model.D_opt.load_state_dict(checkpoint['D_opt'])
                self.loss_dict = checkpoint['loss_dict']
                self.rng = checkpoint['rng']
                # lr_schedule = checkpoint['lr_schedule']

        if start_epoch is None:
            start_epoch = 0
            self.loss_dict = {
                    'g': [],
                    'g_d': [],
                    'l1': [],
                    'd': [],
                    'd_real': [],
                    'd_fake': [],
                    'train_l1': []
                }
            if test_subjects is not None:
                self.loss_dict['test_l1'] = []

            if self.model.gan_mode != 'rsgan':
                self.loss_dict['gp'] = []

        # Get learning rate schedule
        lr_schedule = self.lr_scheduler(n_epochs)
        self.plot_lr_schedule(
                lr_schedule, join(model_path, 'lr_schedule.png'))

        # Train GAN

        print('\nStarting GAN training')

        # Get learning rate schedule
        lr_schedule = self.lr_scheduler(n_epochs)
        self.plot_lr_schedule(
                lr_schedule, join(model_path, 'lr_schedule.png'))

        # Load images for evaluating L1 error
        train_imgs, train_imgs_x, train_imgs_y = self.load_images(
            training_subjects, batches_x_dir, batches_y_dir
        )

        if test_subjects is not None:
            test_imgs, test_imgs_x, test_imgs_y = self.load_images(
                test_subjects, batches_x_dir, batches_y_dir
            )

        for ne in range(start_epoch, n_epochs):

            self.epoch_start = time.time()

            # Update learning rate according to schedule
            self.set_lr(self.model.D_opt, lr_schedule[ne])
            self.set_lr(self.model.G_opt, lr_schedule[ne])

            # Load training batches
            train_x, train_y = self.fetch_batch(
                                    batches_x_dir,
                                    batches_y_dir,
                                    training_subjects,
                                    n_epoch=ne
                                )

            n_train = len(train_x)
            # print('Total batches: %i' % n_train)

            # Randomize patches & cast to tensors
            index = self.rng.permutation(n_train)
            train_x = train_x[index, ...]
            train_y = train_y[index, ...]

            self.model.update_data_pool(train_x, train_y)

            # Train
            # for nb in [0]:
            self.model.G.train()
            for nb in range(0, n_train, batch_size):

                # Define batch
                index = np.arange(nb, nb + batch_size)
                index = list(index[index < n_train])
                batch_x = train_x[index, ...].to(self.device)
                batch_y = train_y[index, ...].to(self.device)

                # Update batch and optimize model
                self.model.update_batch_data(batch_x, batch_y)
                self.model.optimize_parameters()

            # Record losses
            self.loss_dict['g'].append(np.mean(self.model.g_loss))
            self.loss_dict['g_d'].append(np.mean(self.model.g_d_loss))
            self.loss_dict['l1'].append(np.mean(self.model.l1_loss))
            self.loss_dict['d'].append(np.mean(self.model.d_loss))
            self.loss_dict['d_real'].append(np.mean(self.model.d_real))
            self.loss_dict['d_fake'].append(np.mean(self.model.d_fake))
            if self.model.gan_mode != 'rsgan':
                self.loss_dict['gp'].append(np.mean(self.model.gp_loss))
            self.model.reset_losses()

            # Evaluate L1 error and save out test predictions
            self.model.G.eval()
            l1_error = self.evaluate_l1_error(
                self.model.G, train_imgs_x, train_imgs_y,
            )
            self.loss_dict['train_l1'].append(l1_error)

            if test_subjects is not None:
                fout = [join(model_path, 'epoch_%i_%s.nii.gz' % (ne + 1, subject))
                        for subject in test_subjects]
                l1_error = self.evaluate_l1_error(
                    self.model.G, test_imgs_x, test_imgs_y,
                    img=test_imgs, fout=fout
                )
                self.loss_dict['test_l1'].append(l1_error)

            # Plot & print losses
            self.plot_loss(out_dir=model_path)
            self.print_loss(ne + 1)

            # Checkpoint model, every 10 iterations
            if (ne + 1) % 10 == 0:
                fout = join(model_path, 'checkpoint_%i.pt' % (ne + 1))
                torch.save({
                        'n_train': ne + 1,
                        'G': self.model.G.state_dict(),
                        'D': self.model.D.state_dict(),
                        'G_opt': self.model.G_opt.state_dict(),
                        'D_opt': self.model.D_opt.state_dict(),
                        'rng': self.model.rng,
                        'loss_dict': self.loss_dict,
                        # 'lr_schedule': lr_schedule
                    }, fout)


    def predict(self, f_checkpoint, subjects, in_dir, out_dir, transform=None):

        # Load model
        print('\nLoading model from checkpoint ' + f_checkpoint)
        checkpoint = torch.load(f_checkpoint)
        self.model.G.load_state_dict(checkpoint['G'])
        self.model.G.eval()

        # Evaluate predictions
        for subject in subjects:
            fout = join(out_dir, subject + '.nii.gz')
            if not isfile(fout):
                fin = join(in_dir, subject + '.nii.gz')
                img = ants.image_read(fin)
                pred = self.predict_image(self.model.G, img.numpy())

                if transform is not None:
                    pred = transform(pred)

                print(fout)
                ants.image_write(img.new_image_like(pred), fout)
