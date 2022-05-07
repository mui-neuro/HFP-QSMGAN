import ants
import torch
import os

import numpy as np
import torch.nn as nn

from time import time
from kornia.augmentation import CenterCrop3D
from os.path import join, isfile
from training_module import TrainingdModule

try:
    import utils as ut
except:
    import src.utils as ut


class UNet3dModule(TrainingdModule):

    
    def __init__(
                self,
                patch_shape,
                step_size,
                crop_size=None,
                initial_learning_rate=2e-4,
                upsampling='trilinear',
                lr_scheduler=None,
                model='unet',
                full_image_coverage=True
            ):

        super().__init__(
                    patch_shape,
                    step_size,
                    crop_size=crop_size,
                    initial_learning_rate=initial_learning_rate,
                    lr_scheduler=lr_scheduler,
                    full_image_coverage=full_image_coverage
                )

        if model == 'unet':
            self.model = UNet3d(
                                crop_size=self.crop_size,
                                upsampling=upsampling
                            ).to(self.device)
            print(self.model)

        self.optimizer = torch.optim.Adam(
                                self.model.parameters(),
                                betas=(0.5, 0.999),
                                lr=self.initial_learning_rate)

        self.loss_fn = torch.nn.L1Loss().to(self.device)

        if self.crop_size is not None:
            self.crop_fn = CenterCrop3D(tuple(self.crop_size))

        print('\nParameters:')
        print('patch_size: ' + str(self.patch_shape).strip('[]'))
        print('step_size: ' + str(self.step_size).strip('[]'))
        if crop_size is not None:
            print('crop_size: ' + str(self.crop_size).strip('[]'))
        print('full_image_coverage: ' + str(self.full_image_coverage))
        if model == 'unet':
            print('upsampling: ' + upsampling)


    def train(
                self,
                model_path,
                training_subjects,
                test_subjects,
                batches_x_dir,
                batches_y_dir,
                n_epochs=1,
                batch_size=1,
                resume=None,
            ):

        # Start or resume training

        start_epoch = None

        if resume is not None:
            fin = None
            if isinstance(resume, int):
                fin = join(model_path, 'checkpoint_%i.pt' % resume)
            elif resume == 'latest':
                files = np.array(os.listdir(model_path))
                if len(files) != 0:
                    files = files[[f.startswith('checkpoint') for f in files]]
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
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.loss_dict = checkpoint['loss_dict']
                self.rng = checkpoint['rng']
                # lr_schedule = checkpoint['lr_schedule']

        if start_epoch is None:
            print('\nStarting training...')
            start_epoch = 0
            self.loss_dict = {
                    'train_loss': [],
                    'train_l1': []
                }
            if test_subjects is not None:
                self.loss_dict['test_loss'] = []
                self.loss_dict['test_l1'] = []

        # Get learning rate schedule
        lr_schedule = self.lr_scheduler(n_epochs)
        self.plot_lr_schedule(
                lr_schedule, join(model_path, 'lr_schedule.png'))

        # Load test batches
        if test_subjects is not None:
            test_x, test_y = self.fetch_batch(
                                batches_x_dir,
                                batches_y_dir,
                                test_subjects
                            )

        # Load images for evaluating L1 error
        train_imgs, train_imgs_x, train_imgs_y = self.load_images(
            training_subjects, batches_x_dir, batches_y_dir
        )

        if test_subjects is not None:
            test_imgs, test_imgs_x, test_imgs_y = self.load_images(
                test_subjects, batches_x_dir, batches_y_dir
            )

        for ne in range(start_epoch, n_epochs):

            self.epoch_start = time()

            # Update learning rate according to schedule
            self.set_lr(self.optimizer, lr_schedule[ne])

            # Load training data for corresponding epoch
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

            # Crop
            if self.crop_size is not None:
                train_y = self.crop_fn(train_y)

            # Train
            self.model.train()
            batch_loss = []
            # for nb in [0]:
            for nb in range(0, n_train, batch_size):

                # print('Batch %i' % nb)

                # Load batches on GPU
                index = np.arange(nb, nb + batch_size)
                index = list(index[index < n_train])
                batch_x = train_x[index, ...].to(self.device)
                batch_y = train_y[index, ...].to(self.device)

                # Throw out previous gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(batch_x)

                # Backward pass & loss
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                batch_loss.append(loss.cpu().detach().numpy())

                # Update model
                self.optimizer.step()

                # Free up memory
                del batch_x, batch_y
                torch.cuda.empty_cache()

            self.loss_dict['train_loss'].append(np.mean(batch_loss))

            # Evaluate loss for test data
            self.model.eval()
            loss = []
            # # self.memory_usage('Before test:')

            if test_subjects is not None:
                for ni in range(test_x.shape[0]):
                    batch_x = test_x[None, ni].to(self.device)
                    batch_y = test_y[None, ni].to(self.device)
                    with torch.no_grad():
                        output = self.model(batch_x)
                        loss_ = self.loss_fn(output, batch_y)
                    loss.append(loss_.cpu().detach().numpy())
                    del batch_x, batch_y
                    torch.cuda.empty_cache()
                # self.memory_usage('After test:')
                self.loss_dict['test_loss'].append(np.mean(loss))

            # Evaluate L1 error and save out test predictions
            l1_error = self.evaluate_l1_error(
                self.model, train_imgs_x, train_imgs_y,
            )
            self.loss_dict['train_l1'].append(l1_error)

            if test_subjects is not None:
                fout = [join(model_path, 'epoch_%i_%s.nii.gz' % (ne + 1, subject))
                        for subject in test_subjects]
                l1_error = self.evaluate_l1_error(
                    self.model, test_imgs_x, test_imgs_y,
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
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss_dict': self.loss_dict,
                        'rng': self.rng,
                        'lr_schedule': lr_schedule
                    }, fout)


    def predict(self, f_checkpoint, subjects, in_dir, out_dir, transform=None):

        # Load model
        print('\nLoading model from checkpoint ' + f_checkpoint)
        checkpoint = torch.load(f_checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        # Evaluate predictions
        for subject in subjects:
            fout = join(out_dir, subject + '.nii.gz')
            if not isfile(fout):
                fin = join(in_dir, subject + '.nii.gz')
                img = ants.image_read(fin)
                pred = self.predict_image(self.model, img.numpy())

                if transform is not None:
                    pred = transform(pred)

                print(fout)
                ants.image_write(img.new_image_like(pred), fout)


class UnetConvBlock3d(nn.Module):
    
    
    def __init__(self, n_filters, in_filters=None):
        super().__init__()

        blocks = []
        if in_filters is None:
            blocks += [ConvBlock3d(n_filters//2, n_filters)]
        else:
            blocks += [ConvBlock3d(in_filters, n_filters)]

        blocks += [ConvBlock3d(n_filters, n_filters)]
        self.conv_block = nn.Sequential(*blocks)


    def forward(self, input):
        return self.conv_block(input)


class ConvBlock3d(nn.Module):
    
    
    def __init__(self, in_filters, out_filters):
        super().__init__()

        blocks = []
        blocks += [nn.Conv3d(
            in_filters, out_filters, kernel_size=3, padding=1)]
        blocks += [nn.BatchNorm3d(num_features=out_filters)]
        blocks += [nn.LeakyReLU(0.2)]
        self.conv_block = nn.Sequential(*blocks)


    def forward(self, input):
        return self.conv_block(input)


""" Models """


class UNet3d(nn.Module):

    def __init__(
                self,
                n_base_filters=16,
                depth=5,
                crop_size=None,
                upsampling='trilinear'
            ):

        super().__init__()

        self.crop_size = crop_size

        # Instntiate model parts
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # Encoder
        for d in range(depth):
            n_filters = n_base_filters*(2**d)
            if d == 0:
                self.encoder += [UnetConvBlock3d(n_filters, in_filters=1)]
            else:
                self.encoder += [UnetConvBlock3d(n_filters)]
            if d < depth-1:  # Continue on to synthesis path
                self.downsampling += [nn.AvgPool3d(2)]

        # Decoder
        for d in range(depth-2, -1, -1):
            n_filters = n_base_filters*(2**d)
            if upsampling == 'trilinear':
                self.upsampling += [nn.Upsample(scale_factor=2, mode='trilinear')]
                self.decoder += [UnetConvBlock3d(
                                            n_filters,
                                            in_filters=n_filters*3)]
            else:
                print('Upsampling %s is not implemented.' % upsampling)

        # Final convolution and activation
        final_blocks = []
        final_blocks += [nn.Conv3d(n_base_filters, 1, kernel_size=1)]
        final_blocks += [nn.Tanh()]

        if crop_size is not None:
            final_blocks += [CenterCrop3D(size=tuple(crop_size))]

        self.final_block = nn.Sequential(*final_blocks)

    def forward(self, input):

        skip_blocks = []

        for ne, encoder in enumerate(self.encoder):

            # print('Encoder: %i' % ne)
            # print(y.shape)

            input = encoder(input)

            if ne < len(self.downsampling):
                skip_blocks += [input]
                input = self.downsampling[ne](input)

        # nd = 0
        for decoder, upsampling, skip_block in \
                zip(self.decoder, self.upsampling, skip_blocks[::-1]):

            # print('Decoder: %i' % nd)
            # print(y.shape)
            input = upsampling(input)
            # print(y.shape)
            # print(skip_block.shape)

            input = torch.cat((input, skip_block), dim=1)
            # print(y.shape)

            input = decoder(input)
            # nd += 1

        return self.final_block(input)