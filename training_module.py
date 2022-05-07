import ants
import torch

import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from time import time

try:
    import utils as ut
except:
    import src.utils as ut


class TrainingdModule():

    def __init__(
                self,
                patch_shape,
                step_size,
                crop_size=None,
                initial_learning_rate=1e-4,
                lr_scheduler=None,
                full_image_coverage=True
            ):

        super().__init__()

        self.model = None
        self.rng = np.random.default_rng()  # for randomizing batches
        self.epoch_start = None  # for keep track of time
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Selected device: ' + self.device)

        # Assign learning rate scheduler
        self.initial_learning_rate = initial_learning_rate
        if lr_scheduler is None:
            self.lr_scheduler = get_lr_scheduler(initial_learning_rate)
        else:
            self.lr_scheduler = lr_scheduler

        # Assign patches parameters
        self.patch_shape = np.array(patch_shape)
        self.step_size = np.array(step_size)
        if crop_size is None:
            self.crop_size = None
        else:
            self.crop_size = np.array(crop_size)
        self.full_image_coverage = full_image_coverage

    def fetch_batch(
                self,
                x_dirs,
                y_dir,
                subjects,
                n_epoch=None
            ):

        """ Fetch batch data """

        x = []
        y = []

        if not isinstance(x_dirs, list):
            x_dirs = [x_dirs]

        for subject in subjects:

            # Extract input data
            x_data = []
            for nd, x_dir in enumerate(x_dirs):
                if n_epoch is None:
                    fx = join(x_dir, subject + '.nii.gz')
                    fy = join(y_dir, subject + '.nii.gz')
                else:
                    fx = join(x_dir, 'augment', subject, '%i.nii.gz' % n_epoch)
                    fy = join(y_dir, 'augment', subject, '%i.nii.gz' % n_epoch)

                x_data.append(ants.image_read(fx).numpy())

            # Pad input to compensate for cropping
            # if self.crop_size is not None:
            #     x_data = self._pad_data_list(x_data)

            x.append(self._extract_patches(x_data))

            # Extract output data
            if n_epoch is None:
                fy = join(y_dir, subject + '.nii.gz')
            else:
                fy = join(y_dir, 'augment', subject, '%i.nii.gz' % n_epoch)

            # Read data and extract views
            y_data = [ants.image_read(fy).numpy()]
            y.append(self._extract_patches(y_data))

        # Stack patches across all subjects
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        # Crop output
        if self.crop_size is not None:
            y = self._crop(y)

        # Cast output to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

    def predict_image(self, model, data, overlap=False):

        # Make sure input data is a lst
        if not isinstance(data, list):
            data = [data]

        # Pad data and retrieve patch cropping indexes
        if self.crop_size is not None:
            data = self._pad_data_list(data)
            li, hi = self._get_crop_indices(self.patch_shape)

        # Stack images into array
        data = self._stack_images(data)

        # Allocate outputs
        pred = np.zeros(data.shape[-3:])
        sum_mask = np.zeros(data.shape[-3:])

        # Extract patches
        data_views, index = ut.extract_views(
                                data,
                                self.patch_shape,
                                self.step_size,
                                full_image_coverage=self.full_image_coverage
                            )
        pred_views, _ = ut.extract_views(
                                pred,
                                self.patch_shape,
                                self.step_size,
                                full_image_coverage=self.full_image_coverage
                            )
        sum_mask_views, _ = ut.extract_views(
                                sum_mask,
                                self.patch_shape,
                                self.step_size,
                                full_image_coverage=self.full_image_coverage
                            )

        # Make sure we're in eval mode, fixes batchnorm layers
        model.eval()

        # Compute prediction for each patches
        for xi, yi, zi in index:

            # Move patch to GPU
            patch = torch.from_numpy(
                data_views[:, xi, yi, zi, ...]).float().to(self.device)

            # Predict & clean memory
            with torch.no_grad():
                y = model(patch)
                y_ = y.cpu().numpy()
            del patch, y
            torch.cuda.empty_cache()

            # Assign predicted values and update sum mask
            if self.crop_size is None:
                mask = sum_mask_views[xi, yi, zi, ...] == 0.
                pred_views[xi, yi, zi, mask] += y_[0, 0, mask]
                sum_mask_views[xi, yi, zi, mask] += 1.
            else:  # perform cropping
                mask = np.zeros(self.patch_shape, dtype=bool)
                mask[li[0]:hi[0], li[1]:hi[1], li[2]:hi[2]] = True  # select crop region
                mask[sum_mask_views[xi, yi, zi, ...] != 0.] = False  # remove any voxel already assigned
                cropped_mask = self._crop(mask)
                pred_views[xi, yi, zi, mask] += y_[0, 0, cropped_mask]
                sum_mask_views[xi, yi, zi, mask] += 1.

        # Crop output
        if self.crop_size is not None:
            pred = self._crop(pred)

        return pred

    def evaluate_l1_error(self, model, x, y, img=None, fout=None):

        l1_err = []
        for ni, (x_, y_) in enumerate(zip(x, y)):

            # Predict image & calculate L1 error
            y_pred = self.predict_image(model, x_)
            l1_err.append(np.mean(np.abs(y_ - y_pred)))

            # Save out prediction if requested
            if fout is not None:
                # x_out = fout[ni][:-7] + '_x.nii.gz'
                # ants.image_write(
                #     img[ni][0].new_image_like(x_[0]), x_out)
                ants.image_write(
                    img[ni][0].new_image_like(y_pred), fout[ni])

        return np.mean(l1_err)

    def plot_loss(self, out_dir=None):
        for loss in self.loss_dict.keys():
            plt.plot(self.loss_dict[loss], linewidth=0.5)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            # plt.ylim(0., np.max(self.g_loss))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            fout = join(out_dir, loss + '.png')
            plt.savefig(fout, format='png')
            plt.close()

    def plot_lr_schedule(self, lr_schedule, fout):
        plt.plot(lr_schedule, linewidth=0.5)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        # plt.ylim(0., np.max(self.g_loss))
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.grid(True)
        plt.savefig(fout, format='png')
        plt.close()

    def print_loss(self, n_epoch=None):

        if n_epoch is None:
            loss_str = ''
        else:
            loss_str = 'Epoch %i: ' % n_epoch
        loss_str += ', '.join([key + ': %f' % self.loss_dict[key][-1]
                               for key in self.loss_dict.keys()])

        if self.epoch_start is not None:
            elapsed_time = (time() - self.epoch_start)/60
            loss_str += ', Elapsed time: %0.2f min' % elapsed_time

        print(loss_str)

    def set_lr(self, opt, lr):
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def load_images(self, subjects, batches_x_dir, batches_y_dir):

        imgs = []
        x = []
        y = []
        for subject in subjects:
            fin = [join(in_dir, subject + '.nii.gz')
                   for in_dir in batches_x_dir]
            imgs.append([ants.image_read(fin_) for fin_ in fin])
            x.append([img.numpy() for img in imgs[-1]])
            y.append(ants.image_read(
                join(batches_y_dir, subject + '.nii.gz')).numpy())
        return imgs, x, y

    def memory_usage(self, prefix=''):
        memory_dct = torch.cuda.memory_stats(self.device)
        bytes_per_gb = 1073741824
        used_memory = memory_dct['allocated_bytes.all.current'] / bytes_per_gb
        print(prefix + 'Allocated memory: %f GB' % used_memory)

    def _pad_data_list(self, data):
        # data_ = [copy.deepdopy(d) for d in data]
        lpad = np.floor((self.patch_shape - self.crop_size)/2).astype(int)
        hpad = np.ceil((self.patch_shape - self.crop_size)/2).astype(int)
        return [self._zero_padding_3d(d, lpad, hpad) for d in data]

    def _get_crop_indices(self, data_shape):
        diff = (self.patch_shape - self.crop_size)/2
        li = np.floor(diff).astype(int)
        hi = np.array(np.array(data_shape[-3:]) - np.ceil(diff)).astype(int)
        return li, hi

    def _crop(self, data):
        li, hi = self._get_crop_indices(data.shape)
        return data[..., li[0]:hi[0], li[1]:hi[1], li[2]:hi[2]]

    def _zero_padding_3d(self, input, lpad, hpad):
        shape = input.shape
        input = np.concatenate([
            np.zeros((lpad[0], shape[1], shape[2])),
            input,
            np.zeros((hpad[0], shape[1], shape[2]))], axis=0)
        shape = input.shape
        input = np.concatenate([
            np.zeros((shape[0], lpad[1], shape[2])),
            input,
            np.zeros((shape[0], hpad[1], shape[2]))], axis=1)
        shape = input.shape
        input = np.concatenate([
            np.zeros((shape[0], shape[1], lpad[2])),
            input,
            np.zeros((shape[0], shape[1], hpad[2]))], axis=2)
        return input

    def _stack_images(self, data_list):
        if len(data_list) > 1:
            data = np.stack(data_list, axis=0)
        else:
            data = np.expand_dims(data_list[0], axis=0)
        return data

    def _extract_patches(self, data):
        # Extract views, add to list and concatenate
        data = self._stack_images(data)
        views, index = ut.extract_views(
                            data,
                            self.patch_shape,
                            self.step_size,
                            full_image_coverage=self.full_image_coverage
                        )
        patches = []
        for xi, yi, zi in index:
            patches.append(views[:, xi, yi, zi, ...])
        return np.concatenate(patches, axis=0)


def get_lr_scheduler(
            initial_learning_rate,
            mode='constant',
            start_decay=None,  # Linear decay
            end_learning_rate=None,  # Step decay
            n_steps=None  # Step decay
        ):

    def constant_scheduler(
                n_epoch,
                initial_learning_rate=initial_learning_rate
            ):
        return np.repeat(initial_learning_rate, n_epoch)

    def linear_decay_scheduler(
                n_epoch,
                initial_learning_rate=initial_learning_rate,
                start_decay=start_decay
            ):

        if start_decay is None:
            start_decay = n_epoch // 2

        lr = np.repeat(initial_learning_rate, start_decay)
        lr_decay = np.linspace(
                initial_learning_rate, 0.,
                n_epoch - start_decay + 1
            )[:-1]

        return np.hstack([lr, lr_decay])

    def step_decay_scheduler(
                n_epochs,
                initial_learning_rate=initial_learning_rate,
                end_learning_rate=end_learning_rate,
                n_steps=n_steps
            ):

        step_lr = np.linspace(
                            initial_learning_rate,
                            end_learning_rate,
                            n_steps
                        )

        splits = np.array_split(np.ones(n_epochs), n_steps)

        return np.hstack([split*lr for split, lr in zip(splits, step_lr)])

    if mode == 'constant':
        return constant_scheduler
    if mode == 'linear_decay':
        return linear_decay_scheduler
    if mode == 'step_decay':
        return step_decay_scheduler
