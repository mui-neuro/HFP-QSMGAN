import torch
import numpy as np


class GANModule():

    
    def __init__(
                self,
                generator,
                discriminator,
                batch_size=1,
                gan_mode='dragan',
                critic_iter=5,
                lambda_adv=0.01,
                lambda_gp=100,
                device='cuda'
            ):
        super(GANModule, self).__init__()

        # Send models to GPU
        self.G = generator.to(device)
        self.D = discriminator.to(device)

        # Initialize optimizers
        self.G_opt = torch.optim.Adam(
                                self.G.parameters(),
                                betas=(0.5, 0.999)
                            )

        self.D_opt = torch.optim.Adam(
                                self.D.parameters(),
                                betas=(0.5, 0.999)
                            )

        # Assign parameters
        self.gan_mode = gan_mode
        self.critic_iter = critic_iter
        self.lambda_adv = lambda_adv
        self.lambda_gp = lambda_gp
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng()
        self.reset_losses()

        # Print
        print('\nGAN Parameters:')
        print('GAN mode: ' + self.gan_mode)
        if self.gan_mode == 'wgan-gp':
            print('critic_iter: %i' % self.critic_iter)
        print('lambda_adv: %f' % self.lambda_adv)
        print('lambda_gp: %f' % self.lambda_gp)
        print('batch_size: %i\n' % self.batch_size)


    def optimize_parameters(self):

        self.d_loss_iter = []
        self.gp_loss_iter = []
        self.d_real_iter = []
        self.d_fake_iter = []

        # Discriminator
        if self.gan_mode == 'wgan-gp':
            for ni in range(self.critic_iter):
                self.train_discriminator_iter(randomize=True)
        else:
            self.train_discriminator_iter(randomize=True)

        # Average discriminator losses
        self.d_loss.append(np.mean(self.d_loss_iter))
        self.d_real.append(np.mean(self.d_real_iter))
        self.d_fake.append(np.mean(self.d_fake_iter))
        if self.gan_mode != 'rsgan':
            self.gp_loss.append(np.mean(self.gp_loss_iter))

        # Generator
        self.train_generator_iter()


    def train_discriminator_iter(self, randomize=True):

        # Update training params
        self.D.requires_grad = True
        self.G.eval()  # fixes batchnorm layers

        # Load a random batch
        batch_x, batch_y = self._select_batch(randomize=randomize)
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Generate new data
        y_fake = self.G(batch_x)

        # Calulate discriminator outputs
        d_real = self.D(batch_y)
        d_fake = self.D(y_fake)

        # Calculate gradient penalty
        if self.gan_mode == 'wgan-gp':
            gradient_penalty = self._gradient_penalty(batch_y, y_fake)
        elif self.gan_mode == 'dragan':
            gradient_penalty = self._gradient_penalty(batch_y)

        # Calculate total loss and optimize
        self.D_opt.zero_grad()
        if self.gan_mode in ['wgan-gp', 'dragan']:
            d_loss = d_fake.mean() - d_real.mean() + \
                self.lambda_gp*gradient_penalty
        elif self.gan_mode == 'rsgan':
            d_loss = -torch.log(torch.sigmoid(d_real - d_fake)).mean()
        d_loss.backward()
        self.D_opt.step()

        # Record losses
        self.d_loss_iter.append(d_loss.detach().cpu().numpy())
        self.d_real_iter.append(d_real.mean().detach().cpu().numpy())
        self.d_fake_iter.append(d_fake.mean().detach().cpu().numpy())
        if self.gan_mode in ['wgan-gp', 'dragan']:
            self.gp_loss_iter.append(gradient_penalty.detach().cpu().numpy())

        # Clean up
        del batch_x, batch_y
        torch.cuda.empty_cache()


    def _gradient_penalty(
                    self,
                    real_data,
                    fake_data=None,
                    one_sided=False
                ):

        batch_size = real_data.size()[0]

        # Calculate interpolation
        if self.gan_mode == 'dragan':
            # https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
            beta = torch.rand(real_data.size()).to(self.device)
            fake_data = 0.5 * real_data.std() * beta

        alpha = torch.rand(batch_size, 1, 1, 1, 1).expand_as(real_data).to(self.device)
        differences = real_data - fake_data
        interp = torch.autograd.Variable(
                            real_data + alpha * differences,
                            requires_grad=True)

        # Calculate probability of interpolated examples
        d_interp = self.D(interp)

        # Get gradients
        grad_outputs = torch.ones(d_interp.size()).to(self.device)
        gradients = torch.autograd.grad(
                       outputs=d_interp,
                       inputs=interp,
                       grad_outputs=grad_outputs,
                       create_graph=True,
                       retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Select penalty mode
        if one_sided:
            def clip_fn(x):
                return x.clamp(max=0)
        else:
            def clip_fn(x):
                return x

        # Return gradient penalty
        return ((gradients_norm - 1.) ** 2).mean()


    def train_generator_iter(self):

        # Predict new data
        self.G.train()  # will update batchnorm layers
        batch_x = self.batch_x.to(self.device)
        batch_y = self.batch_y.to(self.device)
        y_fake = self.G(batch_x)

        # Evaluate prediction
        self.D.requires_grads = False  # make sure discriminator is fixed
        d_fake = self.D(y_fake)
        if self.gan_mode == 'rsgan':
            d_real = self.D(batch_y)

        # Reset gradients
        self.G_opt.zero_grad()

        # Adversarial loss
        if self.gan_mode in ['wgan-gp', 'dragan']:
            g_d_loss = -d_fake.mean()
        elif self.gan_mode == 'rsgan':
            g_d_loss = -torch.log(torch.sigmoid(d_fake - d_real)).mean()

        # Consistency loss
        l1_loss = torch.abs(batch_y - y_fake).mean()

        # Total loss
        g_loss = l1_loss + self.lambda_adv * g_d_loss

        # Train network
        g_loss.backward()
        self.G_opt.step()

        # Record losses
        self.g_loss.append(g_loss.detach().cpu().numpy())
        self.g_d_loss.append(g_d_loss.detach().cpu().numpy())
        self.l1_loss.append(l1_loss.detach().cpu().numpy())

        # Clean up
        del batch_x, batch_y
        torch.cuda.empty_cache()


    def update_data_pool(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


    def update_batch_data(self, batch_x, batch_y):
        self.batch_x = batch_x
        self.batch_y = batch_y


    def _select_batch(self, randomize=True):
        if randomize:
            index = self.rng.choice(
                        self.x_data.shape[0],
                        size=self.batch_size,
                        replace=True
                    )
            batch_x = self.x_data[index, ...]
            batch_y = self.y_data[index, ...]
        else:  # assumes data is of correct dimensions
            batch_x = self.x_data
            batch_y = self.y_data

        return batch_x, batch_y


    def reset_losses(self):
        self.g_loss = []
        self.g_d_loss = []
        self.l1_loss = []
        self.d_loss = []
        self.d_real = []
        self.d_fake = []
        self.gp_loss = []
        self.d_loss_iter = []
        self.gp_loss_iter = []
        self.d_real_iter = []
        self.d_fake_iter = []
