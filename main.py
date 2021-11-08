from numpy.core.numeric import ones_like
from numpy.lib.twodim_base import triu_indices_from
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from model import Discriminator, Generator
from optimizer import CESP, SimGA, ConOpt, CNCO
import time

    # Hyperparameters etc.
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # lr = 0.001
    # gamma = 0.1
    # alpha = 0.1
    # z_dim = 64
    # img_dim = 784
    # batch_size = 128
    # epochs = 50

    # disc = Discriminator(img_dim).to(device)
    # gen = Generator(z_dim, img_dim).to(device)
    # fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    # transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # )

    # dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # opt_disc = optim.Adagrad(disc.parameters(), lr=lr)
    # opt_gen = optim.Adagrad(gen.parameters(), lr=lr)

    # opts = ['SimGA']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
class GAN(nn.Module):
    def __init__(self, lr=0.001, gamma=0.1, alpha=0.1, z_dim=64, batch_size=128, epochs=50, datasets='mnist', opt='CNCO'):
        self.lr = lr
        self.gamma = gamma
        self.alpha =alpha
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = datasets
        self.opt = opt

    def train(self):
        if self.dataset == 'mnist':
            dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            img_dim = 784
        disc = Discriminator(img_dim).to(self.device)
        gen = Generator(self.z_dim, img_dim).to(self.device)
        fixed_noise = torch.randn((self.batch_size, self.z_dim)).to(device)
        opt_disc = optim.Adagrad(disc.parameters(), lr=self.lr)
        opt_gen = optim.Adagrad(gen.parameters(), lr=self.lr)
        
        try:
            os.makedirs('image_{}'.format(self.opt))
        except FileExistsError:
            print('existed!')
        iter = 0
        
        D_loss_list = []
        G_loss_list = []
        d_grads_list = []
        g_grads_list = []
        nc_d_list = []
        nc_g_list = []
        eigen_d_list = []
        eigen_g_list = []

        for epoch in range(self.epochs+1):
            start = time.time()
            for batch_idx, (real, _) in enumerate(loader):
                real = real.view(-1, 784).to(device)
                batch_size = real.shape[0]

                # Train Discriminator: max log(D(real))+log(1-D(G(z)))
                noise = torch.randn(batch_size, self.z_dim).to(device)
                fake = self.gen(noise)
                # fake_for_CESP = disc(fake)

                disc_real = self.disc(real).view(-1)
                
                d_out_real = self.disc(real)
                d_out_fake = self.disc(fake.detach())

                real_labels = torch.ones(batch_size, 1).to(self.device)                                                          
                d_loss_real = -torch.mean(real_labels * torch.log(d_out_real))                                  
                d_loss_fake = -torch.mean(real_labels * torch.log(1-d_out_fake))                                
                #lossD_real = criterion(disc_real, torch.ones_like(disc_real))

                disc_fake = self.disc(fake).view(-1)
                #lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

                d_loss = d_loss_real + d_loss_fake                                                              
                
                g_loss = -torch.mean(real_labels * torch.log(self.disc(fake)))                                     
                
                opt_disc.zero_grad()
                # opt_gen.zero_grad()

                if self.opt == 'SimGA':
                    opt_disc, opt_gen, grad_d, grad_g, e_d_max, e_g_min = SimGA(disc, gen, d_loss, g_loss, opt_gen, opt_disc, real_labels, d_out_real, d_out_fake)
                elif self.opt == 'ConOpt':
                    opt_disc, opt_gen, grad_d, grad_g, e_d_max, e_g_min = ConOpt(disc, gen, d_loss, g_loss, opt_gen, opt_disc, self.gamma, real_labels, d_out_real, d_out_fake)
                elif self.opt == 'CESP':
                    opt_disc, opt_gen, grad_d, grad_g, nc_step_g, nc_step_d, e_d_max, e_g_min = CESP(disc, gen, d_loss, g_loss, opt_gen, opt_disc, self.alpha, real_labels, d_out_real, d_out_fake)
                elif self.opt == 'CNCO':
                    opt_disc, opt_gen, grad_d, grad_g, nc_step_g, nc_step_d, e_d_max, e_g_min = CNCO(disc, gen, d_loss, g_loss, opt_gen, opt_disc, self.gamma, self.alpha, real_labels, d_out_real, d_out_fake)
                
                d_grads_list.append(np.linalg.norm(grad_d[3].detach().cpu().numpy(), 2))
                g_grads_list.append(np.linalg.norm(grad_g[3].detach().cpu().numpy(), 2))
                D_loss_list.append(d_loss.detach().cpu().numpy())
                G_loss_list.append(g_loss.detach().cpu().numpy())
                eigen_d_list.append(e_d_max.detach().cpu().numpy())
                eigen_g_list.append(e_g_min.detach().cpu().numpy())
                if self.opt == 'CESP' or self.opt == 'CNCO':
                    
                    nc_d_list.append(np.linalg.norm(nc_step_d.detach().cpu().numpy(), 2))
                    nc_g_list.append(np.linalg.norm(nc_step_g.detach().cpu().numpy(), 2))


                if batch_idx % 10 == 0 or batch_idx == int(60000/batch_size)-1:
                    print(
                        "epoch: {}, iter: {}, loss_D: {}, loss_real: {}, loss_fake: {}, loss_G: {}".format(
                            epoch, iter, d_loss, d_loss_real, d_loss_fake, g_loss
                        )
                    )

                    with torch.no_grad():
                        fake = gen(fixed_noise).reshape(-1, 28, 28, 1)
                        img_numpy = (fake.data.cpu().numpy()+1.0)/2.0
                        fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
                        ax = ax.flatten()
                        for j in range(25):
                            ax[j].imshow(img_numpy[j,:,:,0], cmap='gray_r') 
                        plt.savefig('./image_{}/mnist_{}.png'.format(opt,iter))
                        plt.close()

                iter +=1
            end = time.time()
            print('time per epoch:', end - start)

        plt.figure(1)
        self.plot_list(D_loss_list, 'discriminator loss', self.opt, iter)
        plt.figure(2)
        self.plot_list(G_loss_list, 'generator loss', self.opt, iter)
        plt.figure(3)
        self.plot_list(d_grads_list, 'gradient of Discriminator', self.opt, iter)
        plt.figure(4)
        self.plot_list(g_grads_list, 'gradient of Generator', self.opt, iter)
        plt.figure(5)
        self.plot_list(eigen_d_list, 'max eigenvalue of Discriminator', self.opt, iter)
        plt.figure(6)
        self.plot_list(eigen_g_list, 'min eigenvalue of Generator', self.opt, iter)
        
        if self.opt == 'CESP' or self.opt == 'CNCO':
            
            plt.figure(7)
            self.plot_list(nc_d_list, 'negative curvature of Discriminator', self.opt, iter)
            plt.figure(8)
            self.plot_list(nc_g_list, 'negative curvature of Generator', self.opt, iter)

    def plot_list(target_list, name, opt, iter):
        plt.title(name)
        plt.plot([i for i in range(iter)], target_list)
        plt.savefig('image_{}/{}_{}.png'.format(opt, name, opt))
        plt.close()

def main(used_opt):
    gan = GAN(opt=used_opt)
    gan.train()

if __name__ == '__main__':
    main('SimGA')                   # 'SimGA', 'ConOpt', 'CESP', 'CNCO' choose one