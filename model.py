import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import optimizer
import time
import random
import math
from fid import calculate_fretchet

logger = logging.getLogger(__name__)


# Simple GAN
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


class GAN:
    def __init__(self, args, z_dim=100, datasets="mnist"):
        self.args = args
        self.z_dim = z_dim
        self.dataset = datasets

    def train(self):
        logger.info(
            f"lr={self.args.lr}, gamma={self.args.gamma}, alpha={self.args.alpha}, z_dim={self.z_dim}, batch_size={self.args.batch_size}, epochs={self.args.epochs}, dataset={self.dataset}, optimization={self.args.opt}, load_model_path={self.args.load_model_path}, is_fid={self.args.is_fid}, is_eigen={self.args.is_eigen}, generated_num={self.args.generated_num}, fid_range={self.args.fid_range}"
        )
        if self.dataset == "mnist":
            dataset = datasets.MNIST(
                root="dataset/", transform=transforms, download=True
            )
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            img_dim = 784

        disc = Discriminator(img_dim).to(device)
        gen = Generator(self.z_dim, img_dim).to(device)
        fixed_noise = torch.randn(self.args.batch_size, self.z_dim).to(device)
        criterion = nn.BCELoss()
        opt_disc = optim.RMSprop(disc.parameters(), lr=self.args.lr)
        opt_gen = optim.RMSprop(gen.parameters(), lr=self.args.lr)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx])
        model = model.to(device)
        if self.args.load_model_path != "":
            checkpoint = torch.load(self.args.load_model_path)
            disc.load_state_dict(checkpoint["Discriminator_state_dict"])
            gen.load_state_dict(checkpoint["Generator_state_dict"])
            start_epoch = checkpoint["epoch"]
            opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
            opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
            D_loss_list = checkpoint["D_loss_list"]
            G_loss_list = checkpoint["G_loss_list"]
            d_grads_list = checkpoint["d_grads_list"]
            g_grads_list = checkpoint["g_grads_list"]
            nc_d_list = checkpoint["nc_d_list"]
            nc_g_list = checkpoint["nc_g_list"]
            eigen_d_list = checkpoint["eigen_d_list"]
            eigen_g_list = checkpoint["eigen_g_list"]
            fid_list = checkpoint["fid_list"]
            check_list = checkpoint["check_list"]
            fid_score = fid_list[len(fid_list) - 1]
        else:
            start_epoch = -1  # it means start from epoch=0
            D_loss_list = []
            G_loss_list = []
            d_grads_list = []
            g_grads_list = []
            nc_d_list = []
            nc_g_list = []
            eigen_d_list = []
            eigen_g_list = []
            fid_list = []
            check_list = []

        try:
            os.makedirs(
                "image_{}_gamma{}_alpha{}".format(
                    self.args.opt, self.args.gamma, self.args.alpha
                )
            )
        except FileExistsError:
            print("existed!")

        try:
            os.makedirs(
                "image_{}_gamma{}_alpha{}/nc_step_greater_than_zero".format(
                    self.args.opt, self.args.gamma, self.args.alpha
                )
            )
        except FileExistsError:
            print("existed!")

        try:
            os.makedirs(
                "image_{}_gamma{}_alpha{}/model".format(
                    self.args.opt, self.args.gamma, self.args.alpha
                )
            )
        except FileExistsError:
            print("existed!")

        iter = (start_epoch + 1) * int(math.ceil(60000 / self.args.batch_size))
        fid_count = len(fid_list)

        logger.info("Start training")
        for epoch in range(start_epoch + 1, self.args.epochs + 1):
            start = time.time()
            for batch_idx, (real, _) in enumerate(loader):
                start_iter = time.time()
                real = real.view(-1, img_dim).to(device)

                batch_size = real.shape[
                    0
                ]  # Since the end of the epoch iteration size <= self.args.batch_size

                # Train Discriminator: max log(D(real))+log(1-D(G(z)))
                # Train Generator: max log(1-D(G(z)))
                noise = torch.randn(batch_size, self.z_dim).to(device)
                fake = gen(noise)

                d_out_real = disc(real)
                d_out_fake = disc(fake).view(-1)

                d_loss_real = criterion(
                    d_out_real, torch.ones_like(d_out_real)
                ).mean()  # -torch.mean(real_labels * torch.log(d_out_real))
                d_loss_fake = criterion(
                    d_out_fake, torch.zeros_like(d_out_fake)
                ).mean()  # -torch.mean(real_labels * torch.log(1-d_out_fake))

                disc_fake = disc(fake).view(-1)

                d_loss = d_loss_real + d_loss_fake

                g_loss = criterion(
                    disc_fake, torch.ones_like(disc_fake)
                ).mean()  # -torch.mean(real_labels * torch.log(disc_fake))

                opt_disc.zero_grad()
                opt_gen.zero_grad()

                # Check is G and D follow a zero-sum game?
                check_list.append((d_loss - g_loss).item())

                if self.args.opt == "SimGA":
                    #######################################################
                    #   #####  #####  ###    ###    #######      ##      #
                    #   #        #    #  #  #  #   ##           #  #     #
                    #   #####    #    #    #   #  ##   ####    #    #    #
                    #       #    #    #        #   ##     #   ########   #
                    #   #####  #####  #        #     ######  #        #  #
                    #######################################################
                    (
                        opt_disc,
                        opt_gen,
                        nc_step_g,
                        nc_step_d,
                        e_d_max,
                        e_g_min,
                        flatten_grad_d,
                        flatten_grad_g,
                    ) = optimizer.SimGA(
                        disc,
                        gen,
                        d_loss,
                        g_loss,
                        opt_gen,
                        opt_disc,
                        self.args.gamma,
                        self.args.alpha,
                        self.args,
                    )
                elif self.args.opt == "ConOpt":
                    ###########################################################
                    #     ###      ###     ##    #    ###     #####   #####   #
                    #    #   #    #   #    # #   #   #   #    #    #    #     #
                    #   #        #     #   #  #  #  #     #   #####     #     #
                    #    #   #    #   #    #   # #   #   #    #         #     #
                    #     ###      ###     #    ##    ###     #         #     #
                    ###########################################################
                    (
                        opt_disc,
                        opt_gen,
                        nc_step_g,
                        nc_step_d,
                        e_d_max,
                        e_g_min,
                        flatten_grad_d,
                        flatten_grad_g,
                    ) = optimizer.ConOpt(
                        disc,
                        gen,
                        d_loss,
                        g_loss,
                        opt_gen,
                        opt_disc,
                        self.args.gamma,
                        self.args.alpha,
                        self.args,
                    )
                elif self.args.opt == "CESP":
                    #######################################
                    #     ###    #####   #####   #####    #
                    #    #   #   #       #       #    #   #
                    #   #        #####   #####   #####    #
                    #    #   #   #           #   #        #
                    #     ###    #####   #####   #        #
                    #######################################
                    (
                        opt_disc,
                        opt_gen,
                        nc_step_g,
                        nc_step_d,
                        e_d_max,
                        e_g_min,
                        flatten_grad_d,
                        flatten_grad_g,
                    ) = optimizer.CESP(
                        disc,
                        gen,
                        d_loss,
                        g_loss,
                        opt_gen,
                        opt_disc,
                        self.args.gamma,
                        self.args.alpha,
                        self.args,
                    )
                elif self.args.opt == "CNCO":
                    ########################################
                    #     ###    ##    #    ###     ###    #
                    #    #   #   # #   #   #   #   #   #   #
                    #   #        #  #  #  #       #     #  #
                    #    #   #   #   # #   #   #   #   #   #
                    #     ###    #    ##    ###     ###    #
                    ########################################
                    (
                        opt_disc,
                        opt_gen,
                        nc_step_g,
                        nc_step_d,
                        e_d_max,
                        e_g_min,
                        flatten_grad_d,
                        flatten_grad_g,
                    ) = optimizer.CNCO(
                        disc,
                        gen,
                        d_loss,
                        g_loss,
                        opt_gen,
                        opt_disc,
                        self.args.gamma,
                        self.args.alpha,
                        self.args,
                    )

                nc_norm_d = torch.linalg.norm(nc_step_d, 2).detach().cpu()
                nc_norm_g = torch.linalg.norm(nc_step_g, 2).detach().cpu()

                d_grads_list.append(torch.linalg.norm(flatten_grad_d, 2).detach().cpu())
                g_grads_list.append(torch.linalg.norm(flatten_grad_g, 2).detach().cpu())
                D_loss_list.append(d_loss.detach().cpu().numpy())
                G_loss_list.append(g_loss.detach().cpu().numpy())
                eigen_d_list.append(e_d_max.detach().cpu().numpy())
                eigen_g_list.append(e_g_min.detach().cpu().numpy())
                nc_d_list.append(nc_norm_d)
                nc_g_list.append(nc_norm_g)

                x = random.sample(range(batch_size), self.args.generated_num)
                if iter % self.args.fid_range == 0:
                    if self.args.is_fid == 1:
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        real_expand = torch.Tensor().to(device)
                        fake_expand = torch.Tensor().to(device)
                        for i in x:
                            r = real[i].view(28, 28)
                            f = fake[i].detach().view(28, 28)
                            r.unsqueeze_(0)
                            f.unsqueeze_(0)
                            r = r.repeat(3, 1, 1)
                            f = f.repeat(3, 1, 1)
                            real_expand = torch.cat((real_expand, r))
                            fake_expand = torch.cat((fake_expand, f))

                        real_expand = real_expand.view(len(x), 3, 28, 28)
                        fake_expand = fake_expand.view(len(x), 3, 28, 28)

                        fid_score = calculate_fretchet(real_expand, fake_expand, model)
                        fid_list.append(fid_score)
                        fid_count += 1

                    else:
                        fid_list.append(0)

                real_range = [0, 1, 2, 3, 4]
                fake_range = [5, 6, 7, 8, 9]
                for i in range(10, self.args.generated_num * 2):
                    if i % 10 <= 4:
                        real_range.append(i)
                    else:
                        fake_range.append(i)

                if nc_norm_d > 0 or nc_norm_g > 0:
                    fake = gen(fixed_noise).detach().reshape(-1, 28, 28, 1).to(device)
                    real = real.reshape(-1, 28, 28, 1)
                    img_numpy = (fake.data.cpu().numpy() + 1.0) / 2.0
                    img_real_numpy = (real.data.cpu().numpy() + 1.0) / 2.0
                    _, ax = plt.subplots(
                        nrows=int(self.args.generated_num / 5.0),
                        ncols=10,
                        sharex="all",
                        sharey="all",
                    )
                    ax = ax.flatten()
                    for j, k in zip(range(len(x + x)), x + x):
                        if j in fake_range:
                            ax[j].imshow(img_numpy[k, :, :, 0], cmap="gray_r")
                            ax[j].axis("off")
                        elif j in real_range:
                            ax[j].imshow(img_real_numpy[k, :, :, 0], cmap="gray_r")
                            ax[j].axis("off")

                    plt.savefig(
                        "./image_{}_gamma{}_alpha{}/nc_step_greater_than_zero/mnist_{}.png".format(
                            self.args.opt, self.args.gamma, self.args.alpha, iter
                        )
                    )
                    plt.close()

                if batch_idx == int(60000 / self.args.batch_size) - 1 or batch_idx == 0:
                    print(
                        "epoch: {}, iter: {}, loss_D: {}, loss_real: {}, loss_fake: {}, loss_G: {}".format(
                            epoch, iter, d_loss, d_loss_real, d_loss_fake, g_loss
                        )
                    )

                    with torch.no_grad():
                        fake = gen(fixed_noise).reshape(-1, 28, 28, 1)
                        real = real.reshape(-1, 28, 28, 1)
                        img_numpy = (fake.data.cpu().numpy() + 1.0) / 2.0
                        img_real_numpy = (real.data.cpu().numpy() + 0.1) / 2.0
                        _, ax = plt.subplots(
                            nrows=int(self.args.generated_num / 5.0),
                            ncols=10,
                            sharex="all",
                            sharey="all",
                        )
                        ax = ax.flatten()
                        for j, k in zip(range(len(x + x)), x + x):
                            if j in fake_range:
                                ax[j].imshow(img_numpy[k, :, :, 0], cmap="gray_r")
                                ax[j].axis("off")
                            elif j in real_range:
                                ax[j].imshow(img_real_numpy[k, :, :, 0], cmap="gray_r")
                                ax[j].axis("off")
                        plt.savefig(
                            "./image_{}_gamma{}_alpha{}/mnist_{}_{}.png".format(
                                self.args.opt,
                                self.args.gamma,
                                self.args.alpha,
                                iter,
                                self.args.opt,
                            )
                        )
                        plt.close()

                    torch.save(
                        {
                            "Discriminator_state_dict": disc.state_dict(),
                            "Generator_state_dict": gen.state_dict(),
                            "opt_disc_state_dict": opt_disc.state_dict(),
                            "opt_gen_state_dict": opt_gen.state_dict(),
                            "epoch": epoch,
                            "lr": self.args.lr,
                            "alpha": self.args.alpha,
                            "gamma": self.args.gamma,
                            "D_loss_list": D_loss_list,
                            "G_loss_list": G_loss_list,
                            "d_grads_list": d_grads_list,
                            "g_grads_list": g_grads_list,
                            "nc_d_list": nc_d_list,
                            "nc_g_list": nc_g_list,
                            "eigen_d_list": eigen_d_list,
                            "eigen_g_list": eigen_g_list,
                            "fid_list": fid_list,
                            "check_list": check_list,
                        },
                        "./image_{}_gamma{}_alpha{}/model/save{}_{}.pt".format(
                            self.args.opt,
                            self.args.gamma,
                            self.args.alpha,
                            self.args.opt,
                            epoch,
                        ),
                    )

                end_iter = time.time()
                if self.args.is_fid == 1:
                    print(
                        "time iter : {}".format(iter),
                        end_iter - start_iter,
                        "sec,",
                        "fid score:{}".format(fid_score),
                    )
                else:
                    print("time iter : {}".format(iter), end_iter - start_iter, "sec")
                iter += 1

            logger.info(f"{epoch}: Training Success!")

            end = time.time()
            print("time of epoch {}:".format(epoch), end - start)

        plt.figure(1)
        self.plot_list(D_loss_list, "discriminator_loss", self.args.opt, iter)
        plt.figure(2)
        self.plot_list(G_loss_list, "generator_loss", self.args.opt, iter)
        plt.figure(3)
        self.plot_list(d_grads_list, "gradient_of_Discriminator", self.args.opt, iter)
        plt.figure(4)
        self.plot_list(g_grads_list, "gradient_of_Generator", self.args.opt, iter)
        if self.args.is_eigen == 1:
            plt.figure(5)
            self.plot_list(
                eigen_d_list, "max_eigenvalue_of_Discriminator", self.args.opt, iter
            )
            plt.figure(6)
            self.plot_list(
                eigen_g_list, "min_eigenvalue_of_Generator", self.args.opt, iter
            )
            plt.figure(7)
            self.plot_list(
                nc_d_list, "negative_curvature_of_Discriminator", self.args.opt, iter
            )
            plt.figure(8)
            self.plot_list(
                nc_g_list, "negative_curvature_of_Generator", self.args.opt, iter
            )
        plt.figure(9)
        self.plot_list(check_list, "zero-sum_game_check", self.args.opt, iter)
        if self.args.is_fid == 1:
            plt.figure(10)
            self.plot_list(fid_list, "fid_score", self.args.opt, fid_count)
        if device == "cuda":
            torch.cuda.empty_cache()

    def plot_list(self, target_list, name, opt, iter):
        plt.title(opt + " " + name)
        plt.plot([i for i in range(iter)], target_list)
        plt.savefig(
            "image_{}_gamma{}_alpha{}/{}_{}.png".format(
                opt, self.args.gamma, self.args.alpha, name, opt
            )
        )
        plt.close()


# The InceptionV3 model is given by https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid?fbclid=IwAR1imIsSuiU30ujfIr0EQl2pFpG4iBAw9FrHkq2kbVYLWiO7EUBZBppRfxM
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=[DEFAULT_BLOCK_INDEX],
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
    ):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
