import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

class VIGANModel(BaseModel):
    def name(self):
        return 'VIGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)
        self.AE = networks.define_AE(28*28, 28*28, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.AE, 'AE', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionAE = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A_AE = torch.optim.Adam(self.netD_A.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_AE = torch.optim.Adam(self.netD_B.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE_GA_GB = torch.optim.Adam(
                itertools.chain(self.AE.parameters(), self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.AE)
            print('-----------------------------------------------')

    def set_input(self, images_a, images_b):
        input_A =images_a
        input_B =images_b

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B  = self.netG_A.forward(self.fake_A)

        # Autoencoder loss: fakeA
        self.AEfakeA, AErealB = self.AE.forward(self.fake_A, self.real_B)
        # Autoencoder loss: fakeB
        AErealA, self.AEfakeB = self.AE.forward(self.real_A, self.fake_B)




    #get image pathss
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B =  self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    ############################################################################
    # Define backward function for VIGAN
    ############################################################################

    def backward_AE_pretrain(self):
        # Autoencoder loss
        AErealA, AErealB = self.AE.forward(self.real_A, self.real_B)
        self.loss_AE_pre = self.criterionAE(AErealA, self.real_A) + self.criterionAE(AErealB, self.real_A)
        self.loss_AE_pre.backward()

    def backward_AE(self):

        # fake data
        self.fake_B = self.netG_A.forward(self.real_A)
        self.fake_A = self.netG_B.forward(self.real_B)

        # Autoencoder loss: fakeA
        AEfakeA, AErealB = self.AE.forward(self.fake_A, self.real_B)
        self.loss_AE_fA_rB = (
                             self.criterionAE(AEfakeA, self.real_A) + self.criterionAE(AErealB, self.real_B)) * 1

        # Autoencoder loss: fakeB
        AErealA, AEfakeB = self.AE.forward(self.real_A, self.fake_B)
        self.loss_AE_rA_fB = (
                             self.criterionAE(AErealA, self.real_A) + self.criterionAE(AEfakeB, self.real_B)) * 1

        # combined loss
        self.loss_AE = (self.loss_AE_fA_rB + self.loss_AE_rA_fB) * 0.5
        self.loss_AE.backward()


    # input is vector
    def backward_D_A_AE(self):
        fake_B = self.AEfakeB
        self.loss_D_A_AE = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B_AE(self):
        fake_A = self.AEfakeA
        self.loss_D_B_AE =  self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def backward_AE_GA_GB(self):

        lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D

        # fake data
        # G_A(A)
        self.fake_B = self.netG_A.forward(self.real_A)
        # G_B(B)
        self.fake_A = self.netG_B.forward(self.real_B)

        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A_AE = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B_AE = self.criterionCycle(self.rec_B, self.real_B)

        # Autoencoder loss: fakeA
        self.AEfakeA, AErealB = self.AE.forward(self.fake_A, self.real_B)
        self.loss_AE_fA_rB = (self.criterionAE(self.AEfakeA, self.real_A) + self.criterionAE(AErealB, self.real_B)) * 1

        # Autoencoder loss: fakeB
        AErealA, self.AEfakeB = self.AE.forward(self.real_A, self.fake_B)
        self.loss_AE_rA_fB = (self.criterionAE(AErealA, self.real_A) + self.criterionAE(self.AEfakeB, self.real_B)) * 1
        self.loss_AE = (self.loss_AE_fA_rB + self.loss_AE_rA_fB)

        # D loss
        pred_fake = self.netD_A.forward(self.AEfakeB)
        self.loss_AE_GA = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B.forward(self.AEfakeA)
        self.loss_AE_GB = self.criterionGAN(pred_fake, True)

        self.loss_AE_GA_GB = lambda_C * ( self.loss_AE_GA + self.loss_AE_GB) + \
                             lambda_D * self.loss_AE + 1 * (self.loss_cycle_A_AE + self.loss_cycle_B_AE)
        self.loss_AE_GA_GB.backward()


    #########################################################################################################

    def optimize_parameters_pretrain_cycleGAN(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    ############################################################################
    # Define optimize function for VIGAN
    ############################################################################
    def optimize_parameters_pretrain_AE(self):
        # forward
        self.forward()
        # AE
        self.optimizer_AE.zero_grad()
        self.backward_AE_pretrain()
        self.optimizer_AE.step()

    def optimize_parameters(self):
        # forward
        self.forward()

        # AE+G_A+G_B
        for i in range(2):
            self.optimizer_AE_GA_GB.zero_grad()
            self.backward_AE_GA_GB()
            self.optimizer_AE_GA_GB.step()

        for i in range(1):
            # D_A
            self.optimizer_D_A_AE.zero_grad()
            self.backward_D_A_AE()
            self.optimizer_D_A_AE.step()
            # D_B
            self.optimizer_D_B_AE.zero_grad()
            self.backward_D_B_AE()
            self.optimizer_D_B_AE.step()

    ############################################################################################
    # Get errors for visualization
    ############################################################################################
    def get_current_errors_cycle(self):
        AE_D_A = self.loss_D_A.data[0]
        AE_G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        AE_D_B = self.loss_D_B.data[0]
        AE_G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A),
                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B)])

    def get_current_errors(self):
        D_A = self.loss_D_A_AE.data[0]
        G_A = self.loss_AE_GA.data[0]
        Cyc_A = self.loss_cycle_A_AE.data[0]
        D_B = self.loss_D_B_AE.data[0]
        G_B = self.loss_AE_GB.data[0]
        Cyc_B = self.loss_cycle_B_AE.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A  = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B  = util.tensor2im(self.rec_B.data)

        AE_fake_A = util.tensor2im(self.AEfakeA.view(1,1,28,28).data)
        AE_fake_B = util.tensor2im(self.AEfakeB.view(1,1,28,28).data)


        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A),
                                ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.AE, 'AE', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
