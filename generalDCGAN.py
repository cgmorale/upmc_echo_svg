import argparse
import os
import logging
import random
import pickle

import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision.transforms as xforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import ttools #checkpointer

#https://github.com/BachiLi/diffvg
import pydiffvg

from xml.dom import minidom
import svgpathtools

# parameters
canvas_width = 64
canvas_height = 64
latent_dim = 10

# Use GPU if available
pydiffvg.render_pytorch.print_timing = False
pydiffvg.set_use_gpu(torch.cuda.is_available())


class CustomImageDataset(Dataset):
    ''' loads and divvies out images '''
    def __init__(self, path, transform=None, target_transform=None):
        super(CustomImageDataset, self).__init__()
        
        self.transform = transform
        self.target_transform = target_transform
    
        # load data in pickle file
        #    data is a iterable of numpy arrays or similar in WHC format
        self.data = pickle.load(open(path,'rb'))
        
        if transform is not None:
            self.data = [transform(x) for x in self.data] 
        else:
            self.data = [x for x in self.data] 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def weights_init_normal(m):
    ''' simple weight initializer'''
    
    classname = m.__class__.__name__
    
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_eye(m):
    ''' simple weight initializer'''

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.constant_(m.weight.data,0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.eye_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(torch.nn.Module):
    ''' generator network '''
    def __init__(self,svg_template='smiley64.svg',canvas_width=64,canvas_height=64):
        super(Generator, self).__init__()

        #TODO: make sure architecture is ok
        hidden_dim = 32
        self.shape_transformer =  torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d( latent_dim, 64 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, z,device=None):
        if device is None:
            device = torch.device("cpu")
            
        # transform random noise z (batch_size x latent_dim) into transformations
        shape_trans = self.shape_transformer(z)
        
        shape_trans = shape_trans.permute(0,3,2,1)
        return shape_trans

class Discriminator(torch.nn.Module):
    ''' simple image discriminator '''
    def __init__(self,canvas_width=64,canvas_height=64):
        super(Discriminator, self).__init__()
    
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 64, kernel_size=4, 
                stride=2, padding=1, bias=False),
             torch.nn.BatchNorm2d(64),
             torch.nn.LeakyReLU(0.2, inplace=True),
             torch.nn.Conv2d(
                 64, 128, kernel_size=4, 
                 stride=2, padding=1, bias=False),
             torch.nn.BatchNorm2d(128),
             torch.nn.LeakyReLU(0.2, inplace=True),
             torch.nn.Conv2d(
                 128, 256, kernel_size=4, 
                 stride=2, padding=1, bias=False),
             torch.nn.BatchNorm2d(256),
             torch.nn.LeakyReLU(0.2, inplace=True),
             torch.nn.Conv2d(
                 256, 512, kernel_size=4, 
                 stride=2, padding=1, bias=False),
             torch.nn.BatchNorm2d(512),
             torch.nn.LeakyReLU(0.2, inplace=True),
             torch.nn.Conv2d(
                 512, 1, kernel_size=4, 
                 stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )
        

    def forward(self, img):
        img = img.permute(0, 3, 1, 2)
        return self.model(img)


def train(bs=128,datapath='pngImagesSmiley64x64.pkl',epochs=100,lr_g=0.0003,lr_d=0.0002,out='./',transform=False,check_interval=10):
    print("Checking if cuda is available...")
    cuda = True if torch.cuda.is_available() else False
    print("   "+str(cuda))
    print("Loading data...")
    _xform = xforms.Compose([xforms.ToPILImage(),                            # convert numpy array to PIL image
                            xforms.Grayscale(),                             # reduce to 1 channel
                            xforms.Resize([canvas_width, canvas_height]),   # resize to working resolution
                            xforms.ToTensor()])                             # transform to pytorch tensor
    #   the dataset object
    if transform:
        data = CustomImageDataset(datapath,transform=_xform)
    else:
        data = CustomImageDataset(datapath,transform=None)

    # Initialize asynchronous dataloaders
    loader = DataLoader(data, batch_size=bs, num_workers=2)
    
    print("Initializing model...")
    # Instantiate the models
    gen = Generator(canvas_width=canvas_width,canvas_height=canvas_height)
    discrim = Discriminator(canvas_width=canvas_width,canvas_height=canvas_height)

    # Checkpointer to save/recall model parameters
    checkpointer_gen = ttools.Checkpointer(os.path.join(out, "checkpoints"), model=gen, prefix="gen_")
    checkpointer_discrim = ttools.Checkpointer(os.path.join(out, "checkpoints"), model=discrim, prefix="discrim_")

    print("Checking for checkpoints...")
    # resume from a previous checkpoint, if any
    checkpointer_gen.load_latest()
    checkpointer_discrim.load_latest()

    # choose loss, e.g. binary cross entropy
    lossfunc = torch.nn.BCELoss()
    if cuda:
        gen.cuda()
        discrim.cuda()
        
        lossfunc.cuda()

    #gen.apply(weights_init_normal)
    gen.apply(weights_init_normal)
    discrim.apply(weights_init_normal)
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.99))
    optimizer_D = torch.optim.Adam(discrim.parameters(), lr=lr_d, betas=(0.5, 0.99))
    
    print("Training...")
    L2Reg = 1e5
    TINY = 1e-8
    device = torch.device("cuda" if cuda else "cpu")
    
    real_label = 1.
    fake_label = 0.
    
    
    for epoch in range(epochs):
        if (epoch % check_interval)==0:
            path = "{}{}".format("epoch_", epoch)
            checkpointer_gen.save(path)
            checkpointer_discrim.save(path)
            z = torch.zeros([1, latent_dim], device = device).normal_()
            noise = torch.randn(1, latent_dim, 1, 1, device=device).normal_()
            #z = torch.randn((1, latent_dim), device=device).normal_()
            img = gen(noise,device).detach()
            plt.imshow(img.cpu().numpy()[0,:,:,:])
            plt.savefig(os.path.join(out,"checkpoints/genfig_epoch_"+str(epoch)+".png") )
        for i, imgs in enumerate(loader):
            b_size = imgs.shape[0]
            z = torch.zeros([b_size, latent_dim], device = device).normal_()
            #z = torch.randn((b_size, latent_dim), device=device).normal_()
            imgs = imgs.to(device)
            
            
            # -----------------
            #  Train Generator
            # -----------------
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = gen(noise,device)
            fake_pred = discrim(fake_imgs)
#             tpars = gen.shape_transformer(z)[:,4]/(max(canvas_width,canvas_height)*0.1)
#             generator_loss = - torch.mean(torch.log(fake_pred + TINY))+ L2Reg * (tpars**2).mean()
#             optimizer_G.zero_grad()
#             generator_loss.backward()
#             optimizer_G.step()
            
            # --------------------
            #  Train Discriminator
            # --------------------
            
            discrim.zero_grad() # reset the gradient
            
            imgs = imgs.float()
            real_pred = discrim(imgs) # added device 
            fake_pred = discrim(fake_imgs.detach())
            #________________________
            
            output = real_pred.reshape(-1)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            errD_real = lossfunc(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            label.fill_(fake_label)
            output = fake_pred.reshape(-1)
            errD_fake = lossfunc(output,label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()
            
            gen.zero_grad()
            label.fill_(real_label)
            output = discrim(fake_imgs).reshape(-1)
            errG = lossfunc(output,label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()
            
            #_____________________________
            
            #discriminator_loss = - torch.mean(torch.log(real_pred + TINY) + torch.log(1. - fake_pred + TINY))
            
            #optimizer_D.zero_grad()
            #discriminator_loss.backward()
            #optimizer_D.step()
            
        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, epochs, errD.item(), 
                                                        errG.item()))
        
if __name__ == "__main__":
    ''' usage:
        $ CUDA_VISIBLE_DEVICES=2 python echoGAN.py testframes_psax.pkl ./
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data", help="directory containing training dataset.")
    parser.add_argument("out", help="directory to write the checkpoints.")
    parser.add_argument("--lr_g", type=float, default=3e-4, help="learning rate for the generator.")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="learning rate for the discriminator.")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train for.")
    parser.add_argument("--bs", type=int, default=64, help="number of elements per batch.")
    parser.add_argument("--transform", action='store_true', default=False, help="use transformer on data.")
    parser.add_argument("--check_interval", type=int, default=10, help="Number of epochs between checkpoints.")
    args = parser.parse_args()
    #ttools.set_logger(True)  # activate debug prints
    train(args.bs, args.data, args.epochs, args.lr_g, args.lr_d, args.out, args.transform, args.check_interval)