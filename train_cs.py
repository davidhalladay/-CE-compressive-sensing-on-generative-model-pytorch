import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import random
import string
import os
from CelebA_dataset import CelebA_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import get_celeba
from dcgan import weights_init, Generator, Discriminator
from reconstructor import reconstructor, UnNormalize

# Set random seed for reproducibility.
seed = 18
random.seed(seed)
torch.manual_seed(seed)
print("<" + "="*20 + ">")
print("Random Seed: ", seed)

def freeze_model(model):
    for p in model.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
        p.requires_grad = False
    return model

# Parameters to define the model.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='celeba | mnist')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--checkpoint_root', required=True, help='Path to the preprocessed data')
parser.add_argument('--image_root', required=True, help='Path to the preprocessed data')
parser.add_argument('--save_folder', required=True, help='Path to save the data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nmeasure', type=int, default=100, help='size of measurements')
parser.add_argument('--noise_std', type=int, default=0.01, help='noise_std')
parser.add_argument('--nc', type=int, default=3, help='Number of channles in the training images. For coloured images this is 3.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niters', type=int, default=2000, help='number of iteri to train for')
parser.add_argument('--save_iter', type=int, default=1000, help='save step.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

parser.add_argument('--mloss1_weight', type=float, default=0., help='mloss1_weight')
parser.add_argument('--mloss2_weight', type=float, default=1., help='mloss2_weight')
parser.add_argument('--zprior_weight', type=float, default=1., help='zprior_weight')
parser.add_argument('--dloss1_weight', type=float, default=0., help='dloss1_weight')
parser.add_argument('--dloss2_weight', type=float, default=0., help='dloss2_weight')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print("<" + "="*20 + ">")
print(opt)

# folder
os.makedirs(opt.save_folder, exist_ok=True)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print("<" + "="*20 + ">")
print(device, " will be used.")

# Get the data.
transform = transforms.Compose([
        transforms.Resize((opt.imageSize,opt.imageSize)),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor()
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
img = Image.open(os.path.join(opt.image_root))
img = transform(img).unsqueeze(0)#.transpose(0,1).transpose(1,2)
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# Plot the training images.
# sample_batch = next(iter(train_loader))

# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch[0].to(device).transpose(1,2).transpose(0,1,), padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()

# Load checkpoint model
checkpoint = torch.load(opt.checkpoint_root)
# reconstructor
netR = reconstructor(opt).to(device)
print("<" + "="*20 + ">")
print(netR)

# Create the generator.
netG = Generator(opt).to(device)
netG.load_state_dict(checkpoint['generator'])
print("Freeze model {}.".format('generator'))
netG = freeze_model(netG)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
# netG.apply(weights_init)
# Print the model.
print("<" + "="*20 + ">")
print(netG)

# Create the discriminator.
netD = Discriminator(opt).to(device)
netD.load_state_dict(checkpoint['discriminator'])
print("Freeze model {}.".format('discriminator'))
netD = freeze_model(netD)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
# netD.apply(weights_init)
# Print the model.
print("<" + "="*20 + ">")
print(netD)


# Binary Cross Entropy loss function.
m1_criterion = nn.L1Loss()
m2_criterion = nn.MSELoss()
zp_criterion = nn.MSELoss()
d_criterion = nn.BCELoss()
g_criterion = nn.BCELoss()

# compressive sensing part
A = torch.randn(opt.nmeasure , opt.nc* opt.imageSize* opt.imageSize, device=device)
z_batch = torch.randn(opt.batchSize , opt.nz, 1, 1, device=device)
zero_batch = torch.zeros(opt.batchSize , opt.nz, 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the z_batch.
optimizer = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Stores generated images as training progresses.
# img_list = []
# Stores generator losses during training.
# G_losses = []
# Stores discriminator losses during training.
m1_losses = []
m2_losses = []
zp_losses = []
D_losses = []
G_losses = []
total_losses = []

# GT data 
eta_noise = (1./opt.nmeasure) * torch.randn(opt.nmeasure, 1, device=device)
y_batch = torch.matmul(A, img.to(device).view(-1, img.shape[0])) + eta_noise
y_batch = y_batch.view(opt.batchSize, -1)

iters = 0
###################
# fixed_noise = torch.randn(16, opt.nz, 1, 1, device=device)
# with torch.no_grad():
#     fake_data = netG(fixed_noise).detach().cpu()
#     fake_data = vutils.make_grid(fake_data, padding=2, normalize=True)
#     plt.figure(figsize=(10,5))
# #     plt.title("Reconstructed image")
#     plt.imshow(np.transpose(fake_data,(1,2,0)))
#     plt.savefig(os.path.join(opt.save_folder, 'rand_test.png'))
#     exit()
###################
print("Starting Training Loop...")
print("-"*25)

for iteri in range(opt.niters):
    # Transfer data tensor to GPU/CPU (device)
    torch.autograd.set_detect_anomaly(True)
    real_data = img.to(device)
    # Get batch size. Can be different from params['nbsize'] for last batch in iteri.
    b_size = real_data.size(0)
    z_batch = Variable(z_batch)

    # Make accumalated gradients of the discriminator zero.
    f_label = Variable(torch.zeros((b_size, ), device=device))
    r_label = Variable(torch.ones((b_size, ), device=device))
    optimizer.zero_grad()

    z_rec_batch = netR(z_batch)
    x_hat_batch = netG(z_rec_batch)
#     x_hat_batch = unorm(x_hat_batch)
    x_hat_batch = torch.add(x_hat_batch, 0.5, alpha=0.5)

#     rec_image = vutils.make_grid(unorm(x_hat_batch).detach().cpu().squeeze(0), padding=2, normalize=True)
#     plt.figure(figsize=(10,5))
#     plt.title("Reconstructed image")
#     plt.imshow(np.transpose(rec_image,(1,2,0)))
#     plt.savefig(os.path.join(opt.save_folder, 'tmp.png'))
#     exit()
    output = netD(x_hat_batch).view(-1)

    y_hat_batch = torch.matmul(A, x_hat_batch.view(-1, x_hat_batch.shape[0]))
    y_hat_batch = y_hat_batch.view(opt.batchSize, -1)

    m1_loss = m1_criterion(y_hat_batch, y_batch)
    m2_loss = m2_criterion(y_hat_batch, y_batch)
    zp_loss = zp_criterion(z_rec_batch, zero_batch)
    # discriminator loss
    d_loss = d_criterion(output, f_label)
    # generator loss
    g_loss = g_criterion(output, r_label)
    total_loss = m1_loss * opt.mloss1_weight + \
                 m2_loss * opt.mloss2_weight + \
                 zp_loss * opt.zprior_weight + \
                 d_loss  * opt.dloss1_weight + \
                 g_loss  * opt.dloss2_weight

    total_loss.backward()
    optimizer.step()
    
    # Check progress of training.
    if (iteri+1)%1000 == 0:
#             print(torch.cuda.is_available())
        print('[%d/%d] m1_loss %.3f | m2_loss %.3f | zp_loss %.3f | d_loss %.3f | g_loss %.3f | total_loss %.3f'
              % (iteri+1, opt.niters,
                     m1_loss.item(), m2_loss.item(), zp_loss.item(), d_loss.item(), 
                     g_loss.item(), total_loss.item()))

    # Save the losses for plotting.
    total_losses.append(total_loss.item())

#     # Check how the generator is doing by saving G's output on a fixed noise.
#     if (iters % 100 == 0) or ((iteri+1 == opt.niters) and (i == len(train_loader)-1)):
#         with torch.no_grad():
#             fake_data = netG(fixed_noise).detach().cpu()
#         img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

#     iters += 1

    # Save the model.
#     if (iteri+1) % opt.save_iter == 0:
#         print("Saving the model...")
#         torch.save({
#             'optimizer' : optimizer.state_dict(),
#             }, os.path.join(opt.save_folder, 'model_rec_iteri_{}.pth'.format(iteri+1)))

# Save the final trained model.
print("Saving the final model...")
torch.save({
    'optimizer' : optimizer.state_dict(),
    }, os.path.join(opt.save_folder, 'model_rec_iteri_{}.pth'.format(iteri+1)))

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Totla Loss During Training")
plt.plot(total_losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig(os.path.join(opt.save_folder, 'loss.png'))

# Testing 
with torch.no_grad():
    netR.eval()
    z_rec_batch = netR(z_batch)
    rec_image = netG(z_rec_batch).detach().cpu()
    test_image = netG(z_batch).detach().cpu()
    
img = Image.open(os.path.join(opt.image_root))
img = transform(img).transpose(0,1).transpose(1,2)

img = vutils.make_grid(img, padding=2, normalize=True)
plt.figure(figsize=(10,5))
plt.title("Real image")
plt.imshow(img)
plt.savefig(os.path.join(opt.save_folder, 'real.png'))

rec_image = vutils.make_grid(rec_image, padding=2, normalize=True)
plt.figure(figsize=(10,5))
plt.title("Reconstructed image")
plt.imshow(np.transpose(rec_image,(1,2,0)))
plt.savefig(os.path.join(opt.save_folder, 'Reconstructed.png'))

test_image = vutils.make_grid(test_image, padding=2, normalize=True)
plt.figure(figsize=(10,5))
plt.title("Reconstructed image")
plt.imshow(np.transpose(test_image,(1,2,0)))
plt.savefig(os.path.join(opt.save_folder, 'test.png'))

