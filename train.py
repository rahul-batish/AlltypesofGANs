#########################loading libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GANs import Discriminator, Generator, initialize_weights
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms


######################### Data Loading  #######################################
    
def returnloader(DATA_DIR = './celeb',insize = 256,outsize=16,batch_size=64):  
    transform = transforms.Compose([transforms.Resize(outsize),
                                     transforms.ToTensor()])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform) # TODO: create the ImageFolder
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32) # TODO: use the ImageFolder dataset to create the DataLoader
    return dataloader

################################ image show  ##################################

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
    

########################################## Hyperparameters etc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 16
CHANNELS_IMG = 3
NOISE_DIM = 16*16
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64


# If you train on MNIST, remember to set channels_img to 1
dataloader=returnloader()
# images, labels = next(iter(dataloader))
# for images , labels in dataloader:
#     for image in images:
#         print(image.shape)
#         # image=maxpooler(image,256,4)
#         print(image.shape)
#         imshow(image, normalize=False)
        
#     break

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, NOISE_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/real")
# writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            # with torch.no_grad():
            #     fake = gen(fixed_noise)
            #     # take out (up to) 32 examples
            #     img_grid_real = torchvision.utils.make_grid(
            #         real[:32], normalize=True
            #     )
            #     img_grid_fake = torchvision.utils.make_grid(
            #         fake[:32], normalize=True
            #     )
                
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            

noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
fake = gen(noise)
print(fake.shape)
print(fake.type)
fake = fake.to("cpu")
fake=fake.detach()
print(fake.shape)
imshow(fake[0], normalize=False)

dataloader=returnloader()
# for images , labels in dataloader:
#     for image in images:
#         print(image.shape)
#         print(image.type)
#         # image=maxpooler(image,256,4)
#         print(image.shape)
#         imshow(image, normalize=False)
        
#     break