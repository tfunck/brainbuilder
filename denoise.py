import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import os, pickle, h5py
from PIL import Image
from skimage.util import view_as_window


epoch = 100
learning_rate = 0.0002

data_dir = '/data1/users/adoyle/ReceptorMaps/'
workdir = '/home/users/adoyle/ReceptorMaps/'

image_size = (4164, 3120)
patch_size = (64, 64)

batch_size = 256

def setup_experiment(workdir):
    try:
        experiment_number = pickle.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pickle.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    return results_dir, experiment_number

class ReceptorMapDataset(Dataset):
    image_filenames = []

    def __init__(self, data_dir, augmentation_type=None):
        for filenames in os.listdir(data_dir):
            self.image_filenames.append(data_dir + filenames)

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index])

        img_batch = np.zeros((batch_size, patch_size[0], patch_size[1]), dtype='float32')

        x_coords = np.random.randint(0, image_size[0] - patch_size[0], batch_size)
        y_coords = np.random.randint(0, image_size[1] - patch_size[1], batch_size)

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            img_batch[i, :, :] = img[x:x+patch_size[0], y:y+patch_size[1]]

        return img_batch

    def __len__(self):
        return len(self.image_filenames)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(batch_size, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


if __name__ == '__main___':
    results_dir, experiment_number = setup_experiment(workdir)

    train_dataset = ReceptorMapDataset(data_dir)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)

    encoder = Encoder().cuda()
    decoder = Decoder().cuda()

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # train encoder and decoder
    for i in range(epoch):
        for image in train_loader:
            noise = torch.rand(batch_size, 1, patch_size[0], patch_size[1])
            image_n = torch.mul(image + 0.25, 0.1 * noise)

            image = Variable(image).cuda()
            image_n = Variable(image_n).cuda()

            optimizer.zero_grad()

            # encode noisy image, decode it
            output = encoder(image_n)
            output = decoder(output)

            loss = loss_func(output, image)
            loss.backward()
            optimizer.step()

        # torch.save([encoder, decoder], './model/deno_autoencoder.pkl')
        print('Loss:', loss)

    img = image[0].cpu()
    input_img = image_n[0].cpu()
    output_img = output[0].cpu()

    origin = img.data.numpy()
    inp = input_img.data.numpy()
    out = output_img.data.numpy()

    plt.imshow(origin[0], cmap='gray')
    plt.savefig(results_dir + 'original.png', bbox_inches='tight')
    plt.imshow(inp[0], cmap='gray')
    plt.savefig(results_dir + 'input.png', bbox_inches='tight')
    plt.imshow(out[0], cmap="gray")
    plt.savefig(results_dir + 'output.png', bbox_inches='tight')