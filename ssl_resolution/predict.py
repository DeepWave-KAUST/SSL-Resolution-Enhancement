import torch
import os
import numpy as np
import scipy.io as sio
import random
from scipy import signal
from model import UNet
import yaml
from train import load_config

args = load_config('config.yaml')

dir_input='./test/input/'
dir_output='./test/output/'
dir_cp = './checkpoints/CP_epoch'

os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

test_ids = [os.path.splitext(file)[0] for file in os.listdir(args.dir_test) 
            if not file.startswith('.')]

device = torch.device('cuda')
net = UNet(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
net.eval()

print('------ Test starting -------')
for ip, cp in enumerate(args.cp_list):

    # Load corresponding model architecture and weights
    net.load_state_dict(torch.load(f'{dir_cp}{cp}.pth', map_location=device))

    os.makedirs(dir_output + 'epoch', exist_ok=True)

    for i, fn in enumerate(test_ids):
        tar_file = os.path.join(args.dir_test, fn)
        print("\nPredicting seismic data {} ...".format(fn))
        dict = sio.loadmat(tar_file)
        tar_img = dict['label']
        tar_img = torch.from_numpy(tar_img.copy()).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            pred = net(tar_img)

        pred = pred.cpu().squeeze().numpy()

        sio.savemat(dir_output + 'epoch' + str(cp) + '/' + fn + '_out.mat', {'pred': pred})

        print("\nPredicting seismic data {} have done...".format(fn))

print('------ Test completed successfully -------')
    