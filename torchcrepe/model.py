import functools

import torch
import torch.nn.functional as F

import torchcrepe

from typing import List


###########################################################################
# Model definition
###########################################################################


class Crepe(torch.nn.Module):
    """Crepe model definition"""

    def __init__(self, model: str='full'):
        super().__init__()

        # Model-specific layer parameters
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f'Model {model} is not supported')

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=0.0010000000474974513,
                                          momentum=0.0)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=torchcrepe.PITCH_BINS)

    def forward(self, x: torch.Tensor, embed: bool=False):
        # Forward pass through first five layers
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv6_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return torch.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x: torch.Tensor):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers (sadly I can't for loop this because it would change the weights names)
        x = F.pad(x, (0, 0, 254, 254))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv5_BN(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))

        return x
