import os
import argparse

import random
import zarr

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

import numcodecs
import caecodec
numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)


class ViTClassifierHead(VisionTransformer):
    """Implementation of the classifier head from the ViT-B-16 architecture.

    This model is compatible with the features extracted by any other model,
    such is the case of the analysis track of the CAE.
    """
    def __init__(self, channels_org=3, channels_bn=768, cut_position=6,
                 patch_size=128,
                 compression_level=4,
                 num_classes=1000,
                 dropout=0.0,
                 **kwargs):
        if cut_position is None:
            cut_position = 6

        if cut_position > 0:
            image_size = patch_size // 2**compression_level
            vit_patch_size = 1

        else:
            image_size = patch_size
            vit_patch_size = 16

        super(ViTClassifierHead, self).__init__(
            image_size=image_size,
            patch_size=vit_patch_size,
            num_layers=12 - cut_position,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_classes,
            dropout=dropout)

        # When using pre-extracted features, such is the case of the analysis
        # track of the CAE, the projection layer maps the number of latent
        # channels to the ones required by the encoder layers. Also, the number
        # of encoder layers are reduced since the analysis track provides a set
        # of feature maps ready for computation.
        if cut_position > 0:
            self.conv_proj = nn.Conv2d(channels_bn, 768, kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
        elif channels_org != 3:
            self.conv_proj = nn.Conv2d(channels_org,
                                       self.conv_proj.out_channels,
                                       kernel_size=self.conv_proj.kernel_size,
                                       stride=self.conv_proj.stride,
                                       padding=self.conv_proj.padding,
                                       bias=self.conv_proj.bias is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform donwstream analysis on the "
                                     "compressed representaton of an image "
                                     "that has been compacted using the "
                                     "Convolutional Autoencoder codec")
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+",
                        help="Compressed input image(s)",
                        required=True)

    parser.add_argument("-m", "--model", dest="checkpoint", type=str,
                        help="Pre-trained model checkpoint")

    args = parser.parse_args()

    batch = []
    for fn in args.input:
        # Use the BottleneckStore storage to open the compressed image and use
        # it as a bottleneck tensor for other deep learning model inference.
        store = caecodec.BottleneckStore(fn, mode="r")
        z_arr = zarr.open(store)

        # Take a random crop of size 256x256 uncompressed pixels, equivalent to
        # a 16x16 patch in compressed pixels.
        h, w, _ = z_arr.shape

        r_y = random.randint(0, h - 17)
        r_x = random.randint(0, w - 17)

        crop_range = (slice(r_y, r_y + 16, None), slice(r_x, r_x + 16, None),
                      slice(None))

        # Reorder the tensor axes to have channels, height, and width (cyx)
        # structure.
        z_arr = torch.from_numpy(z_arr[crop_range].transpose(2, 0, 1))

        batch.append(z_arr)

    batch = torch.stack(batch)
    print("Batched images", batch.shape)

    # This configuration works with the CAE models of higher capacity. Those
    # generate feature maps of 320 channels (channels_bn) at a compression of
    # four levels.
    net = ViTClassifierHead(channels_org=3, channels_bn=320, cut_position=6,
                            patch_size=256,
                            compression_level=4,
                            num_classes=1000)

    if torch.cuda.is_available():
        net.cuda()
        map_to_cpu = None
        batch = batch.cuda()

    else:
        map_to_cpu = "cpu"

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=map_to_cpu)
        net.load_state_dict(checkpoint)

    output = net(batch)

    print("Output:", output.shape)
    print("Inferred class:", output.max(dim=1))
