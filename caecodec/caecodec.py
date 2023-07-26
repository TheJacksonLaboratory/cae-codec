import struct
import functools
import operator

import numpy as np
import torch

import compressai as cai

from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray


class ConvolutionalAutoencoder(Codec):
    codec_id = 'cae'
    def __init__(self, quality=8, metric="mse",
                 patch_size=256,
                 gpu=True,
                 partial_decompress=False,
                 bottleneck_channels=None,
                 downsampling_factor=None):

        # The size of the patches depend on the available GPU memory.
        self.patch_size = 256

        # Whether to reconstruct the image, or just retrieve the bottleneck
        # tensors.
        self.partial_decompress = partial_decompress

        # Use GPUs when available and reuqested by the user.
        self.gpu = gpu and torch.cuda.is_available()
        self.quality = quality
        self.metric = metric

        self._net = cai.zoo.bmshj2018_factorized(quality=self.quality,
                                                 metric=self.metric,
                                                 pretrained=True)
        self.bottleneck_channels = self._net.g_a[6].weight.size(0)
        self.downsampling_factor = self._net.downsampling_factor

        self._net.eval()
        if self.gpu:
            self._net.cuda()

    def encode(self, buf):
        h, w, c = buf.shape
        pad_h = (self.patch_size - h) % self.patch_size
        pad_w = (self.patch_size - w) % self.patch_size

        buf_x = np.pad(buf, [(0, pad_h), (0, pad_w), (0, 0)])
        with torch.no_grad():
            buf_x = torch.from_numpy(buf_x)
            if self.gpu:
                buf_x = buf_x.cuda()
            buf_x = buf_x.permute(2, 0, 1).float().div(255.0)
            buf_x_ps = torch.nn.functional.unfold(
                buf_x[None, ...],
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size), 
                padding=0)
            buf_x_ps = buf_x_ps.transpose(1, 2)
            buf_x_ps = buf_x_ps.reshape(-1, c, self.patch_size,
                                        self.patch_size)
            buf_dl = torch.utils.data.DataLoader(
                buf_x_ps,
                batch_size=max(1, torch.cuda.device_count()),
                pin_memory=False,
                num_workers=0)

            out_str = []
            for buf_x_ps_k in buf_dl:
                out_str += self._net.compress(buf_x_ps_k)["strings"][0]

        chunk_size_code = struct.pack('>QQQ', h, w, self.patch_size)
        chunk_size_code += struct.pack('>' + 'Q' * len(out_str),
                                       *list(map(len, out_str)))
        chunk_buf = chunk_size_code + functools.reduce(operator.add, out_str)
        return chunk_buf

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        h, w, patch_size = struct.unpack('>QQQ', buf[:24])
        nh = int(np.ceil(h / patch_size))
        nw = int(np.ceil(w / patch_size))
        n_patches = nh * nw

        comp_patch_size = patch_size // self.downsampling_factor
        buf_shape = torch.Size([comp_patch_size, comp_patch_size])

        buf_offset = 24 + 8 * n_patches
        len_strs = list(struct.unpack('>' + 'Q' * n_patches,
                                      buf[24:buf_offset]))
        start_strs = np.cumsum([buf_offset] + len_strs[:-1])
        strs = [buf[s:s+l] for s, l in zip(start_strs, len_strs)]

        with torch.no_grad():
            strs_dl = strs

            if self.partial_decompress:
                h_comp = h // self.downsampling_factor
                w_comp = w // self.downsampling_factor
                y_hat_ps = []
                for buf_k in strs_dl:
                    y_hat_ps_k =\
                        self._net.entropy_bottleneck.decompress(
                            [buf_k],
                            size=buf_shape)
                    y_hat_ps.append(y_hat_ps_k.cpu())

                y_hat_ps = torch.cat(y_hat_ps, dim=0)
                y_hat_ps = y_hat_ps.reshape(n_patches, -1)
                y_hat_ps = y_hat_ps.transpose(0, 1)

                y_hat = torch.nn.functional.fold(
                    y_hat_ps,
                    output_size=(nh * comp_patch_size, nw * comp_patch_size),
                    kernel_size=(comp_patch_size, comp_patch_size),
                    stride=(comp_patch_size, comp_patch_size),
                    padding=0)

                y_hat = y_hat[..., :h_comp, :w_comp]
                buf_x_r = y_hat

            else:
                x_hat_ps = []
                for buf_k in strs_dl:
                    x_hat_ps_k = self._net.decompress([[buf_k]],
                                                      shape=buf_shape)
                    x_hat_ps.append(x_hat_ps_k["x_hat"].cpu())

                x_hat_ps = torch.cat(x_hat_ps, dim=0)
                x_hat_ps = x_hat_ps.reshape(n_patches, -1)
                x_hat_ps = x_hat_ps.transpose(0, 1)

                x_hat = torch.nn.functional.fold(
                    x_hat_ps,
                    output_size=(nh * patch_size, nw * patch_size),
                    kernel_size=(patch_size, patch_size),
                    stride=(patch_size, patch_size),
                    padding=0)

                x_hat = x_hat[..., :h, :w]
                buf_x_r = x_hat.mul(255.0).clip(0, 255).to(torch.uint8)

            buf_x_r = buf_x_r.permute(1, 2, 0)
            buf_x_r = buf_x_r.numpy()

        buf_x_r = np.ascontiguousarray(buf_x_r)
        buf_x_r = ensure_contiguous_ndarray(buf_x_r)

        return ndarray_copy(buf_x_r, out)
