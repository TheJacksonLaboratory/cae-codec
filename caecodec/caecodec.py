import struct
import math

import numpy as np
import torch

import compressai as cai

from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray


class ConvolutionalAutoencoder(Codec):
    """
    A codec class for compressing images using a convolutional autoencoder deep
    learning model. This codec is compatible with the Zarr file format.

    Attributes:
    -----------
    codec_id : str
        The codec identifier. Used to associate compressed images generated
        with this codec.
    patch_size : int
        The size of the patches in which the input image is divided to fit in
        the GPU's memory.
    partial_decompress : bool
        Whether the image will be decompressed normally, or it will be
        decompressed as a bottleneck tensor for using in downstream analysis.
    quality : int
        Compression quality between 1 and 8. Compression with quality 1
        provides low compression ratios with high distortion, while quality 8
        provides high compression ratios at low distortion
    metric : str
        The metric used when the autoencoder model was trained (``mse``, or
        ``ms-ssim``).
    bottleneck_channels : int
        The number of channels in the bottleneck tensor.
    downsampling_factor : int
        The downsampling factor applied by the autoencoder model.
    gpu : bool
        Whether GPU processing is enabled or not. When no GPUs are present this
        is automatically set to False.

    Methods:
    --------
    encode(buf)
        Encode buffer `buf` into a compressed array of bytes.

    decode(buf, out=None)
        Decode buffer `buf` into a numpy ndarray.
    """
    codec_id = 'cae'
    bottleneck_channels = None
    downsampling_factor = None
    patch_size = None
    partial_decompress = False
    gpu = False

    def __init__(self, quality: int = 8, metric: str = "mse",
                 patch_size: int = 256,
                 partial_decompress: bool = False,
                 gpu: bool = True,
                 **kwargs) -> None:
        """
        Parameters:
        -----------
        quality: int
            Compression quality between 1 and 8.
        metric: str
            The metric used when the autoencoder model was trained (``mse``, or
            ``ms-ssim``).
        patch_size: int
            The size of the patches in which the input image is divided to fit
            in the GPU's memory.
        partial_decompress : bool
            Whether the image will be decompressed normally, or it will be
            decompressed as a bottleneck tensor for using in downstream
            analysis.
        gpu: bool
            Whether GPU processing is enabled or not. When no GPUs are present
            this is automatically set to False.
        """
        # The size of the patches depend on the available GPU memory and must
        # be mdified in the header of the file, preferably before registering
        # with `numcodecs.register_codec`.
        if self.patch_size is None:
            self.patch_size = patch_size

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
            self._net.g_a.cuda()
            self._net.g_s.cuda()

    def encode(self, buf: np.ndarray) -> bytes:
        """Encode image-like buffer `buf` into a set of bytes.

        Parameters:
        -----------
        buf : numpy.ndarray
            Image/chunk with axes in order Color, Height, and Width (cyx).

        Returns:
        --------
        chunk_buf : bytes
            Encoded image/chunk.
        """
        h, w, c = buf.shape
        pad_h = (self.patch_size - h) % self.patch_size
        pad_w = (self.patch_size - w) % self.patch_size

        comp_patch_size = self.patch_size // self.downsampling_factor
        nh = (h + pad_h) // self.patch_size
        nw = (w + pad_w) // self.patch_size
        n_patches = nh * nw

        offset_in = self.downsampling_factor
        offset_out = 1

        buf_x = np.pad(buf, [(offset_in, pad_h + offset_in),
                             (offset_in, pad_w + offset_in),
                             (0, 0)],
                       mode='reflect')
        with torch.no_grad():
            buf_x = torch.from_numpy(buf_x)

            # Divide the input image into patches of size `patch_size`, so it
            # can fit in the GPU's memory.
            buf_x = buf_x.permute(2, 0, 1).float().div(255.0)
            buf_x_ps = torch.nn.functional.unfold(
                buf_x[None, ...],
                kernel_size=(self.patch_size + 2 * offset_in,
                             self.patch_size + 2 * offset_in),
                stride=(self.patch_size, self.patch_size), 
                padding=0)

            buf_x_ps = buf_x_ps.transpose(1, 2)

            buf_x_ps = buf_x_ps.reshape(-1, c,
                                        self.patch_size + 2 * offset_in,
                                        self.patch_size + 2 * offset_in)

            buf_dl = torch.utils.data.DataLoader(
                buf_x_ps,
                batch_size=max(1, torch.cuda.device_count()),
                pin_memory=self.gpu,
                num_workers=0)

            # Compress each patch with the autoencoder model. 
            buf_y_ps = []
            for buf_x_ps_k in buf_dl:
                if self.gpu:
                    buf_x_ps_k = buf_x_ps_k.cuda()
                buf_y_ps_k = self._net.g_a(buf_x_ps_k).detach().cpu()
                buf_y_ps.append(
                    buf_y_ps_k[...,
                               offset_out:comp_patch_size + offset_out,
                               offset_out:comp_patch_size + offset_out]
                               )

            buf_y_ps = torch.cat(buf_y_ps, dim=0)
            buf_y_ps = buf_y_ps.reshape(n_patches, -1)
            buf_y_ps = buf_y_ps.transpose(0, 1)

            # Stitch the patches into a single image.
            buf_y = torch.nn.functional.fold(
                buf_y_ps,
                output_size=(nh * comp_patch_size, nw * comp_patch_size),
                kernel_size=(comp_patch_size, comp_patch_size),
                stride=(comp_patch_size, comp_patch_size),
                padding=0)

            out_str = self._net.entropy_bottleneck.compress(buf_y[None, ...])

        chunk_size_code = struct.pack('>QQQQ', h, w, pad_h, pad_w)
        chunk_buf = chunk_size_code + out_str[0]
        return chunk_buf

    def decode(self, buf: bytes, out: np.ndarray = None) -> np.ndarray:
        """Decode the set of bytes encodec in buffer `buf` into an image-like
        array.
        
        Parameters:
        -----------
        buf : bytes
            Image/chunk with axes in order Color, Height, and Width (cyx).
        out : numpy.ndarray
            Pre-allocated buffer to store the decoded image/chunk

        Returns:
        --------
        buf_x_r : numpy.ndarray
            Decoded image/chunk, or bottleneck tensor.
        """
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        h, w, pad_h, pad_w = struct.unpack('>QQQQ', buf[:32])
        comp_h = (h + pad_h) // self.downsampling_factor
        comp_w = (w + pad_w) // self.downsampling_factor

        with torch.no_grad():
            buf_y = self._net.entropy_bottleneck.decompress([buf[32:]],
                                                            (comp_h, comp_w))

            # If downstream analysis is performed uisng the compressed
            # bottleneck tensor, just apply the arithmetic decoder to recover
            # the downsampled tensor. Otherwise, apply the synthesis track of
            # the autoencoder model to recover the compressed image.
            if self.partial_decompress:
                dwn_h = int(math.ceil(h / self.downsampling_factor))
                dwn_w = int(math.ceil(w / self.downsampling_factor))
                buf_x_r = buf_y[0, :, :dwn_h, :dwn_w]

            else:
                comp_patch_size = self.patch_size // self.downsampling_factor

                comp_pad_h = (comp_patch_size - comp_h) % comp_patch_size
                comp_pad_w = (comp_patch_size - comp_w) % comp_patch_size

                nh = (comp_h + comp_pad_h) // comp_patch_size
                nw = (comp_w + comp_pad_w) // comp_patch_size

                offset_in = 1
                offset_out = self.downsampling_factor

                buf_y = torch.nn.functional.pad(
                    buf_y,
                    (offset_in, comp_pad_w + offset_in,
                     offset_in, comp_pad_h + offset_in,
                     0, 0),
                    mode='reflect')

                # Divide the input image into patches of size `patch_size`, so
                # it can fit in the GPU's memory.
                buf_y_ps = torch.nn.functional.unfold(
                    buf_y,
                    kernel_size=(comp_patch_size + 2 * offset_in,
                                 comp_patch_size + 2 * offset_in),
                    stride=(comp_patch_size, comp_patch_size),
                    padding=0)
                buf_y_ps = buf_y_ps.transpose(1, 2)
                buf_y_ps = buf_y_ps.reshape(-1, self.bottleneck_channels,
                                            comp_patch_size + 2 * offset_in,
                                            comp_patch_size + 2 * offset_in)
                buf_dl = torch.utils.data.DataLoader(
                    buf_y_ps,
                    batch_size=max(1, torch.cuda.device_count()),
                    pin_memory=self.gpu,
                    num_workers=0)

                # Compress each patch with the autoencoder model. 
                buf_x_ps = []
                for buf_y_ps_k in buf_dl:
                    if self.gpu:
                        buf_y_ps_k = buf_y_ps_k.cuda()
                    buf_x_ps_k = self._net.g_s(buf_y_ps_k).detach().cpu()
                    buf_x_ps.append(
                        buf_x_ps_k[...,
                                   offset_out:self.patch_size+offset_out,
                                   offset_out:self.patch_size+offset_out]
                                   )

                n_patches = nh * nw
                buf_x_ps = torch.cat(buf_x_ps, dim=0)
                buf_x_ps = buf_x_ps.reshape(n_patches, -1)
                buf_x_ps = buf_x_ps.transpose(0, 1)

                # Stitch the patches into a single image.
                buf_x = torch.nn.functional.fold(
                    buf_x_ps,
                    output_size=(nh * self.patch_size, nw * self.patch_size),
                    kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size),
                    padding=0)

                buf_x = buf_x[..., :h, :w]
                buf_x_r = buf_x.mul(255.0).clip(0, 255).to(torch.uint8)

            buf_x_r = buf_x_r.permute(1, 2, 0)
            buf_x_r = buf_x_r.numpy()

        buf_x_r = np.ascontiguousarray(buf_x_r)
        buf_x_r = ensure_contiguous_ndarray(buf_x_r)
        buf_x_r = ndarray_copy(buf_x_r, out)

        return buf_x_r
