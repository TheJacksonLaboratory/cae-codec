import struct
import functools
import operator

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
    _patch_size = 256
    _bottleneck_channels = None
    _downsample_factor = None
    _partial_decompress = False
    _gpu = False

    def __init__(self, quality: int = 8, metric: str = "mse",
                 patch_size: int = None,
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
        # The size of the patches depend on the available GPU memory.
        if patch_size is not None:
            self._patch_size = patch_size

        self._partial_decompress = partial_decompress

        # Use GPUs when available and reuqested by the user.
        self._gpu = gpu and torch.cuda.is_available()

        self.quality = quality
        self.metric = metric

        self._net = cai.zoo.bmshj2018_factorized(quality=self.quality,
                                                 metric=self.metric,
                                                 pretrained=True)

        self._bottleneck_channels = self._net.g_a[6].weight.size(0)
        self._downsampling_factor = self._net.downsampling_factor

        self._net.eval()
        if self.gpu:
            self._net.cuda()

    @property
    def patch_size(self) -> int:
        """The size of the patches in which the input image is divided to fit
        in the GPU's memory.
        """
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: int) -> None:
        self._patch_size = patch_size

    @property
    def partial_decompress(self) -> bool:
        """Whether to reconstruct the image or just retrieve the bottleneck 
        tensors.
        """
        return self._partial_decompress

    @partial_decompress.setter
    def partial_decompress(self, partial_decompress: bool = False) -> None:
        self._partial_decompress = partial_decompress

    @property
    def bottleneck_channels(self) -> int:
        return self._bottleneck_channels

    @property
    def downsampling_factor(self) -> int:
        return self._downsampling_factor

    @property
    def gpu(self) -> bool:
        return self._gpu

    @gpu.setter
    def gpu(self, gpu: bool) -> None:
        self._gpu = gpu

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
        pad_h = (self._patch_size - h) % self._patch_size
        pad_w = (self._patch_size - w) % self._patch_size

        buf_x = np.pad(buf, [(0, pad_h), (0, pad_w), (0, 0)])
        with torch.no_grad():
            buf_x = torch.from_numpy(buf_x)
            if self.gpu:
                buf_x = buf_x.cuda()

            # Divide the input image into patches of size `patch_size`, so it
            # can fit in the GPU's memory.
            buf_x = buf_x.permute(2, 0, 1).float().div(255.0)
            buf_x_ps = torch.nn.functional.unfold(
                buf_x[None, ...],
                kernel_size=(self._patch_size, self._patch_size),
                stride=(self._patch_size, self._patch_size), 
                padding=0)
            buf_x_ps = buf_x_ps.transpose(1, 2)
            buf_x_ps = buf_x_ps.reshape(-1, c, self._patch_size,
                                        self._patch_size)
            buf_dl = torch.utils.data.DataLoader(
                buf_x_ps,
                batch_size=max(1, torch.cuda.device_count()),
                pin_memory=False,
                num_workers=0)

            # Compress each patch with the autoencoder model. 
            out_str = []
            for buf_x_ps_k in buf_dl:
                out_str += self._net.compress(buf_x_ps_k)["strings"][0]

        chunk_size_code = struct.pack('>QQQ', h, w, self._patch_size)
        chunk_size_code += struct.pack('>' + 'Q' * len(out_str),
                                       *list(map(len, out_str)))
        chunk_buf = chunk_size_code + functools.reduce(operator.add, out_str)
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

        h, w, patch_size = struct.unpack('>QQQ', buf[:24])
        nh = int(np.ceil(h / patch_size))
        nw = int(np.ceil(w / patch_size))
        n_patches = nh * nw

        comp_patch_size = patch_size // self.downsampling_factor
        buf_shape = torch.Size([comp_patch_size, comp_patch_size])

        # Unpack the set of encoded patches to decode using the arithmetic
        # decoder.
        buf_offset = 24 + 8 * n_patches
        len_strs = list(struct.unpack('>' + 'Q' * n_patches,
                                      buf[24:buf_offset]))
        start_strs = np.cumsum([buf_offset] + len_strs[:-1])
        strs = [buf[s:s+l] for s, l in zip(start_strs, len_strs)]

        with torch.no_grad():
            strs_dl = strs

            # If downstream analysis is performed uisng the compressed
            # bottleneck tensor, just apply the arithmetic decoder to recover
            # the downsampled tensor. Otherwise, apply the synthesis track of
            # the autoencoder model to recover the compressed image.
            if self._partial_decompress:
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

                # Stitch the patches into a single tensor.
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

                # Stitch the patches into a single image.
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
        buf_x_r = ndarray_copy(buf_x_r, out)

        return buf_x_r
