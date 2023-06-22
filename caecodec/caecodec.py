import io
import base64
import struct

import numpy as np
import torch

import compressai as cai

from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray


class ConvolutionalAutoencoder(Codec):
    codec_id = 'cae'
    def __init__(self, quality=8, metric="mse", gpu=False):
        self.gpu = gpu and torch.cuda.is_available()
        self.quality = quality
        self.metric = metric

        self._net = cai.zoo.bmshj2018_factorized(quality=quality,
                                                 metric=metric,
                                                 pretrained=True)
        if self.gpu:
            self._net.cuda()

        self._net.eval()

    def encode(self, buf):
        h, w, c = buf.shape

        buf_x = torch.from_numpy(buf)

        if self.gpu:
            buf_x = buf_x.cuda()

        with torch.no_grad():
            buf_x = buf_x.permute(2, 0, 1)
            buf_x = buf_x.view(1, c, h, w)
            buf_x = buf_x.float() / 255.0
            out = self._net.compress(buf_x)

        chunk_size_code = struct.pack('>QQ', h, w)

        return chunk_size_code + out["strings"][0][0]

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        downsampling_factor = self._net.downsampling_factor

        h, w = struct.unpack('>QQ', buf[:16])

        buf_shape = torch.Size([h // downsampling_factor,
                                w // downsampling_factor])

        with torch.no_grad():
            dec_out = self._net.decompress(strings=[[buf[16:]]],
                                           shape=buf_shape)

            buf_x_r = dec_out["x_hat"][0].cpu().detach()
            buf_x_r = buf_x_r * 255.0
            buf_x_r = buf_x_r.clip(0, 255).to(torch.uint8)
            buf_x_r = buf_x_r.permute(1, 2, 0)
            buf_x_r = buf_x_r.numpy()

        buf_x_r = np.ascontiguousarray(buf_x_r)
        buf_x_r = ensure_contiguous_ndarray(buf_x_r)

        return ndarray_copy(buf_x_r, out)


class ConvolutionalAutoencoderBottleneck(Codec):
    codec_id = 'cae_bn'
    def __init__(self, channels_bn, quality=8, metric="mse", filters=None,
                 fact_ent_checkpoint=None,
                 gpu=False):
        self.gpu = gpu and torch.cuda.is_available()
        self.quality = quality
        self.metric = metric

        # Because only parameters that can be stored as json are supported by
        # numcodecs, the parameters of the entropy bottleneck model are
        # converted to bytes and stored as strings to be compatible with json.
        if fact_ent_checkpoint is None:
            self.net = cai.zoo.bmshj2018_factorized(quality=quality,
                                                    metric=metric,
                                                    pretrained=True)

            filters = self.net.entropy_bottleneck.filters

            fact_ent_checkpoint = {}
            for n, par in self.net.entropy_bottleneck.named_parameters():
                fact_ent_checkpoint[n] = self._tensor2bytes(par)

        self.filters = filters
        self.channels_bn = channels_bn
        self.fact_ent_checkpoint = fact_ent_checkpoint

        self._setup_encoder(gpu)

    def _setup_encoder(self, gpu=False):
        # Only the factorized entropy model is needed to convert bytes into the
        # autoencoder's bottleneck.
        self._fact_ent = cai.entropy_models.EntropyBottleneck(
            channels=self.channels_bn,
            filters=self.filters)

        fact_ent_checkpoint = {}
        for n, par in self.fact_ent_checkpoint.items():
            fact_ent_checkpoint[n] = self._bytes2tensor(par)

        self._fact_ent.load_state_dict(fact_ent_checkpoint, strict=False)

        if gpu:
            self._fact_ent.cuda()

    @staticmethod
    def _tensor2bytes(tensor):
        buf = io.BytesIO()
        torch.save(tensor.cpu().detach(), buf)

        # Convert the bytes buffer into a serializable ASCII string
        buf = base64.b64encode(buf.getvalue()).decode('ascii')
        return buf

    @staticmethod
    def _bytes2tensor(buf):
        # Decode the string into bytes
        buf = io.BytesIO(base64.b64decode(buf))
        tensor = torch.load(buf)
        return tensor

    def encode(self, buf):
        h, w, c = buf.shape

        buf_y = torch.from_numpy(buf)
        buf_y = buf_y.permute(2, 0, 1)
        buf_y = buf_y.view(1, c, h, w)

        if self.gpu:
            buf_y = buf_y.cuda()

        buf_ae = self._fact_ent.compress(buf_y)

        chunk_size_code = struct.pack('>QQ', h, w)

        return chunk_size_code + buf_ae[0]

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        h, w = struct.unpack('>QQ', buf[:16])

        buf_shape = (h, w)

        buf_y_q = self._fact_ent.decompress([buf[16:]],
                                            size=buf_shape)

        buf_y_q = buf_y_q[0].detach().cpu().permute(1, 2, 0)
        buf_y_q = buf_y_q.float().numpy()

        buf_y_q = np.ascontiguousarray(buf_y_q)
        buf_y_q = ensure_contiguous_ndarray(buf_y_q)

        return ndarray_copy(buf_y_q, out)
