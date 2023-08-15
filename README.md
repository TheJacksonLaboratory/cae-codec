# CAE-Codec
Convolutional Autoencoder (CAE) codec for image compression in [Zarr](https://zarr.readthedocs.io/en/stable/index.html) format.

The CAE extends the Factorized Prior model from Balle, et. al. (2018) ["Variational Image Compression with a Scale Hyperprior"](https://arxiv.org/abs/1802.01436) to be used as chunk compressor to store images in [zarr](https://zarr.readthedocs.io) format.
This codec uses the pre-trained convolutional autoencoders from the [Compress-AI library](https://github.com/InterDigitalInc/CompressAI).
The main contribution of this project is to enable the use of compressed images as bottleneck tensors for deep learning downstream applications.

## Usage
### Compress and decompress
The `examples/` directory contains some python scripts that illustrate common use cases of this codec.
These tasks involve encoding and decoding one or more images using the CAE-codec, and a downstream application that involves an adapted Vision Transformer for image classification on compressed images.

### Use the CAE codec as Zarr compressor
First register the `ConvolutionalAutoencoder` class as a valid codec with the `numcodecs.register_codec` function.
For now, only two-dimensional + three-color-channel (RGB) images are supported, since the CAE models have been trained on these kind images.
```
>>> import numcodecs
>>> import caecodec
>>> numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)
# Create a CAE compressor with compression quality of 8.
# Enable the use of GPUs if any is present.
>>> cae_compressor = caecodec.ConvolutionalAutoencoder(quality=8, metric="mse", gpu=True)
```

Once registered, the CAE codec is ready to be used as _compressor_ when creating zarr arrays.
```
>>> import numpy as np
>>> import zarr
>>> a = np.random.randint(0, 256, 512 * 768 * 3, dtype='u1').reshape(512, 768, 3)
>>> z = zarr.array(a, chunks=(256, 256, 3), compressor=cae_compressor)
>>> z.info
Type               : zarr.core.Array
Data type          : uint8
Shape              : (512, 768, 3)
Chunk shape        : (256, 256, 3)
Order              : C
Read-only          : False
Compressor         : ConvolutionalAutoencoder(gpu=False, metric='mse',
                   : quality=8)
Store type         : zarr.storage.KVStore
No. bytes          : 1179648 (1.1M)
No. bytes stored   : 343669 (335.6K)
Storage ratio      : 3.4
Chunks initialized : 6/6
```

The CAE-codec must be registered with the `numcodecs.register_codec` function on any script where images compressed with this codec will be used.

### Compression parameters
There are sixteen configurations for the CAE compressor, involving eight levels of compression quality (`[1, 8]`) and two metrics (`"mse"`, `"ms-ssim"`) for each quality level. These meetrics refer to the distortion metric used to train the models.
The configurations can be specified when creating a compressor object from the `ConvolutionAutoencoder` class.

### Use the compressed representation as a bottleneck tensor for donwstream analysis
This package implements the `BottleneckStore` storage class based on the `zarr.storage.FSStore` class, which allows to open zarr arrays compressed with the `ConvolutionalAutoencoder` codec to retrieve the compressed representation of the image as a tensor.
The main use of this is to enable downstream analysis on images without decompressing the image by using the compressed representation as feature maps generated during the encoding step.
To prevent modifications to the compressed representation of images and possible corruption of their compressed files, the `BottleneckStore` can only be used in read-only mode.

The following shows an example of how the `BottleneckStore` can be used to open a zarr array and retrieve its compressed representations.
```
>>> store = caecodec.BottleneckStore("/path/to/zarr/array/", mode="r")
>>> z_in = zarr.open(store=store)
>>> z_in.info
Type               : zarr.core.Array
Data type          : float32
Shape              : (32, 48, 320)
Chunk shape        : (32, 48, 320)
Order              : C
Read-only          : False
Compressor         : ConvolutionalAutoencoder(bottleneck_channels=320,
                   : downsampling_factor=16, gpu=False, metric='mse',
                   : partial_decompress=True, patch_size=256, quality=8)
Store type         : caecodec.storages.BottleneckStore
No. bytes          : 1966080 (1.9M)
No. bytes stored   : 94111 (91.9K)
Storage ratio      : 20.9
Chunks initialized : 1/1
```
