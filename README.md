# CAE-Codec
Convolutional Autoencoder (CAE) codec for image compression in Zarr format.

The CAE extends the Factorized Prior model from Balle, et. al. (2018) ["Variational Image Compression with a Scale Hyperprior"](https://arxiv.org/abs/1802.01436) to be used as chunk compressor to store images in [zarr](https://zarr.readthedocs.io) format.
The original convolutional autoencoder models come from the [Compress-AI library](https://github.com/InterDigitalInc/CompressAI).
This package allows also to use the compressed representation of images as a feature map for machine learning downstream analsysis.

## Usage
### Compress and decompress
The `examples/` directory contains the python scripts to convert one or more images to the `zarr` format, and vice versa, using the CAE codec as compressor.

### Use the CAE codec as Zarr compressor
First register the `ConvolutionalAutoencoder` class as a valid codec with the `numcodecs.register_codec` function.
Now the CAE codec can be used as compressor to store image-like arrays in zarr format.
For now, only two-dimensional + three-channel (RGB) images are supported, since the CAE model has only been trained on these kind images.
```
>>> import numcodecs
>>> import caecodec
>>> numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)
# Create a CAE compressor with compression quality of 8.
# Enable the use of GPUs if any is present.
>>> cae_compressor = caecodec.ConvolutionalAutoencoder(quality=8, metric="mse", gpu=True)
```

Then use the CAE codec as `compressor` when creating the zarr array.
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

There are sixteen configurations for the CAE compressor, involving eight levels of compression quality (`[1, 8]`) and two metrics (`"mse"`, `"ms-ssim"`) for each quality level. These parameters refer to the reconstruction distortion performance metric used when the models were trained.


### Use the compressed representation as a feature map for donwstream analysis
This package implements the `BottleneckStore` based on the `zarr.storage.FSStore` class, which allows to open zarr arrays compressed with the `ConvolutionalAutoencoder` codec to retrieve the compressed representation of the image.
The main use of this functionality is to carry out downstream analysis on images without the need to decompress the image, since the compressed representation can be considered as a feature map representing the original image.
To prevent modiying the compressed representation of images, the `BottleneckStore` can only be used in read-only mode.

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
