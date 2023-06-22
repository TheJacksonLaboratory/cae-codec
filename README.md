# CAE-Codec
Convolutional Autoencoder (CAE) codec for image compression and storing in NGFF (Zarr) format.

The CAE extends the Factorized Prior model from Balle, et. al. (2018) `"Variational Image Compression with a Scale Hyperprior" <https://arxiv.org/abs/1802.01436>` to be used as chunk compressor to store images in `zarr` format.

## Usage
### Compress and decompress
The `examples/` directory contains the python scripts to convert one or more images to the `zarr` format, and vice versa, using the CAE codec as compressor.


### Use the CAE codec as Zarr compressor
First register the ConvolutionalAutoencoder class as a valid codec with the `numcodecs.register_codec` function. Now the CAE codec can be used as compressor to store image-like arrays in zarr format.
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

