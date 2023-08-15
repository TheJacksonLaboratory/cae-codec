# Examples using the Convolutional Autoecoder (CAE)-Codec
## Encode (compress) images
A compression pipeline example is available in the _encode.py_ script. This pipeline opens an image stored in Zarr format, or any format readable with PIL, and saves it into a compressed representation in Zarr format.

### Compressing a PNG image using the CAE-codec
This line execute the script to compress a PNG image file using the codec at quality 6.
```
python encode.py -i /path/to/image.png -o /path/to/output.zarr -c CAE -q 6
```

### Compressing an image stored in Zarr format using the CAE-codec
The following line executes the script to compress an image stored in a "0/0" sub-group whithin a Zarr file by specifying it with the `-ig 0/0` argument. The `-pb` option enables a progress bar to visualize the compression process.
The highest quality model (`-q 8`) is used this time.
```
python encode.py -i /path/to/image.zarr -ig 0/0 -o /path/to/output.zarr -c CAE -q 8 -pb
```

The full list of _encode.py_ options can be displayed by executing the following line.
```
python encode.py -h
```

## Decode (decompress) images
An example decompression pipeline is available in the _decode.py_ script. This scripts can open a compressed representation of an image, reconstruct it and store it as any image file format supported by PIL.
```
python decode.py -i /path/to/image.zarr -o /path/to/output/directory -f png
```

The list of _decode.py_ options can be displayed by executing the following line.
```
python decode.py -h
```

## Decode into bottleneck tensor
The _decode\_bottleneck.py_ script implements an example use case for the _BottleneckStore_ storage class. This script takes an ecoded image and saves the bottleneck tensor as a Numpy _ndarray_ (.npy) file.
```
python decode_bottlenech.py -i /path/to/encoded.zarr -o /path/to/output/directory
```

## Downstream analysis example
A use case for the bottleneck tensor is implemented in the _downstream\_analysis.py_ script. This script shows how the compressed representation of an image can be used as feature maps for subsequent deep learning model layers. This example illustrates how to adapt a Vision Transformer (based on the  [torchvision](https://pytorch.org/vision/stable/index.html)'s [`vit_B_16`](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16) architecture) to use the compressed representation of an image as feature tensors for image classification.
```
python downstream_analysis.py -i /path/to/encodec.zarr
```

A pre-trained model can be used to initialize the vision transformer model by passing the checkpoint weights with the `-m` parameter.
```
python downstream_analysis.py -i /path/to/encodec.zarr -m /path/to/pretrained/model.pth
```
