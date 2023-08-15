import os
import argparse

import PIL
import numpy as np
import torch
import zarr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.callbacks import Callback

import numcodecs

available_codecs = ["None", "Blosc"]
try:
    from imagecodecs.numcodecs import Jpeg2k, Jpeg
    numcodecs.register_codec(Jpeg2k)
    numcodecs.register_codec(Jpeg)
    available_codecs.append("Jpeg2k")
    available_codecs.append("Jpeg")

except ModuleNotFoundError:
    # Jpeg and Jpeg2k will not be available
    pass

import caecodec

caecodec.ConvolutionalAutoencoder.patch_size = 512
numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)
available_codecs.append("CAE")


def _image2array(filename, image_group):
    # If the input is stored in Zarr format, open the specified `image_group`.
    extension = os.path.basename(filename).split(".")[-1]
    if "zarr" in extension:
        # For now, the axes of the zarr image must be in the order TCZYX.
        arr = da.from_zarr(filename, component=image_group)
        if arr.ndim > 3:
            arr = arr[0, :, 0].transpose(1, 2, 0)
    else:
        # Open the input file. This allows any RGB image stored in a format
        # supported by PIL.
        im = PIL.Image.open(filename)
        arr = da.from_array(np.array(im))

    return arr


def encode(in_filenames, out_filenames, image_groups=None, codec="CAE",
           quality=8,
           metric="mse",
           chunk_size=-1,
           use_gpu=False,
           overwrite=False,
           progress_bar=False):
    # Create a compressor object to use with all input images.
    if codec == "CAE":
        compressor = caecodec.ConvolutionalAutoencoder(quality=quality,
                                                       metric=metric,
                                                       gpu=use_gpu)
    elif codec == "Jpeg":
        compressor = Jpeg(level=quality)
    elif codec == "Jpeg2k":
        compressor = Jpeg2k(level=quality)
    elif codec == "Blosc":
        compressor = zarr.Blosc(clevel=quality)
    else:
        raise ValueError("Codec %s is not implemented" % codec)

    if not isinstance(in_filenames, (list, tuple)):
        in_filenames = [in_filenames]

    if not isinstance(out_filenames, (list, tuple)):
        out_filenames = [out_filenames]

    if image_groups is None:
        image_groups = ""

    if not isinstance(image_groups, (list, tuple)):
        image_groups = [image_groups] * len(in_filenames)

    assert len(in_filenames) == len(out_filenames), \
        "The same number of inputs and outputs was expected"
    assert len(in_filenames) == len(image_groups), \
        "The same number of image groups and inputs was expected"

    if progress_bar:
        progress_callback = ProgressBar
    else:
        progress_callback = Callback

    for in_fn, grp, out_fn in zip(in_filenames, image_groups, out_filenames):
        # Convert the image to a pixel array
        x = _image2array(in_fn, grp)
        x = da.rechunk(x, chunks=(chunk_size, chunk_size, 3))

        if not overwrite and os.path.isdir(out_fn):
            raise ValueError("The file %s already exists, if you would like to"
                             " overwrite it, run this again using the "
                             "-ow/--overwrite option")

        # Save the compressed representation of the image as Zarr format.
        with progress_callback():
            x.to_zarr(out_fn, component=grp if len(grp) else None,
                      compressor=compressor,
                      overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compress an image into zarr using a "
                                     "Convolutional Autoencoder codec")
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+",
                        help="Input image to compress",
                        required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, nargs="+",
                        help="Output filenames or a single directory where to "
                             "store the compressed images. If specific output "
                             "filenames are passed, these must have .zarr "
                             "extension",
                        default=["./"])
    parser.add_argument("-ig", "--image-group", dest="image_groups", type=str,
                        nargs="+",
                        help="For Zarr files, specify the group where the "
                             "images are stored",
                        default=[""])
    parser.add_argument("-c", "--codec", dest="codec", type=str, 
                        help="Codec method to compress the images",
                        choices=available_codecs,
                        default="CAE")
    parser.add_argument("-q", "--quality", dest="quality", type=int, 
                        default=8)
    parser.add_argument("-m", "--metric", dest="metric", type=str, 
                        help="Metric used when the mnodel was trained (only CAE"
                             " codec)",
                        choices=["mse", "ms-ssim"],
                        default="mse")
    parser.add_argument("-cs", "--chunk-size", dest="chunk_size", type=int, 
                        help="Size of the chunks used to store the image data",
                        default=1024)
    parser.add_argument("-pb", "--progress-bar", dest="progress_bar",
                        action="store_true", 
                        help="Display progress bar as the image is being "
                             "compressed",
                        default=False)
    parser.add_argument("-g", "--gpu", dest="use_gpu", action="store_true", 
                        help="Use GPU to accelerate compression process (when "
                             "available)",
                        default=False)
    parser.add_argument("-ow", "--overwrite", dest="overwrite",
                        action="store_true", 
                        help="Overwrite existing files with the same output "
                             "name",
                        default=False)

    args = parser.parse_args()

    args.use_gpu = args.use_gpu and torch.cuda.is_available()

    n_in = len(args.input)
    n_out = len(args.output)

    assert (n_out == 1 or n_out == n_in), \
        ("Expected the same number of output filenames, or a single directory "
         "for all inputs")
    assert ((n_in == 1 and n_out == 1)
          or (n_in > 1 and (n_out == 1 or n_out == n_in))), \
            ("A single output directory, or the same number of input and "
             "output filenames was expected")
    assert ((n_in == 1 and n_out == 1) 
          or (n_in > 1 and args.output[0].endswith(args.format))), \
            ("The specified output is not a directory, and cannot be used for "
             "multiple inputs")

    if n_out == 1:
        args.output = [args.output[0]] * n_in

    for idx in range(n_in):
        if not args.output[idx].endswith(".zarr"):
            basename = os.path.basename(args.input[idx])
            basename = ".".join(basename.split(".")[:-1])
            args.output[idx] = os.path.join(args.output[idx],
                                            basename + ".zarr")

    assert all(map(lambda fn: fn.endswith(".zarr"), args.output)), \
        "All outputs must have .zarr extension"

    encode(args.input, args.output, image_groups=args.image_groups,
           codec=args.codec,
           quality=args.quality,
           metric=args.metric,
           chunk_size=args.chunk_size,
           use_gpu=args.use_gpu,
           progress_bar=args.progress_bar,
           overwrite=args.overwrite)
