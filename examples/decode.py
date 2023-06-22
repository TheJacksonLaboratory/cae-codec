import os
import argparse

import PIL
import zarr

import numcodecs
import caecodec

numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)


def _array2image(arr, filename):
    # Open the input file. This allows any RGB image stored in a format
    # supported by PIL.
    im = PIL.Image.fromarray(arr)
    im.save(filename, quality_opts={'compress_level': 9, 'optimize': False})


def decode(in_filenames, out_filenames, image_groups=None, overwrite=False):

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

    for in_fn, grp, out_fn in zip(in_filenames, image_groups, out_filenames):
        if not overwrite and os.path.isfile(out_fn):
            raise ValueError("The file %s already exists, if you would like to"
                             " overwrite it, run this again using the "
                             "-ow/--overwrite option")

        # Save the compressed representation of the image as Zarr format.
        if len(grp):
            z_in = zarr.open(in_fn, mode="r")[grp]
        else:
            z_in = zarr.open(in_fn)

        arr = z_in[:]

        # Convert the pixel array to an image
        _array2image(arr, out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compress an image into zarr using a "
                                     "Convolutional Autoencoder codec")
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+",
                        help="Input image to decompress",
                        required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, nargs="+",
                        help="Output filenames or a single directory where to "
                             "store the decompressed images.",
                        default=["./"])
    parser.add_argument("-f", "--format", dest="format", type=str,
                        help="Output image format",
                        default="png")
    parser.add_argument("-ig", "--image-group", dest="image_groups", type=str,
                        nargs="+",
                        help="For Zarr files, specify the group where the "
                             "images are stored",
                        default=[""])
    parser.add_argument("-ow", "--overwrite", dest="overwrite",
                        action="store_true", 
                        help="Overwrite existing files with the same output "
                             "name",
                        default=False)

    args = parser.parse_args()

    if not args.format.startswith("."):
        args.format = "." + args.format

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
                                            basename + args.format)

    assert all(map(lambda fn: fn.endswith(args.format), args.output)), \
        "All outputs must have %s extension" % args.format
    assert all(map(lambda fn: fn.endswith(".zarr"), args.input)), \
        "All inputs must have .zarr extension"

    decode(args.input, args.output, image_groups=args.image_groups,
           overwrite=args.overwrite)
