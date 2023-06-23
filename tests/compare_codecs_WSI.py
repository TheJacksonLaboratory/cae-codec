import argparse
import functools
import itertools

import os
import shutil
import logging

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.callbacks import Callback
import numpy as np
import torch
import zarr

from imagecodecs.numcodecs import Jpeg2k, Jpeg
import numcodecs
import caecodec

numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Jpeg)
numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)

from time import perf_counter
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity
from pytorch_msssim import ms_ssim

available_metrics = ["rate", "rmse", "msssim", "ssim", "psnr", "delta_cie76",
                     "comp_time",
                     "decomp_time"]


def parse_filenames_list(filenames_list, input_format):
    if (isinstance(filenames_list, str)
      and not (filenames_list.lower().endswith(input_format.lower())
             or filenames_list.lower().endswith(".txt"))):
        return []

    if (isinstance(filenames_list, str)
      and filenames_list.lower().endswith(input_format.lower())):
        return [filenames_list]

    if (isinstance(filenames_list, str)
      and filenames_list.lower().endswith(".txt")):
        with open(filenames_list, "r") as fp:
            filenames_list = [fn.strip("\n ") for fn in  fp.readlines()]

    if isinstance(filenames_list, list):
        filenames_list = functools.reduce(lambda l1, l2: l1 + l2,
                                          map(parse_filenames_list,
                                              filenames_list,
                                              itertools.repeat(input_format)),
                                          [])
    return filenames_list


def compress_image(codec, quality, input_filename, output_filename,
                   patch_size=512,
                   data_group='0/0',
                   progress_bar=False,
                   gpu=False):
    if progress_bar:
        progress_callback = ProgressBar
    else:
        progress_callback = Callback

    if "CAE" in codec:
        compressor = caecodec.ConvolutionalAutoencoder(quality=quality,
                                                       metric="mse",
                                                       gpu=gpu)
    elif "Blosc" in codec:
        compressor = numcodecs.Blosc(clevel=quality)
    elif "Jpeg2k" in codec:
        compressor = Jpeg2k(level=quality)
    elif "Jpeg" in codec:
        compressor = Jpeg(level=quality)
    elif "None" in codec:
        compressor = None
    else:
        raise ValueError("Codec %s not supported" % codec)

    z = da.from_zarr(input_filename, component=data_group)
    z = z[0, :, 0].transpose(1, 2, 0)
    z = z.rechunk(chunks=(patch_size, patch_size, 3))

    if not len(data_group):
        data_group = "0/0"

    with progress_callback():
        z.to_zarr(output_filename, component=data_group, overwrite=True,
                  compressor=compressor)


def rgb2CIELab(x):
    x_lab = rgb2lab(x)
    return x_lab


def ssim_block_fun(x, x_r):
    ssim_bkl = structural_similarity(x[0], x_r[0], channel_axis=2)
    return np.array([[ssim_bkl]], dtype=np.float64)


def ms_ssim_block_fun(x, x_r):
    ms_ssim_bkl = ms_ssim(
        torch.from_numpy(np.moveaxis(x_r[0], -1, 0)[np.newaxis]).float(),
        torch.from_numpy(np.moveaxis(x[0], -1, 0)[np.newaxis]).float(),
        data_range=255)
    return np.array([[ms_ssim_bkl]], dtype=np.float64)


def compute_metrics(x, lab_x, x_r, progress_bar=False):
    if progress_bar:
        progress_callback = dask.diagnostics.ProgressBar
    else:
        progress_callback = dask.callbacks.Callback

    # Compute mean squared error, that is used to measure PSNR and RMSE
    decomp_time = perf_counter()
    with progress_callback():
        _ = x_r.sum().compute()
    decomp_time = perf_counter() - decomp_time

    with progress_callback():
        mse = ((x - x_r) ** 2).mean().compute()

    psnr = 10 * np.log10((255 ** 2) / mse)
    rmse = np.sqrt(mse)

    # Compute structural similarity measure index
    ssim_blocks = da.blockwise(ssim_block_fun, 'ij',
                                       x, 'ijk',
                                       x_r, 'ijk',
                                       dtype=np.float64).mean()
    with progress_callback():
        ssim_mean = ssim_blocks.compute()

    msssim_blocks = da.blockwise(ms_ssim_block_fun, 'ij',
                                         x, 'ijk',
                                         x_r, 'ijk',
                                         dtype=np.float32).mean()
    with progress_callback():
        msssim_mean = msssim_blocks.compute()

    # Compute delta E in the CIE Lab space for color distortion
    lab_x_r = x_r.map_blocks(rgb2CIELab,
                             dtype=np.float32,
                             meta=np.empty((), dtype=np.float32))

    with progress_callback():
        delta_cie76 = np.sqrt(((lab_x - lab_x_r) ** 2).sum(axis=-1)).compute()

    delta_cie76 = delta_cie76.mean()

    metrics = dict(
        rmse=rmse,
        msssim=msssim_mean,
        ssim=ssim_mean,
        psnr=psnr,
        delta_cie76=delta_cie76,
        decomp_time=decomp_time
    )

    return metrics


def test_compression_per_image(out_fp, in_filename, codec, quality,
                               ref_codec="Blosc",
                               patch_size=1024,
                               data_group="0/0",
                               output_dir="./",
                               progress_bar=False,
                               gpu=True):
    logger = logging.getLogger('codecs_tests_log')

    basename = os.path.basename(in_filename).split(".zarr")[0]

    ref_output_fn = os.path.join(output_dir,
                                 basename + "_%s.zarr" % ref_codec)

    temp_output_fn = os.path.join(output_dir,
                                  basename + "_%s.zarr" % codec)

    comp_time = perf_counter()
    compress_image(codec, quality, in_filename, temp_output_fn,
                   patch_size=patch_size,
                   data_group=data_group,
                   progress_bar=progress_bar,
                   gpu=gpu)
    comp_time = perf_counter() - comp_time

    x = da.from_zarr(ref_output_fn, component=data_group)
    lab_x = x.map_blocks(rgb2CIELab, dtype=np.float32,
                         meta=np.empty((), dtype=np.float32))

    # Compute compression rate
    z_cmp = zarr.open(temp_output_fn, mode="r")["0/0"]
    rate = 8 * float(z_cmp.nbytes_stored) / (x.shape[0] * x.shape[1])

    x_r = da.from_zarr(temp_output_fn, component="0/0")

    metrics = compute_metrics(x, lab_x, x_r, progress_bar)
    metrics["comp_time"] = comp_time
    metrics["rate"] = rate

    # Log and write the metric values to the output files
    out_fp.write(f"{codec},{quality},{ref_codec},{patch_size},{in_filename}")
    metrics_str = ""
    for k in available_metrics:
        metrics_str += f",{metrics[k]}"

    out_fp.write(metrics_str + "\n")
    logger.info(metrics_str)
    
    # Remove the compressed image generated with the current codec to free
    # space on disk.
    if (temp_output_fn != ref_output_fn
      and os.path.isdir(temp_output_fn)
      and temp_output_fn.endswith(".zarr")):
        shutil.rmtree(temp_output_fn)


def test_codecs(out_fp, in_filename, patch_size=1024, data_group="0/0",
                output_dir="./",
                progress_bar=False,
                gpu=True):
    logger = logging.getLogger('codecs_tests_log')

    # Test all these configurations
    codecs_qualities = {
        "Blosc": [9],
        "CAE": [1, 2, 3, 4, 5, 6, 7, 8],
        "Jpeg": [1, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 99],
        "Jpeg2k": [1, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 99],
    }

    basename = os.path.basename(in_filename).split(".zarr")[0]
    ref_output_fn = os.path.join(output_dir,
                                 basename + "_Blosc.zarr")

    for codec in codecs_qualities.keys():
        for q in codecs_qualities[codec]:
            logger.info(f"Compressing image {in_filename} with codec {codec} "
                        f"at {q} compression quality")
            test_compression_per_image(out_fp, in_filename, codec=codec,
                                       quality=q,
                                       ref_codec="Blosc",
                                       patch_size=patch_size,
                                       data_group=data_group,
                                       output_dir=output_dir,
                                       progress_bar=progress_bar,
                                       gpu=gpu and codec == "CAE")
            out_fp.flush()

    # Remove the reference compressed image to free space on disk.
    if os.path.isdir(ref_output_fn) and ref_output_fn.endswith(".zarr"):
        shutil.rmtree(ref_output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test different codecs in a dataset of "
                                     "images converted into zarr format")
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+",
                        help="Input files to be used for testing the different"
                             " codecs and qualities",
                        required=True)
    parser.add_argument("-if", "--input-format", dest="input_format", type=str,
                        help="Format of the input images used for testing",
                        default=".zarr")
    parser.add_argument("-dg", "--data-group", dest="data_group", type=str,
                        help="Group in the zarr file where the images are stored",
                        default="")
    parser.add_argument("-o", "--output", dest="output_dir", type=str,
                        help="Output directory where to store the results of "
                             "the experiment",
                        default="./")
    parser.add_argument("-li", "--identifier", dest="log_identifier", type=str,
                        help="Identifier added to the output filename "
                             "`metrics.csv`",
                        default="")
    parser.add_argument("-ps", "--patch-size", dest="patch_size", type=int,
                        help="Size of the chunks used to store the zarr files",
                        default=1024)
    parser.add_argument("-pb", "--progress-bar", dest="progress_bar",
                        action="store_true",
                        help="Whether to show progress bars when processing "
                             "the zarr files, or not.",
                        default=False)
    parser.add_argument("-pl", "--print-log", dest="print_log",
                        action="store_true",
                        help="Whether to print log messages on console or not",
                        default=False)
    parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
                        help="Whether to use GPU to accelerate CAE codec "
                             "compression (when available).",
                        default=False)

    args = parser.parse_args()

    in_filenames = parse_filenames_list(args.input, args.input_format)

    logger = logging.getLogger('codecs_tests_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger_fn = os.path.join(args.output_dir,
                             'test_codecs%s.log' % args.log_identifier)
    fh = logging.FileHandler(logger_fn, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.setLevel(logging.DEBUG)

    if args.print_log:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    # Log and write the metric values to the output files
    output_metrics_filename = os.path.join(
        args.output_dir,
        "metrics_codec_tests%s.csv" % args.log_identifier)

    logger.info("Saving metric in %s" % output_metrics_filename)
    out_fp = open(output_metrics_filename, "w")
    out_fp.write("codec,quality,ref_codec,patch_size,in_filename")
    for k in available_metrics:
        out_fp.write(f",{k}")
    out_fp.write("\n")

    for in_fn in in_filenames:
        test_codecs(out_fp, in_fn, patch_size=args.patch_size,
                    data_group=args.data_group,
                    output_dir=args.output_dir,
                    progress_bar=args.progress_bar,
                    gpu=args.gpu)

    out_fp.close()
    logging.shutdown()
