import argparse

import os
import shutil
import logging

import PIL
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

from time import perf_counter
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity
from pytorch_msssim import ms_ssim

from skimage import morphology, color, filters, transform

numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Jpeg)
caecodec.ConvolutionalAutoencoder.patch_size = 1024
numcodecs.register_codec(caecodec.ConvolutionalAutoencoder)


available_metrics = ["height", "width", "channels", "rate", "rmse", "msssim",
                     "ssim",
                     "psnr",
                     "delta_cie76",
                     "comp_time",
                     "decomp_time"]
available_codecs = ["CAE", "Jpeg", "Jpeg2k", "Blosc", "None"]


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


def compute_mask(chunk, mask_scale=1.0, min_size=16, area_threshold=128,
                 thresh=None):
    gray = color.rgb2gray(chunk)
    scaled_gray = transform.rescale(gray, scale=mask_scale, order=0,
                                    preserve_range=True)

    if thresh is None:
        thresh = filters.threshold_otsu(scaled_gray)

    chunk_mask = scaled_gray > thresh
    chunk_mask = morphology.remove_small_objects(
        chunk_mask == 0, min_size=min_size ** 2, connectivity=2)
    chunk_mask = morphology.remove_small_holes(
        chunk_mask, area_threshold=area_threshold ** 2)
    chunk_mask = morphology.binary_dilation(
        chunk_mask, morphology.disk(min_size))

    chunk_mask = chunk_mask.sum(keepdims=True) > 0

    return chunk_mask


def mask_tissue(chunk, mask, block_info=None):
    m_i, m_j, _ = block_info[0]['chunk-location']
    if mask[m_i, m_j]:
        return chunk
    else:
        return np.full_like(chunk, fill_value=255)


def encode(in_filenames, out_filenames, image_groups=None, codec="CAE",
           quality=8,
           metric="mse",
           chunk_size=-1,
           mask=None,
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

        if mask is not None:
            x = da.map_blocks(mask_tissue, x, mask,
                              dtype=np.uint8,
                              meta=np.empty((0, 0, 0), dtype=np.uint8))

        if not overwrite and os.path.isdir(out_fn):
            raise ValueError("The file %s already exists, if you would like to"
                             " overwrite it, run this again using the "
                             "-ow/--overwrite option")

        # Save the compressed representation of the image as Zarr format.
        with progress_callback():
            x.to_zarr(out_fn, component=grp if len(grp) else None,
                      compressor=compressor,
                      fill_value=255,
                      overwrite=overwrite)


def rgb2CIELab(x):
    x_lab = rgb2lab(x)
    return x_lab


def ssim_block(x, x_r, mask, block_info=None):
    m_i, m_j, _ = block_info[0]['chunk-location']
    if mask[m_i, m_j]:
        ssim_bkl = structural_similarity(x, x_r, channel_axis=2)
    else:
        ssim_bkl = 0

    ssim_bkl = np.array([[ssim_bkl]], dtype=np.float64)
    return ssim_bkl


def ms_ssim_block(x, x_r, mask, block_info=None):
    m_i, m_j, _ = block_info[0]['chunk-location']

    if mask[m_i, m_j]:
        x_pad = np.copy(x)
        x_r_pad = np.copy(x_r)
        x_pad = np.moveaxis(x_pad, -1, 0)[np.newaxis]
        x_r_pad = np.moveaxis(x_r_pad, -1, 0)[np.newaxis]

        ms_ssim_bkl = ms_ssim(
            torch.from_numpy(x_r_pad).float(),
            torch.from_numpy(x_pad).float(),
            data_range=255)

    else:
        ms_ssim_bkl = 0

    ms_ssim_bkl = np.array([[ms_ssim_bkl]], dtype=np.float64)

    return ms_ssim_bkl


def mse_block(x, x_r, mask, block_info=None):
    m_i, m_j, _ = block_info[0]['chunk-location']

    if mask[m_i, m_j]:
        se = np.mean((x - x_r) ** 2).reshape(1, 1)
    else:
        se = np.zeros((1, 1), dtype=np.float64)

    return se


def delta_cielab_block(lab_x, lab_x_r, mask, block_info=None):
    m_i, m_j, _ = block_info[0]['chunk-location']

    if mask[m_i, m_j]:
        lab_se = np.sum((lab_x - lab_x_r) ** 2, axis=2)
        lab_rse = np.mean(np.sqrt(lab_se), keepdims=True)

    else:
        lab_rse = np.zeros((1, 1), dtype=np.float64)

    return lab_rse


def compute_metrics(x, lab_x, x_r, mask, progress_bar=False):
    logger = logging.getLogger('codecs_tests_log')
    if progress_bar:
        progress_callback = dask.diagnostics.ProgressBar
    else:
        progress_callback = dask.callbacks.Callback

    # Compute the sum as an example of a function that iterates on the image
    # and requires its decompression.
    logger.debug("Computing sum of all values to estimate decompression time")
    decomp_time = perf_counter()
    with progress_callback():
        _ = x_r.sum().compute()
    decomp_time = perf_counter() - decomp_time

    m_h, m_w = mask.shape
    mask_sum = mask.sum()
    mask_ratio = (m_h * m_w) / mask_sum

    # Compute mean squared error, that is used to measure PSNR and RMSE
    logger.debug("Computing MSE")
    mse_blks = da.map_blocks(mse_block, x, x_r, mask,
                             chunks=(1, 1),
                             drop_axis=(2,),
                             dtype=np.float64,
                             meta=np.empty((0, 0), dtype=np.float64))
    mse_blks = mse_blks.sum()
    with progress_callback():
        mse_sum = mse_blks.compute()
    mse = mse_sum / mask_sum

    logger.debug("Computing PSNR")
    psnr = 10 * np.log10((255 ** 2) / mse)

    logger.debug("Computing RMSE")
    rmse = np.sqrt(mse)

    # Compute structural similarity measure index
    ssim_blks = da.map_blocks(ssim_block, x, x_r, mask,
                              dtype=np.float64,
                              chunks=(1, 1),
                              drop_axis=(2,),
                              meta=np.empty((0,), dtype=np.float64))
    ssim_blks = ssim_blks.sum()
    logger.debug("Computing mean SSIM")
    with progress_callback():
        ssim_sum = ssim_blks.compute()
    ssim_mean = ssim_sum / mask_sum

    ms_ssim_blks = da.map_blocks(ms_ssim_block, x, x_r, mask,
                                 dtype=np.float32,
                                 chunks=(1, 1),
                                 drop_axis=(2,),
                                 meta=np.empty((0,), dtype=np.float32))
    ms_ssim_blks = ms_ssim_blks.sum()

    logger.debug("Computing mean MS-SSIM")
    with progress_callback():
        ms_ssim_sum = ms_ssim_blks.compute()
    ms_ssim_mean = ms_ssim_sum / mask_sum

    # Compute delta E in the CIE Lab space for color distortion
    lab_x_r = x_r.map_blocks(rgb2CIELab,
                             dtype=np.float32,
                             meta=np.empty((), dtype=np.float32))
    logger.debug("Computing Error in CIE Lab color space")
    delta_cie76_blks = da.map_blocks(delta_cielab_block, lab_x, lab_x_r, mask,
                                     chunks=(1, 1),
                                     drop_axis=(2,),
                                     dtype=np.float64,
                                     meta=np.empty((0, 0), dtype=np.float64))
    delta_cie76_blks = delta_cie76_blks.sum()
    with progress_callback():
        delta_cie76_sum = delta_cie76_blks.compute()

    delta_cie76 = delta_cie76_sum / mask_sum

    metrics = dict(
        rmse=rmse,
        msssim=ms_ssim_mean,
        ssim=ssim_mean,
        psnr=psnr,
        delta_cie76=delta_cie76,
        decomp_time=decomp_time
    )

    return metrics


def test_compression(out_fp, in_filename, codec, quality,
                     chunk_size=1024,
                     image_group="",
                     output_dir="./",
                     progress_bar=False,
                     gpu=True):
    logger = logging.getLogger('codecs_tests_log')
    if progress_bar:
        progress_callback = dask.diagnostics.ProgressBar
    else:
        progress_callback = dask.callbacks.Callback

    if not len(image_group):
        image_group = None

    source_format = os.path.basename(in_filename).split(".")[-1]
    basename = os.path.basename(in_filename).split(source_format)[0]

    x = _image2array(in_filename, image_group)
    x = da.rechunk(x, (chunk_size, chunk_size, 3))
    lab_x = x.map_blocks(rgb2CIELab, dtype=np.float32,
                         meta=np.empty((), dtype=np.float32))

    mask = x.map_blocks(compute_mask,
                        mask_scale=1/16,
                        min_size=16,
                        area_threshold=128,
                        chunks=(chunk_size // 16, chunk_size // 16),
                        drop_axis=(2,),
                        dtype=bool,
                        meta=np.empty((0, 0), dtype=bool))

    with progress_callback():
        mask = mask.compute()

    temp_output_fn = os.path.join(output_dir,
                                  basename + "_%s.zarr" % codec)

    logger.info(f"Compressing {in_filename} with {codec} at quality {quality}")
    comp_time = perf_counter()
    encode(in_filename, temp_output_fn, image_groups=image_group,
           codec=codec,
           quality=quality,
           metric="mse",
           chunk_size=chunk_size,
           mask=mask,
           use_gpu=gpu,
           overwrite=True,
           progress_bar=progress_bar)
    comp_time = perf_counter() - comp_time

    # Compute compression rate
    z_cmp = zarr.open(temp_output_fn, mode="r")
    if image_group:
        z_cmp = z_cmp[image_group]
    rate = 8 * float(z_cmp.nbytes_stored) / (x.shape[0] * x.shape[1])

    x_r = da.from_zarr(temp_output_fn, component=image_group)
    x_r = da.rechunk(x_r, (chunk_size, chunk_size, 3))
    metrics = compute_metrics(x, lab_x, x_r, mask, progress_bar)

    metrics["height"] = x_r.shape[0]
    metrics["width"] = x_r.shape[1]
    metrics["channels"] = x_r.shape[2]

    metrics["comp_time"] = comp_time
    metrics["rate"] = rate

    # Log and write the metric values to the output files
    out_fp.write(f"{codec},{quality},{chunk_size},{in_filename}")
    metrics_str = ""
    for k in available_metrics:
        metrics_str += f",{metrics[k]}"

    out_fp.write(metrics_str + "\n")
    logger.info(metrics_str)

    # Remove the compressed image generated with the current codec to free
    # space on disk.
    if (temp_output_fn != in_filename
       and os.path.isdir(temp_output_fn)
       and temp_output_fn.endswith(".zarr")):
        shutil.rmtree(temp_output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test different codecs in a dataset of "
                                     "images converted into zarr format")
    parser.add_argument("-i", "--input", dest="input", type=str,
                        help="Input image file",
                        required=True)
    parser.add_argument("-ig", "--image-group", dest="image_group", type=str,
                        help="Group in the zarr file where the images are "
                             "stored.",
                        default="")
    parser.add_argument("-o", "--output", dest="output_dir", type=str,
                        help="Output directory where to store the results of "
                             "the experiment.",
                        default="./")
    parser.add_argument("-c", "--codec", dest="codec", type=str,
                        help="Testing codec",
                        choices=available_codecs,
                        default=available_codecs[0])
    parser.add_argument("-q", "--quality", dest="quality", type=int,
                        help="Codec compression quality.",
                        default=8)
    parser.add_argument("-li", "--identifier", dest="log_identifier", type=str,
                        help="Identifier added to the output filename "
                             "`metrics.csv`.",
                        default="")
    parser.add_argument("-cs", "--chunk-size", dest="chunk_size", type=int,
                        help="Size of the chunks used to store the image "
                             "data.",
                        default=1024)
    parser.add_argument("-pb", "--progress-bar", dest="progress_bar",
                        action="store_true",
                        help="Whether to show progress bars when compressing "
                             "and comparing the codec or not.",
                        default=False)
    parser.add_argument("-pl", "--print-log", dest="print_log",
                        action="store_true",
                        help="Whether to print log messages on console or "
                             "not.",
                        default=False)
    parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
                        help="Whether to use GPU to accelerate CAE codec "
                             "compression (when available).",
                        default=False)

    args = parser.parse_args()

    logger = logging.getLogger('codecs_tests_log')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger_fn = os.path.join(args.output_dir,
                             'test_codecs_%s_%i%s.log' % (args.codec,
                                                          args.quality,
                                                          args.log_identifier))
    fh = logging.FileHandler(logger_fn, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.print_log:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

    # Log and write the metric values to the output files
    output_metrics_filename = os.path.join(
        args.output_dir,
        "metrics_%s_%i%s.csv" % (args.codec, args.quality,
                                 args.log_identifier))

    out_fp = open(output_metrics_filename, "w")
    logger.info(f"Saving metric in {output_metrics_filename} {out_fp}")
    out_fp.write("codec,quality,chunk_size,in_filename")
    for k in available_metrics:
        out_fp.write(f",{k}")
    out_fp.write("\n")

    basename = os.path.basename(args.input).split(".")[0]

    logger.info(f"Compressing image {args.input} with codec {args.codec} "
                f"at {args.quality} compression quality")
    test_compression(out_fp, args.input, codec=args.codec,
                     quality=args.quality,
                     chunk_size=args.chunk_size,
                     image_group=args.image_group,
                     output_dir=args.output_dir,
                     progress_bar=args.progress_bar,
                     gpu=args.gpu and args.codec == "CAE")

    out_fp.close()
    logging.shutdown()
