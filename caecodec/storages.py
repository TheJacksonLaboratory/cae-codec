import os
import zarr
import json
import requests


class BottleneckStore(zarr.storage.FSStore):
    """Store class that allows to extract the bottleneck tensor from zarr
    arrays compresse wth the ConvolutionalAutoencoder codec.

    Examples
    --------

    Notes
    -----
    Used for read-only operations.
    """
    def __init__(self, url, mode="r", **kwargs):
        if mode != "r":
            raise ValueError("Bottleneck tensors are read-only, please use "
                             "`mode=r` instead")
        super(BottleneckStore, self).__init__(url, mode="r+", **kwargs)

        url_z_array = os.path.join(url, '.zarray')
        try:
            metadata_resp = requests.get(url_z_array)

            if metadata_resp.status_code == 200:
                meta = json.loads(metadata_resp.content.decode("utf-8"))
            else:
                raise ValueError(f"Could not find {url_z_array}")

        except requests.exceptions.MissingSchema:
            if os.path.exists(url_z_array):
                with open(url_z_array, mode="r") as fp:
                    meta = json.load(fp)
            else:
                raise ValueError(f"Could not find {url_z_array}")

        self._output_meta = meta.copy()

        self._output_meta["compressor"]["partial_decompress"] = True

        self._output_meta["chunks"] = [
            meta["chunks"][0]
            // self._output_meta["compressor"]["downsampling_factor"],
            meta["chunks"][1]
            // self._output_meta["compressor"]["downsampling_factor"],
            self._output_meta["compressor"]["bottleneck_channels"]
        ]

        self._output_meta["shape"] = [
            meta["shape"][0]
            // self._output_meta["compressor"]["downsampling_factor"],
            meta["shape"][1]
            // self._output_meta["compressor"]["downsampling_factor"],
            self._output_meta["compressor"]["bottleneck_channels"]
        ]

        self._output_meta["dtype"] = "|f4"

    def __getitem__(self, key):
        if ".zarray" in key:
            return self._output_meta
        else:
            return self[key]
