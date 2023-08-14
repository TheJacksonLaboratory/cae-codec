import os
import zarr
import json
import requests


class BottleneckStore(zarr.storage.FSStore):
    """Store class that allows to extract the bottleneck tensor from zarr
    arrays compressed with the ConvolutionalAutoencoder codec.

    Notes
    -----
    Used for read-only operations.
    """
    def __init__(self, url, mode="r", **kwargs) -> None:
        """
        Parameters:
        -----------
        url : str
            The destination to map.
        mode : str
            Mode is overriden to ``r`` because this storage class is for
            read-only operations.
        kwargs : dict
            FSStore class arguments.
        """
        if mode != "r":
            raise ValueError("Bottleneck tensors are read-only, please use "
                             "`mode=r` instead")
        super(BottleneckStore, self).__init__(url, mode="r", **kwargs)
        self._url = url
        self._output_meta = None

    def _enable_bottleneck_keys(self, key) -> None:
        """Open the .zarray file and modify the fields related to the
        compressor and output shape in order to enable retreival of bottleneck
        tensors.
        """
        if self._output_meta is not None:
            return

        url_z_array = self._url + "/" + key
        try:
            metadata_resp = requests.get(url_z_array)

            if metadata_resp.status_code == 200:
                meta = json.loads(metadata_resp.content.decode("utf-8"))
            else:
                raise ValueError(f"Could not find {url_z_array}")

        except (requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema):
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
            self._enable_bottleneck_keys(key)
            return self._output_meta
        else:
            return self[key]
