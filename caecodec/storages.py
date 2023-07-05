import zarr


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

        self._meta = zarr.util.json_loads(self[".zarray"])

        output_meta = self._meta.copy()

        output_meta["compressor"]["partial_decompress"] = True

        output_meta["chunks"] = [
            self._meta["chunks"][0] 
            // output_meta["compressor"]["downsampling_factor"],
            self._meta["chunks"][1]
            // output_meta["compressor"]["downsampling_factor"],
            output_meta["compressor"]["bottleneck_channels"]
        ]

        output_meta["shape"] = [
            self._meta["shape"][0]
            // output_meta["compressor"]["downsampling_factor"],
            self._meta["shape"][1]
            // output_meta["compressor"]["downsampling_factor"],
            output_meta["compressor"]["bottleneck_channels"]
        ]

        output_meta["dtype"] = "|f4"

        self[".zarray"] = zarr.util.json_dumps(output_meta)

    def __del__(self):
        # Restore the .zarray to its original values
        self._meta["compressor"]["partial_decompress"] = False
        self[".zarray"] = zarr.util.json_dumps(self._meta)
