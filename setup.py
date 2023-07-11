import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="caecodec",
    version=os.environ.get('VERSION', '0.0.0'),
    maintainer="Fernando Cervantes",
    maintainer_email="fernando.cervantes@jax.org",
    description="Convolutional Autoencoder (CAE) codec for image compression and storing in NGFF (Zarr) format.",

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheJacksonLaboratory/caecodec",
    packages=setuptools.find_packages(),
    install_requires=[
        'numcodecs>=0.11.0',
        'imagecodecs>=2021.8.26',
        'compressai>=1.2.4',
    ],
    python_requires='>=3.7'
)