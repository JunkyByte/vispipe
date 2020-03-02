import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VisPipe",
    version="0.0.1",
    author="JunkyByte",
    author_email="adriano.donninelli@hotmail.it",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JunkyByte/vispipe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
