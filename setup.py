import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jhammer",
    version="3.0.0",
    author="Dai Jian",
    author_email="daij@stumail.ysu.edu.cn",
    description="My hammer for medical image processing and deap learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.11",
    install_requires=[
        "numpy >= 1.21.0",
        "nibabel >= 4.0.1",
        "scipy >= 1.7.3",
        "medpy >= 0.4.0",
        "SimpleITK >= 2.1.1.2",
        "pydicom >= 2.4.3",
        "dicom2nifti >= 2.4.8",
        "lazyConfig"
    ]
)
