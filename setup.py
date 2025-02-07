from setuptools import setup, find_packages
import os

# Get full path to source directory
src_dir = os.path.join("/fs/ess/PAA0203/xing244/AF_ClaSeq/src/af_claseq")

setup(
    name="af_claseq",
    version="1.0.0", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "biopython", 
        "tqdm"
    ],
    python_requires=">=3.10",
    author="Enming Xing",
    description="Package for analyzing AlphaFold predictions through ClaSeq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_data={
        'af_claseq': [os.path.join(src_dir, '*')]
    }
)
