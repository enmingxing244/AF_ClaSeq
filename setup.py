from setuptools import setup, find_packages
import os

# Get full path to source directory
src_dir = os.path.join("/fs/ess/PAA0203/xing244/AF_Vote/src/af_vote")

setup(
    name="af_vote",
    version="0.1", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "biopython", 
        "tqdm"
    ],
    python_requires=">=3.6",
    author="Xing",
    description="Package for analyzing AlphaFold predictions through voting",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_data={
        'af_vote': [os.path.join(src_dir, '*')]
    }
)
