from setuptools import setup, find_packages

setup(
    name="concept_models",
    version="0.1.0",
    description="A collection of concept-based neural network models",
    author="Multiple Authors",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "torchvision",
        "einops",
        "numpy",
        "pillow",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)