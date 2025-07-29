import setuptools

with open("README.md", "r") as fh: 
    long_description = fh.read()

setuptools.setup(
    name="pcassie",
    version="0.1",
    author="Kenny T. Phan",
    author_email="kenny.phan@yale.edu",
    description="A Pincipal Component Pipeline based on Damiano et al. 2019",
    packages=["pcassie"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",   
        "Intended Audience :: Science/Research", 
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ]
)

install_requires = ["numpy", "matplotlib.pyplot", "scipy", "jax", "pandas", "numba", "astropy"]

