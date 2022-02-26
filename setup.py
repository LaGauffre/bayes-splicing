import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="bayes-splicing",
    version="0.0.4",
    description="Bayesian fit of splicing models with applications to insurance loss data",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LaGauffre/bayes-splicing",
    author="Pierre-Olivier Goffard",
    author_email="pierre.olivier.goffard@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
    ],
    packages=["bayes_splicing"],
    package_dir={"bayes_splicing": "bayes_splicing"},
    include_package_data=True,
    install_requires=[
        "joblib",
        "numba",
        "numpy>=1.17",
        "scipy>=1.4",
        "matplotlib",
	    "pandas",
	    "seaborn"
	],
)
