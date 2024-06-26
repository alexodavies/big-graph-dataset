from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name="big-graph-dataset",
    version="0.0.2",
    author="Alex O. Davies",
    author_email="alexander.davies@bristol.ac.uk",
    description="A collection of graph datasets in torch_geometric format.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/neutralpronoun/big-graph-dataset",
    packages=find_packages(),
    py_modules=['utils', 'describe_datasets'],  # Include utils.py
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[parse_requirements('requirements.txt')
        # List your project's dependencies
    ],
)
