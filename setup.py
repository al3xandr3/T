import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="T-is-for-Table", # Replace with your own username
    version="0.1",
    author="Alexandre Matos Martins",
    author_email="al3xandr3@gmail.com",
    description="T extends Pandas Dataframes with a collection of table manipulation methods as well as statistical, machine learning, financial and EDA methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/al3xandr3/T",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
          'pandas', 'numpy', 'matplotlib', 
          'numba', 'seaborn', 'scipy', 
          'datetime'
      ],
)