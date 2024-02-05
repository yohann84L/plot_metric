import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='plot_metric',
    version='0.0.6',
    scripts=['plot_metric_package'],
    install_requires=[
        "scipy>=1.12.0",
        "matplotlib>=3.8.2",
        "colorlover>=0.3.0",
        "pandas>=2.2.0",
        "seaborn>=0.13.2",
        "numpy>=1.26.3",
        "scikit_learn>=1.4.0",
    ],
    author="Yohann Lereclus",
    author_email="lereclus84L@gmail.com",
    description="A package with tools for plotting metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yohann84L/plot_metric/",
    packages=setuptools.find_packages(),
    py_modules=['plot_metric/functions'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
