import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='plot_metric',
    version='0.0.3',
    scripts=['plot_metric_package'],
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
