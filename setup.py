import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='plot_metric',
    version='0.0.1',
    scripts=['plot_metric'],
    author="Yohann Lereclus",
    author_email="lereclus84L@gmail.com",
    description="A package with tools for plotting metrics",
    long_description="A package with tools for plotting metrics",
    long_description_content_type="text/markdown",
    url="https://github.com/yohann84L/plot_metric/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
