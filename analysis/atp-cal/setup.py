import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atp_cal",
    version="0.0.1",
    author="Ana Duarte",
    author_email="aduarte {at} caltech {dot} edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/mrazomej/fit_seq.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 