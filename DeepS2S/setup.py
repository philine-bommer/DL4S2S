import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepS2S",
    version="1.0.0",
    author="Philine Bommer",
    author_email="pbommer@atb-potsdam.de",
    description="Package for DL-based S2S forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/philine-bommer/DL4S2S",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={'': ['model', 'preprocessing','utils','dataset']}
)