import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mutransformers",
    version="0.1.0",
    author="Greg Yang, Edward J Hu",
    author_email="gregyang@microsoft.com, edward.hu@umontreal.ca",
    description="some Huggingface transformers reparametrized in muP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/mutransformers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)