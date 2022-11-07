import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchshow", # Replace with your own username
    version="0.5.0",
    author="Xiaowen Ying",
    author_email="shawnying.inbox@gmail.com",
    description="Visualizing PyTorch tensors with a single line of code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xwying/torchshow",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)