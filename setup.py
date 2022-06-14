import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimal_input",
    version="0.0.1",
    author="Akshita Ramya",
    author_email="akamsali@purdue.edu",
    description="Optimal Input for ASR models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akamsali/optimal-input",
    packages=setuptools.find_packages(),
    package_data={
        'opt_inp': [
            'conf/params.yaml'
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'sentencepiece', 'transformers'
    ],
)