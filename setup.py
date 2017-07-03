import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering"
    ],
    extras_require={
        "test": [
            "codecov",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-pep8",
            "pytest-runner"
        ],
    },
    install_requires=[
        "imblearn",
        "keras",
        "keras-resnet"
    ],
    license="BSD",
    name="keras-microscopy",
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/keras-microscopy",
    version="0.0.1"
)
