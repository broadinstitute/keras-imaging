import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering"
    ],
    extras_require={
        "test": [
            "codecov",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pytest-pep8",
            "pytest-runner"
        ],
    },
    install_requires=[
        "futures",
        "imblearn",
        "keras",
        "keras-resnet",
        "numpy",
        "scikit-image",
        "six"
    ],
    license="BSD",
    name="keras-imaging",
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/keras-imaging",
    version="0.0.1"
)
