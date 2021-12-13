from setuptools import setup


setup(
    name="coquma-sim-spooler",
    version="0.0.1",
    description="A library for running cold atom simulations.",
    url="https://www.github.com/synqs/coquma-sim-spooler",
    author="Fred Jendrzejewski",
    author_email="fnj@kip.uni-heidelberg.de",
    license="BSD-2",
    packages=["Spooler_files"],
    zip_safe=False,
    install_requires=[
        "numpy",
    ],
)
