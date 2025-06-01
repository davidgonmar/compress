from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="compress",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)