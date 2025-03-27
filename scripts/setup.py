from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pglandsseg",
    python_requires=">=.9",
    packages=find_packages(),
    install_requires=requirements,
)
