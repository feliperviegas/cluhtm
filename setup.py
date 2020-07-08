from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools import setup, find_packages
import atexit

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='cluhtm',
    version='0.2.1',
    setup_requires=[],
    packages=find_packages('src'),
    install_requires=requirements,
    package_dir={'': 'src'},
    namespace_packages=['cluhtm'],
)
