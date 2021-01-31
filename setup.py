import os
from setuptools import setup, find_packages

abspath = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []


setup(

    name='elgfinder',

    version='1.0',

    description='Searching for Emission Line Galaxy in SITELLE Cube',

    long_description=long_description,

    url='https://github.com/NGC4676/SITELLE_ELG_finder',

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy',

    packages=find_packages(include=['elgfinder','elgfinder.']),

    python_requires='>=3.5',

    install_requires=install_requires,

)
