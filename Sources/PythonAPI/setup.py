import sys
from setuptools import setup


def get_long_description():
    with open('README.md') as f:
        rv = f.read()
    return rv


def get_requirements(suffix=''):
    with open('requirements%s.txt' % suffix) as f:
        rv = f.read().splitlines()
    return rv

setup(
    name='GoodAI-BrainSim',
    version='1.0.0',
    url='https://github.com/GoodAI/python-api',
    license='MIT',
    author='vlasy',
    description='Python API for using GoodAI\'s Brain Simulator',
    long_description=get_long_description(),
    install_requires=get_requirements(),
    packages=['goodai', 'goodai.brainsim']
)