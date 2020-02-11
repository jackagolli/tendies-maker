#from distutils.core import setup
from setuptools import setup

setup(
    name='tendies-maker',
    version='1.0a',
    packages=['stocks'],
    url='',
    license='',
    author='jagolli',
    author_email='',
    description="Let's make some sweet tendies",
    install_requires=['seaborn','pandas','numpy','yfinance','matplotlib'],
    python_requires='>=3'
)
