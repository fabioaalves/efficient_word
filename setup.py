from setuptools import setup
from glob import glob

setup(
    name='efficient_word',
    version='2.0',
    description='Hotword detection',
    url='https://github.com/fabioaalves/efficient-word',
    packages=['efficient_word'],
    install_requires=open("./requirements.txt", 'r').read().split("\n"),
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
)
