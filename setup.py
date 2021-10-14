import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension


setup(
    name='xrecord',
    version='1.0.0',
    author='gang.zhang',
    author_email='zhanggang11021136@gmail.com',
    description='A large key-value dataset for fast data access!',
    packages=find_packages()
)