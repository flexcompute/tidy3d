#!/usr/bin/env python

from distutils.core import setup

# with open('./requirements.txt') as f:
#     required = f.read().splitlines()

# with open('tests/requirements.txt') as f:
#     required += f.read().splitlines()

setup(name='Tidy3D',
      version='0.1.1',
      description='Front end for Tidy3D.',
      author='Flexcompute inc.',
      author_email='tyler@flexcompute.com',
      url='flexcompute.com',
      # install_requires=required,
      packages=['tidy3d'],
      )
