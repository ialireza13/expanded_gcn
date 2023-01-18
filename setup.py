from setuptools import setup
from setuptools import find_packages

setup(name='expanded_gcn',
      version='1.0',
      description='Expanded Graph Convolutional Networks in PyTorch',
      author='Alireza Hashemi',
      author_email='alireza.hashemi13@outlook.com',
      download_url='https://github.com/ialireza13/expanded_gcn',
      license='MIT',
      install_requires=['numpy>=1.15.4',
                        'torch>=1.13.1',
                        'scipy>=1.1.0'
                        ],
      package_data={'expanded_gcn': ['README.md']},
      packages=find_packages())