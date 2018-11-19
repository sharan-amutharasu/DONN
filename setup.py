from setuptools import setup

long_description = '''
DONN (Deep Optimized Neural Networks) is a library that can be used by people new to deep learning, to build optimized neural networks.

Read the documentation at: https://github.com/sharan-amutharasu/donn

DONN is compatible with Python 3
and is distributed under the MIT license.
'''

setup(name='donn',
      version='1.1.8',
      description='Deep Optimized Neural Networks',
      long_description=long_description,
      author='Sharan Amutharasu',
      author_email='sharan.amutharasu@gmail.com',
      url='https://github.com/sharan-amutharasu/donn',
      download_url='https://github.com/sharan-amutharasu/donn/tarball/1.1.1',
      license='MIT',
      install_requires=['keras>=2.2.0',
                        'numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py',
                        'scikit-learn>=0.19.0'
                        ],
      extras_require={
          'tests': ['pytest'],
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=['donn'])