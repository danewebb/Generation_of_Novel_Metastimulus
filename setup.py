#!/usr/bin/env python

from distutils.core import setup, find_packages

CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Other Audience
Intended Audience :: Science/Research
License :: CC BY-NC 4.0 License
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Operating System :: Microsoft :: Windows
"""

setup(
    name='Metatstimuli-Project',
    version=' 0.1',
    author='Dane Webb',
    author_email='dane.webb@stmartin.edu',
    author_email='mr.dane.webb@gmail.com',
    url='',
    description='Generation of metastimuli',
    #long_description=long_description,
    packages=find_packages(),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=['numpy==1.18.5', 'matplotlib==3.3.3', 'tensorflow==2.3.0', 
                      'keras==2.4.3', 'keras-tuner==1.0.2', 'gensim==3.8.3',
                      'nltk==3.5', 'scikit-learn=0.23.2', 'sklearn'
                     ],
)
