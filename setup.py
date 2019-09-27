from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('CFD_demo.pyx', language_level = '3', annotate=True))
