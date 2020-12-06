from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace

# define an extension that will be cythonized and compiled
ext = Extension(
    name="Compute_Similarity_Cython", 
    sources=["Compute_Similarity_Cython.pyx"], 
    include_dirs=[numpy.get_include()])
    
setup(ext_modules=cythonize(ext))