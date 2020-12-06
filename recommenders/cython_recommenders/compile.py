from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
FunkSVD_ext  = Extension(name="FunkSVD_fast", sources=["FunkSVD_fast.pyx"])
SLIM_MSE_ext = Extension(name="SLIM_MSE_fast", sources=["SLIM_MSE_fast.pyx"])

setup(ext_modules=cythonize(FunkSVD_ext))
setup(ext_modules=cythonize(SLIM_MSE_ext))

#python compile.py build_ext --inplace
