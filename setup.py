from pathlib import Path
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
# from setuptools import setup, Extension # for pypi build
from distutils.core import setup, Extension  # python setup

c_ext = Extension('cext_acv', sources=['acpi/cext_acv/_cext.cc'])

cy_ext = Extension('cyext_acv', ['acpi/cyext_acv/cyext_acv.pyx'], extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])

this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text()

setup(name='ACPI',
      author='Salim I. Amoukou',
      author_email='salim.ibrahim-amoukou@universite-paris-saclay.fr',
      version='0.0.0',
      description='Adaptive Conformal Prediction (ACP) is a Python package that aims to provide Adaptive Predictive '
                  'Interval (PI) that better represent the uncertainty of the model by reweighting the NonConformal '
                  'Score with the learned weights of a Random Forest.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='ano',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([cy_ext, c_ext]),
      setup_requires=["setuptools", "wheel", "numpy<1.22", "Cython", "pybind11"],
      install_requires=['numpy<1.22', 'scipy', 'scikit-learn', 'pandas', 'tqdm', 'skranger', 'pybind11', 'PyGenStability@git+https://github.com/barahona-research-group/PyGenStability.git'],
      extras_require={'test': ['xgboost', 'lightgbm', 'catboost', 'pyspark', 'pytest']},
      packages=['acpi'],
      license='MIT',
      zip_safe=False
      )
