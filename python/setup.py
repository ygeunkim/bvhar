from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import os
import glob

cpp_sources = glob.glob(os.path.join('bvhar', '*.cpp'))
include_path = os.path.abspath('../inst/include')

class PybindInclude(object):
    def __init__(self, user):
        self.user = user
    
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

class EigenInclude(object):
    def __str__(self):
        try:
            conda_prefix = os.environ['CONDA_PREFIX']
            eigen_path = os.path.join(conda_prefix, 'include', 'eigen3')
            if os.path.exists(eigen_path):
                return eigen_path
            else:
                raise RuntimeError('No eigen3 in conda environment')
        except KeyError:
            raise RuntimeError('Set CONDA_PREFIX environment variable')

class BuildExt(_build_ext):
    def build_extensions(self):
        if sys.platform == 'win32':
            compile_args = ['/openmp']
            link_args = []
        else:
            compile_args = ['-fopenmp']
            link_args = ['-fopenmp']
        for ext in self.extensions:
            ext.extra_compile_args += compile_args
            ext.extra_link_args += link_args
        _build_ext.build_extensions(self)

module = Extension(
    'bvhar',
    sources=cpp_sources,
    include_dirs=[
        include_path,
        str(PybindInclude(user=False)),
        str(EigenInclude())
    ],
    # define_macros=[('USE_PYBIND', None)],
    extra_compile_args=[
        '-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
        '-DBOOST_DISABLE_ASSERTS'
    ]
)

setup(
    name='bvhar',
    version='0.0.0.9000',
    packages=['bvhar'],
    description='BVHAR',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExt}
)
