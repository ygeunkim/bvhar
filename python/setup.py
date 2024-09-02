from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import os
# import glob
# import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

# cpp_sources = glob.glob(os.path.join('bvhar', '**', '*.cpp'), recursive=True)
include_path = os.path.abspath('../inst/include')
# r_include = subprocess.check_output(["R", "RHOME"]).decode("utf-8").strip() + "/include"

class PythonInclude(object):
    def __init__(self, user):
        self.user = user
    
    def __str__(self):
        import distutils.sysconfig as ds
        return ds.get_python_inc(self.user)

class PybindInclude(object):
    def __init__(self, user):
        self.user = user
    
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

class EigenInclude(object):
    def __str__(self):
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            eigen_path = os.path.join(conda_prefix, 'include', 'eigen3')
            if os.path.exists(eigen_path):
                return eigen_path
            else:
                # raise RuntimeError('No eigen3 in conda environment')
                print('No eigen3 in conda environment')
        # else:
        eigen_dir = os.environ.get('EIGEN3_INCLUDE_DIR')
        if eigen_dir:
            eigen_path = os.path.join(eigen_dir, 'include', 'eigen3')
            if os.path.exists(eigen_path):
                return eigen_path
            else:
                raise RuntimeError('No eigen3 found in EIGEN3_INCLUDE_DIR')
        else:
            # raise RuntimeError('Set EIGEN3_INCLUDE_DIR environment variable for eigen directory')
            raise RuntimeError('Set CONDA_PREFIX or EIGEN3_INCLUDE_DIR environment variable')
        # try:
        #     conda_prefix = os.environ['CONDA_PREFIX']
        #     eigen_path = os.path.join(conda_prefix, 'include', 'eigen3')
        #     if os.path.exists(eigen_path):
        #         return eigen_path
        #     else:
        #         raise RuntimeError('No eigen3 in conda environment')
        # except KeyError:
        #     raise RuntimeError('Set CONDA_PREFIX environment variable')

class BuildExt(_build_ext):
    def build_extensions(self):
        if sys.platform == 'win32':
            compile_args = ['/openmp']
            link_args = []
        else:
            compile_args = ['-fopenmp', '-Wall']
            link_args = ['-fopenmp']
        for ext in self.extensions:
            ext.extra_compile_args += compile_args
            ext.extra_link_args += link_args
        _build_ext.build_extensions(self)

def find_module(base_dir):
    extensions = []
    for root, dirs, files in os.walk(base_dir):
        for cpp_file in files:
            if cpp_file.endswith('.cpp'):
                rel_path = os.path.relpath(root, base_dir)
                module_name = os.path.splitext(cpp_file)[0]
                # module_name = f'bvhar.{rel_path.replace(os.path.sep, ".")}' if rel_path != "." else base_dir
                module_name = f"{base_dir}.{rel_path.replace(os.path.sep, '.')}.{module_name}" if rel_path != "." else f"{base_dir}.{module_name}"
                extensions.append(
                    Extension(
                        module_name,
                        sources=[os.path.join(root, cpp_file)],
                        include_dirs=[
                            include_path,
                            str(PythonInclude(user=False)),
                            str(PybindInclude(user=False)),
                            str(EigenInclude())
                        ],
                        extra_compile_args=[
                            '-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
                            '-DBOOST_DISABLE_ASSERTS'
                        ]
                    )
                )

        # cpp_files = [os.path.join(root, f) for f in files if f.endswith('.cpp')]
        # if cpp_files:
        #     rel_path = os.path.relpath(root, base_dir)
        #     module_name = f'bvhar.{rel_path.replace(os.path.sep, ".")}' if rel_path != "." else "bvhar"
        #     extensions.append(
        #         Extension(
        #             module_name,
        #             # sources=cpp_sources,
        #             sources=cpp_files,
        #             include_dirs=[
        #                 include_path,
        #                 str(PybindInclude(user=False)),
        #                 str(EigenInclude())
        #             ],
        #             extra_compile_args=[
        #                 '-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
        #                 '-DBOOST_DISABLE_ASSERTS',
        #                 '-std=c++11'
        #             ]
        #         )
        #     )
    return extensions

setup(
    name='bvhar',
    version='0.0.0.9000',
    packages=find_packages(include=['bvhar', 'bvhar.*']),
    # packages=find_packages(where='src'),
    # package_dir={'': 'src'},
    description='Bayesian multivariate time series modeling',
    long_description=long_description,
    author='Young Geun Kim',
    author_email='ygeunkimstat@gmail.com',
    keywords=['bayesian', 'time series'],
    install_requires=[
        'pybind11',
        'numpy',
        'pandas'
    ],
    ext_modules=find_module('bvhar'),
    cmdclass={'build_ext': BuildExt}
)
