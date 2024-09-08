from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import os
from pybind11.setup_helpers import Pybind11Extension
import tempfile

with open("README.md", "r") as fh:
    long_description = fh.read()

include_path = os.path.abspath('../inst/include')

class EigenInclude(object):
    def __str__(self):
        # conda_prefix = os.environ.get('CONDA_PREFIX')
        conda_prefix = sys.prefix
        if os.path.exists(os.path.join(conda_prefix, 'conda-meta')):
            # if sys.platform.startswith('win'):
            #     eigen_path = os.path.join(conda_prefix, 'Library', 'include', 'eigen3')
            # else:
            #     eigen_path = os.path.join(conda_prefix, 'include', 'eigen3')
            # if os.path.exists(eigen_path):
            #     return eigen_path
            # else:
            #     print('No eigen3 in conda environment')
            cand_path = [
                os.path.join(conda_prefix, 'include', 'eigen3'),
                os.path.join(conda_prefix, 'Library', 'include', 'eigen3'),
                os.path.join(conda_prefix, 'Lib', 'include', 'eigen3')
            ]
            for eigen_path in cand_path:
                if os.path.exists(eigen_path):
                    return eigen_path
            print('No eigen3 in conda environment')
        eigen_dir = os.environ.get('EIGEN3_INCLUDE_DIR')
        if eigen_dir:
            eigen_path = os.path.join(eigen_dir, 'include', 'eigen3')
            if os.path.exists(eigen_path):
                return eigen_path
            else:
                raise RuntimeError('No eigen3 found in EIGEN3_INCLUDE_DIR')
        else:
            raise RuntimeError('Set CONDA_PREFIX or EIGEN3_INCLUDE_DIR environment variable')

class BuildExt(_build_ext):
    def has_flags(self, compiler, flags):
        with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
            f.write("int main() { return 0; }")
            try:
                compiler.compile([f.name], extra_postargs=[flags])
            except:
                return False
        return True

    def build_extensions(self):
        if sys.platform.startswith('win'):
            compile_args = ['/openmp'] if self.has_flags(self.compiler, '/openmp') else []
            link_args = []
        else:
            # compile_args = ['-fopenmp', '-Wall']
            # link_args = ['-fopenmp']
            compile_args = []
            link_args = []
            if self.has_flags(self.compiler, '-fopenmp'):
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
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
                    # Extension(
                    #     module_name,
                    #     sources=[os.path.join(root, cpp_file)],
                    #     include_dirs=[
                    #         include_path,
                    #         str(PythonInclude(user=False)),
                    #         str(PybindInclude(user=False)),
                    #         str(EigenInclude())
                    #     ],
                    #     extra_compile_args=[
                    #         '-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
                    #         '-DBOOST_DISABLE_ASSERTS'
                    #     ]
                    # )
                    Pybind11Extension(
                        module_name,
                        sources=[os.path.join(root, cpp_file)],
                        macros=[
                            ('EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS', None),
                            ('BOOST_DISABLE_ASSERTS', None)
                        ],
                        include_dirs=[
                            include_path,
                            str(EigenInclude())
                        ]
                    )
                )
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
