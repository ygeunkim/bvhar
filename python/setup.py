from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import shutil
from distutils.command.sdist import sdist
import sys
import os
from pybind11.setup_helpers import Pybind11Extension
import tempfile

with open("README.md", "r") as fh:
    long_description = fh.read()

include_path = os.path.abspath('../inst/include')

class HeaderInclude(object):
    def __init__(self, lib: str):
        self.lib = lib

    def __str__(self):
        conda_prefix = sys.prefix
        print(f"Current environment path: {conda_prefix}")
        if os.path.exists(os.path.join(conda_prefix, 'conda-meta')):
            if sys.platform.startswith('win'):
                self.lib = '' if self.lib == 'boost' else self.lib # should use include/ in windows-conda
                lib_path = os.path.join(conda_prefix, 'Library', 'include', self.lib)
            else:
                lib_path = os.path.join(conda_prefix, 'include', self.lib)
            if os.path.exists(lib_path):
                print(f"Use {lib_path} for {self.lib} header")
                return lib_path
            else:
                print(f"No {self.lib} in conda environment")
        _lib = self.lib.rstrip('0123456789$').upper()
        lib_dir = os.environ.get(f"{_lib}_INCLUDE_DIR")
        if lib_dir:
            # lib_path = os.path.join(lib_dir, 'include', self.lib)
            lib_path = lib_dir
            if os.path.exists(lib_path):
                return lib_path
            else:
                raise RuntimeError(f"No {self.lib} found in {_lib}_INCLUDE_DIR")
        else:
            raise RuntimeError(f"Set CONDA_PREFIX or {_lib}_INCLUDE_DIR environment variable")

class BuildExt(_build_ext):
    def has_flags(self, compiler, flag):
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write("int main() { return 0; }")
            temp_file = f.name
            f.close()
            try:
                compiler.compile([temp_file], extra_postargs=[flag])
                print(f"Use {flag} flag")
                return True
            except Exception as e:
                print(f"Flag {flag} not supported by the compiler: {e}")
                return False
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def build_extensions(self):
        compile_args = []
        link_args = []
        if sys.platform.startswith('win'):
            if self.has_flags(self.compiler, '/openmp'):
                compile_args.append('/openmp')
        else:
            if self.has_flags(self.compiler, '-fopenmp'):
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
        for ext in self.extensions:
            ext.extra_compile_args += compile_args
            ext.extra_link_args += link_args
        _build_ext.build_extensions(self)

class SdistInclude(sdist):
    def run(self):
        shutil.copytree('../inst/include', './include', dirs_exist_ok=True)
        super().run()

def find_module(base_dir):
    extensions = []
    is_src = os.path.basename(base_dir) == 'src'
    for root, dirs, files in os.walk(base_dir):
        for cpp_file in files:
            if cpp_file.endswith('.cpp'):
                rel_path = os.path.relpath(root, base_dir)
                module_name = os.path.splitext(cpp_file)[0]
                if is_src:
                    rel_path = rel_path.replace('bvhar', '').strip(os.path.sep)
                    # module_name = f'bvhar.{rel_path.replace(os.path.sep, ".")}' if rel_path != "." else base_dir
                    module_name = f"bvhar.{rel_path.replace(os.path.sep, '.')}.{module_name}" if rel_path != "" else f"{base_dir}.{module_name}"
                else:
                    module_name = f"{base_dir}.{rel_path.replace(os.path.sep, '.')}.{module_name}" if rel_path != "." else f"{base_dir}.{module_name}"
                extensions.append(
                    Pybind11Extension(
                        module_name,
                        sources=[os.path.join(root, cpp_file)],
                        define_macros=[
                            ('EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS', None),
                            ('BOOST_DISABLE_ASSERTS', None)
                        ],
                        include_dirs=[
                            include_path,
                            # str(EigenInclude())
                            str(HeaderInclude('eigen3')),
                            # str(HeaderInclude('')) if sys.platform.startswith('win') else str(HeaderInclude('boost'))
                            str(HeaderInclude('boost'))
                        ]
                    )
                )
    return extensions

setup(
    name='bvhar',
    version='0.0.0.9000',
    # packages=find_packages(include=['bvhar', 'bvhar.*']),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='Bayesian multivariate time series modeling',
    url='https://github.com/ygeunkim/bvhar/tree/feature/python',
    long_description=long_description,
    author='Young Geun Kim',
    author_email='ygeunkimstat@gmail.com',
    keywords=[
        'bayesian',
        'time series'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=[
        'pybind11',
        'numpy',
        'pandas'
    ],
    # ext_modules=find_module('bvhar'),
    ext_modules=find_module('src'),
    cmdclass={
        'build_ext': BuildExt,
        'sdist': SdistInclude
    }
)
