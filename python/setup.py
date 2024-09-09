from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
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
                lib_path = os.path.join(conda_prefix, 'Library', 'include', self.lib)
            else:
                lib_path = os.path.join(conda_prefix, 'include', self.lib)
            if os.path.exists(lib_path):
                print(f"Use {lib_path} for {self.lib} header")
                return lib_path
            else:
                print(f"No {self.lib} in conda environment")
        lib_dir = os.environ.get(f"{self.lib.upper()}_INCLUDE_DIR")
        if lib_dir:
            lib_path = os.path.join(lib_dir, 'include', self.lib)
            if os.path.exists(lib_path):
                return lib_path
            else:
                raise RuntimeError(f"No {self.lib} found in {self.lib.upper}_INCLUDE_DIR")
        else:
            raise RuntimeError(f"Set CONDA_PREFIX or {self.lib.upper}_INCLUDE_DIR environment variable")

class BuildExt(_build_ext):
    def has_flags(self, compiler, flag):
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write("int main() { return 0; }")
            temp_file = f.name
            try:
                compiler.compile([temp_file], extra_postargs=[flag])
            except Exception as e:
                print(f"Flag {flag} not supported by the compiler: {e}")
                return False
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            print(f"Use {flag} flag")
        return True

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
                            str(HeaderInclude('')) if sys.platform.startswith('win') else str(HeaderInclude('boost'))
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