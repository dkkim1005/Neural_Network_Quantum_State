#!/usr/bin/env python3
import os
import re
import sys
import sysconfig
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Refer to "https://www.benjack.io/2017/06/12/python-cpp-tests.html"
# I changed minor parts of the code written in the above site.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
              "CMake must be installed to build the following extensions: " +
              ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
                 os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DBUILD_PYTHON3_MODULE=TRUE',
                      '-DUSE_CUDA=TRUE',
                      '-Wno-dev']
        build_args = ['--', '-j2']
        env = os.environ.copy()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

module_name = 'cpp_extension'
setup (
    name = module_name,
    version = '0.1',
    author = 'Dongkyu Kim',
    author_email = 'dkkim1005@gmail.com',
    description = 'Python binding for cpp implementation of Neural network quantum states',
    long_description = '',
    platforms = ['linux'],
    # add extension module
    ext_modules = [CMakeExtension('cpp_extension')],
    # add custom build_ext command
    cmdclass = dict(build_ext=CMakeBuild),
    zip_safe = False
)
