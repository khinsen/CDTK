# Set "build_from" to  "c" to compile the pre-built C interface module.
# Choose "pyrex" for building from the original Pyrex source code.
#

build_from = 'pyrex'
fftw_prefix = None

assert build_from in ['c', 'pyrex']


from distutils.core import setup, Extension
if build_from == 'pyrex':
    from Pyrex.Distutils import build_ext
else:
    from distutils.command.build_ext import build_ext
import os, sys

compile_args = []
include_dirs = []

from Scientific import N
try:
    num_package = N.package
except AttributeError:
    num_package = "Numeric"
if num_package == "NumPy":
    compile_args.append("-DNUMPY=1")
    include_dirs.append(os.path.join(sys.prefix,
                            "lib/python%s.%s/site-packages/numpy/core/include"
                                 % sys.version_info [:2]))

if fftw_prefix is None:
    try:
        fftw_prefix=os.environ['FFTW_PREFIX']
    except KeyError:
        for fftw_prefix in ['/usr/local', '/usr', '/sw']:
            fftw_include = os.path.join(fftw_prefix, 'include')
            fftw_lib = os.path.join(fftw_prefix, 'lib')
            if os.path.exists(os.path.join(fftw_include, 'fftw3.h')):
                break
        else:
            fftw_prefix = None

if fftw_prefix is None:
    print "FFTW3 not found!"
    print "If FFTW3 is installed somewhere on this computer,"
    print "please set FFTW_PREFIX to the path where"
    print "include/fftw3.h and lib/libfftw3.a are located"
    print "and re-run the build procedure."
else:
    print "Using FFTW3 installation in ", fftw_prefix
    fftw_include = os.path.join(fftw_prefix, 'include')
    fftw_lib = os.path.join(fftw_prefix, 'lib')

map_module_source = {
    'c': 'Src/CDTK_sf_fft.c',
    'pyrex': 'Src/CDTK_sf_fft.pyx'
    }

setup (name = "CDTK",
       version = "0.1",
       description = "Crystallographic Data Toolkit",

       packages = [],
 
       ext_modules = [Extension('CDTK_sf_fft',
                                [map_module_source[build_from]],
                                include_dirs = include_dirs+[fftw_include],
                                library_dirs = [fftw_lib],
                                libraries = ['fftw3', 'm'],
                                extra_compile_args = compile_args)],
       cmdclass = {'build_ext': build_ext}
       )
