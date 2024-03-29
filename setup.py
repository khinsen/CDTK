
# Set "build_from" to  "c" to compile the pre-built C interface module.
# Choose "cython" for building from the original Cython source code.
#

build_from = 'cython'
fftw_prefix = None
ccp4_prefix = None
gpp4_prefix = None

assert build_from in ['c', 'cython']


from distutils.core import setup, Extension
if build_from == 'cython':
    from Cython.Distutils import build_ext
else:
    from distutils.command.build_ext import build_ext
import os, sys

class Dummy:
    pass
pkginfo = Dummy()
execfile('CDTK/__pkginfo__.py', pkginfo.__dict__)

compile_args = []
include_dirs = []

from Scientific import N
try:
    num_package = N.package
except AttributeError:
    num_package = "Numeric"
if num_package == "NumPy":
    compile_args.append("-DNUMPY=1")
    import numpy.distutils.misc_util
    include_dirs.extend(numpy.distutils.misc_util.get_numpy_include_dirs())

if fftw_prefix is None:
    try:
        fftw_prefix=os.environ['FFTW_PREFIX']
    except KeyError:
        for fftw_prefix in ['/usr/local', '/usr', '/sw', '/opt', '/opt/local']:
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

if ccp4_prefix is None:
    ccp4_prefix = os.environ.get('CCP4', None)
if gpp4_prefix is None:
    gpp4_prefix = os.environ.get('GPP4', None)
if ccp4_prefix is not None:
    print "Using CCP4 installation in ", ccp4_prefix
elif gpp4_prefix is not None:
    print "Using GPP4 installation in ", gpp4_prefix
else:
    sys.stderr.write('CCP4 library not found, '
                     'the MTZ module will not be built.\n')

map_module_source = {
    'c': 'Src/CDTK_sf_fft.c',
    'cython': 'Src/CDTK_sf_fft.pyx'
    }
sfcalc_module_source = {
    'c': 'Src/CDTK_sfcalc.c',
    'cython': 'Src/CDTK_sfcalc.pyx'
    }
math_module_source = {
    'c': 'Src/CDTK_math.c',
    'cython': 'Src/CDTK_math.pyx'
    }
symtensor_module_source = {
    'c': 'Src/CDTK_symtensor.c',
    'cython': 'Src/CDTK_symtensor.pyx'
    }
refinement_module_source = {
    'c': 'Src/CDTK_refinement.c',
    'cython': 'Src/CDTK_refinement.pyx'
    }
mtz_module_source = {
    'c': 'Src/CDTK_MTZ.c',
    'cython': 'Src/CDTK_MTZ.pyx'
    }

extension_modules = [Extension('CDTK_sf_fft',
                               [map_module_source[build_from]],
                               include_dirs = include_dirs+[fftw_include],
                               library_dirs = [fftw_lib],
                               libraries = ['fftw3', 'm'],
                               extra_compile_args = compile_args),
                     Extension('CDTK_sfcalc',
                               [sfcalc_module_source[build_from]],
                               include_dirs = include_dirs,
                               libraries = ['m'],
                               extra_compile_args = compile_args),
                     Extension('CDTK_refinement',
                               [refinement_module_source[build_from]],
                               include_dirs = include_dirs + ['/usr/local/include'],
                               library_dirs = ['/usr/local/lib'],
                               libraries = ['m', 'gsl', 'gslcblas'],
                               extra_compile_args = compile_args),
                     Extension('CDTK_math',
                               [math_module_source[build_from]],
                               include_dirs = include_dirs + ['/usr/local/include'],
                               library_dirs = ['/usr/local/lib'],
                               libraries = ['m', 'gsl', 'gslcblas'],
                               extra_compile_args = compile_args),
                     Extension('CDTK_symtensor',
                               [symtensor_module_source[build_from]],
                               include_dirs = include_dirs + ['/usr/local/include'],
                               #library_dirs = ['/usr/local/lib'],
                               #libraries = [],
                               extra_compile_args = compile_args)]
if ccp4_prefix is not None:
    extension_modules.append(
        Extension('CDTK_MTZ',
                  [mtz_module_source[build_from]],
                  include_dirs = include_dirs+[os.path.join(ccp4_prefix,
                                                            'include', 'ccp4')],
                  library_dirs = [os.path.join(ccp4_prefix, 'lib')],
                  libraries = ['ccp4c'],
                  extra_compile_args = compile_args)
        )
elif gpp4_prefix is not None:
    extension_modules.append(
        Extension('CDTK_MTZ',
                  [mtz_module_source[build_from]],
                  include_dirs = include_dirs+[os.path.join(gpp4_prefix,
                                                            'include',
                                                            'gpp4', 'ccp4')],
                  library_dirs = [os.path.join(gpp4_prefix, 'lib')],
                  libraries = ['gpp4'],
                  extra_compile_args = compile_args)
        )

setup (name = "CDTK",
       version = pkginfo.__version__,
       description = "Crystallographic Data Toolkit",
       long_description = """
The Crystallographic Data Toolkit is a library for working with
crystallographic data on macromolecules, in particular structure
factors and experimental reflection intensities. It is designed
as a companion to the Molecular Modelling Toolkit (MMTK), but
can also be used independently.
""",
       author = "Konrad Hinsen",
       author_email = "hinsen@cnrs-orleans.fr",
       url = "http://dirac.cnrs-orleans.fr/CDTK/",
       license = "CeCILL-C",

       packages = ['CDTK'],
 
       ext_package = 'CDTK.'+sys.platform,
       ext_modules = extension_modules,

       scripts = ['Scripts/convert_mmcif_reflections',
                  'Scripts/convert_pdb_reflections',
                  'Scripts/convert_mtz_reflections'],
 
       cmdclass = {'build_ext': build_ext}
       )
