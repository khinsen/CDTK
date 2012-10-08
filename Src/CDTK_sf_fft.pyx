# Conversion between density maps and structure factors by FFT
# Uses FFTW 3.x
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

cimport cython

include "numeric.pxi"

from Scientific import N
from CDTK.ReflectionData import StructureFactor

cdef extern from "fftw3.h":

    ctypedef void *fftw_plan
    ctypedef double fftw_complex[2]
    unsigned FFTW_ESTIMATE
    unsigned FFTW_MEASURE
    unsigned FFTW_PATIENT
    unsigned FFTW_EXHAUSTIVE
    fftw_plan fftw_plan_dft_r2c_3d(int nx, int ny, int nz,
                                   double *in_array, fftw_complex *out_array,
                                   unsigned flags) nogil
    fftw_plan fftw_plan_dft_c2r_3d(int nx, int ny, int nz,
                                   fftw_complex *in_array, double *out_array,
                                   unsigned flags) nogil
    void fftw_execute(fftw_plan plan) nogil
    void fftw_destroy_plan(fftw_plan plan) nogil

#
# Convert density map to structure factors
#
@cython.cdivision(True)
def map_to_sf(array_type map_array, sf, double factor):
    """
    @param map_array: an 3D array containing an electronic density map
    @type map_object: N.array_type
    @param sf: a StructureFactor object
    @type sf: L{CDTK.Reflections.StructureFactor}
    @param factor: a scale factor to be applied to the map
    @type factor: C{float}
    @raises IndexError: if a reflection is outside of the resolution of the map
    """
    cdef double *in_data
    cdef int nx
    cdef int ny
    cdef int nz
    cdef array_type sf_array
    cdef fftw_complex *out_data
    cdef fftw_plan plan
    cdef int h
    cdef int k
    cdef int l
    cdef double imag_sign
    cdef double real_part
    cdef double imag_part
    if not isinstance(sf, StructureFactor):
        raise TypeError("%s is not a StructureFactor instance" % str(sf))
    assert PyArray_ISCONTIGUOUS(map_array)
    assert map_array.descr.elsize == sizeof(double)
    in_data = <double *>map_array.data
    nx, ny, nz = map_array.shape
    sf_array = N.zeros((nx, ny, nz/2+1), N.Complex)
    out_data = <fftw_complex *>sf_array.data
    with nogil:
        plan = fftw_plan_dft_r2c_3d(nx, ny, nz, in_data, out_data,
                                    FFTW_ESTIMATE)
        fftw_execute(plan)
        fftw_destroy_plan(plan)

    factor = factor/(<double>nx*<double>ny*<double>nz)/(N.sqrt(2.*N.pi)**3)
    for r in sf.reflection_set:
        h = r.h
        k = r.k
        l = r.l
        imag_sign = -1. # to compensate for FFTW's sign conventions
        if h < 0:
            h = nx+h
        if k < 0:
            k = ny+k
        if l < 0:
            if h > 0: h = nx-h
            if k > 0: k = ny-k
            l = -l
            imag_sign = -imag_sign
        if h < nx and k < ny and l < nz/2+1:
            if (h+k+l) % 2 == 1:
                real_part = -out_data[(h*ny+k)*(nz/2+1)+l][0]
                imag_part = -out_data[(h*ny+k)*(nz/2+1)+l][1]
            else:
                real_part = out_data[(h*ny+k)*(nz/2+1)+l][0]
                imag_part = out_data[(h*ny+k)*(nz/2+1)+l][1]
            sf.array[r.index] = sf.array[r.index] + \
                         complex(factor*real_part,
                                 factor*imag_sign*imag_part)

#
# Convert structure factors or intensities to density/Patterson map
#
@cython.cdivision(True)
def reflections_to_map(data, int n1, int n2, int n3, double factor,
                       int check_resolution=1):
    """
    @param data: a StructureFactor or IntensityData object
    @type sf: L{CDTK.Reflections.StructureFactor}
              or L{CDTK.Reflections.IntensityData}
    @param n1: first dimension of the map array
    @type n1: C{int}
    @param n2: second dimension of the map array
    @type n2: C{int}
    @param n3: third dimension of the map array
    @type n3: C{int}
    @param factor: a scale factor to be applied to the map
    @type factor: C{float}
    @param check_resolution: if True, raise an exception for reflections
                             outside of the resolution range of the map.
                             If False, such reflections are ignored.
    @type check_resolution: C{bool}
    """
    cdef array_type map_array
    cdef fftw_complex *complex_data
    cdef int n3r
    cdef int h, k, l
    cdef int out_of_range
    cdef double imag_sign
    cdef fftw_plan plan
    n3r = 2*(n3/2+1)
    map_array = N.zeros((n1, n2, n3r), N.Float)
    complex_data = <fftw_complex *>map_array.data
    factor = factor*N.sqrt(2.*N.pi)**3
    for r_asu in data.reflection_set:
        for r in r_asu.symmetryEquivalents():
            out_of_range = False
            h = r.h
            k = r.k
            l = r.l
            imag_sign = -1. # to compensate for FFTW's sign conventions
            if h < -n1 or h >= n1:
                out_of_range = True
            else:
                if h < 0:
                    h += n1
            if k < -n2 or k >= n2:
                out_of_range = True
            else:
                if k < 0:
                    k += n2
            if l < 0:
                if h > 0: h = n1-h
                if k > 0: k = n2-k
                l = -l
                imag_sign = -imag_sign
            if l >= n3/2+1:
                out_of_range = True
            if out_of_range:
                if check_resolution:
                    raise ValueError("Insufficient map resolution "
                                     "for Miller indices %d, %d, %d"
                                     % (r.h, r.k, r.l))
                else:
                    continue
            value = data[r]
            if value is not None:
                if (h+k+l) % 2 == 1:
                    value = -value
                try:
                    vreal = value.real
                    vimag = value.imag
                except AttributeError:
                    vreal = value
                    vimag = 0.
                complex_data[(h*n2+k)*(n3/2+1)+l][0] = \
                                  factor*vreal
                complex_data[(h*n2+k)*(n3/2+1)+l][1] = \
                                  factor*vimag*imag_sign

    with nogil:
        plan = fftw_plan_dft_c2r_3d(n1, n2, n3,
                                    complex_data, <double*>complex_data,
                                    FFTW_ESTIMATE)
        fftw_execute(plan)
        fftw_destroy_plan(plan)

    return map_array[:,:,:n3]

