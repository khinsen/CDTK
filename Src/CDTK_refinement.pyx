# Core routines for refinement.
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

include "numeric.pxi"

cdef extern from "math.h": 

    cdef double tanh(double x)

cdef extern from "gsl/gsl_sf_bessel.h":

    double gsl_sf_bessel_I0_scaled(double x)
    double gsl_sf_bessel_I1_scaled(double x)


def l(double t, array_type p, array_type rsc, array_type rsa):
    cdef int nc, na
    cdef int *rsc_p, *rsa_p
    cdef double *p_p
    cdef double lv, pv
    cdef int i

    assert PyArray_ISCONTIGUOUS(p)
    assert PyArray_ISCONTIGUOUS(rsc)
    assert PyArray_ISCONTIGUOUS(rsa)
    nc = rsc.dimensions[0]
    rsc_p = <int *>rsc.data
    na = rsa.dimensions[0]
    rsa_p = <int *>rsa.data
    p_p = <double *>p.data

    lv = 0.
    for i from 0 <= i < nc:
        pv = p_p[rsc_p[i]]
        lv = lv + pv*tanh(t*pv)
    for i from 0 <= i < na:
        pv = 2.*p_p[rsa_p[i]]
        lv = lv + pv*gsl_sf_bessel_I1_scaled(pv*t)/gsl_sf_bessel_I0_scaled(pv*t)
    return lv / (nc + 2*na)
