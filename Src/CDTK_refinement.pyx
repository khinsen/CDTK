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

    ctypedef struct gsl_sf_result:
        double val
        double err

    int gsl_sf_bessel_I0_scaled_e(double x, gsl_sf_result *result)
    int gsl_sf_bessel_I1_scaled_e(double x, gsl_sf_result *result)

cdef extern from "gsl/gsl_errno.h":

    ctypedef void gsl_error_handler_t
    int GSL_SUCCESS
    int GSL_EUNDRFLW
    char *gsl_strerror(int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()

gsl_set_error_handler_off()

def l(double t, array_type p, array_type rsc, array_type rsa):
    cdef gsl_sf_result result
    cdef int status
    cdef int nc, na
    cdef int *rsc_p, *rsa_p
    cdef double *p_p
    cdef double lv, pv
    cdef int i

    assert PyArray_ISCONTIGUOUS(p)
    assert PyArray_ISCONTIGUOUS(rsc)
    assert PyArray_ISCONTIGUOUS(rsa)
    assert p.descr.elsize == sizeof(double)
    assert rsc.descr.elsize == sizeof(int)
    assert rsa.descr.elsize == sizeof(int)
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
        status = gsl_sf_bessel_I1_scaled_e(pv*t, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i1 = result.val
        status = gsl_sf_bessel_I0_scaled_e(pv*t, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i0 = result.val
        lv = lv + pv*i1/i0
    return lv / (nc + 2*na)
