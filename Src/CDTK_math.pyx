# Calculate math functions using GSL
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

cdef extern from "gsl/gsl_sf_bessel.h":

    ctypedef struct gsl_sf_result:
        double val
        double err

    int gsl_sf_bessel_I0_e(double x, gsl_sf_result *result)
    int gsl_sf_bessel_I1_e(double x, gsl_sf_result *result)
    int gsl_sf_bessel_I0_scaled_e(double x, gsl_sf_result *result)
    int gsl_sf_bessel_I1_scaled_e(double x, gsl_sf_result *result)

cdef extern from "gsl/gsl_sf_log.h":

    int gsl_sf_log_e(double x, gsl_sf_result *result)

cdef extern from "gsl/gsl_sf_exp.h":

    int gsl_sf_exp_e(double x, gsl_sf_result *result)

cdef extern from "gsl/gsl_errno.h":

    ctypedef void gsl_error_handler_t
    int GSL_SUCCESS
    int GSL_EUNDRFLW
    char *gsl_strerror(int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()

gsl_set_error_handler_off()


def I0(double x):
    cdef gsl_sf_result result
    cdef int status
    status = gsl_sf_bessel_I0_e(x, &result)
    if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
        return result.val
    raise ValueError(gsl_strerror(status))

def I1(double x):
    cdef gsl_sf_result result
    cdef int status
    status = gsl_sf_bessel_I1_e(x, &result)
    if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
        return result.val
    raise ValueError(gsl_strerror(status))

def I1divI0(double x):
    cdef gsl_sf_result result
    cdef int status
    cdef double i1
    status = gsl_sf_bessel_I1_scaled_e(x, &result)
    if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
        i1 = result.val
        status = gsl_sf_bessel_I0_scaled_e(x, &result)
        if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
            return i1/result.val
    raise ValueError(gsl_strerror(status))

def logI0(double x):
    cdef gsl_sf_result result
    cdef int status
    cdef double i0
    status = gsl_sf_bessel_I0_scaled_e(x, &result)
    if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
        i0 = result.val
        status = gsl_sf_log_e(i0, &result)
        if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
            return x+result.val
    raise ValueError(gsl_strerror(status))

def logcosh(double x):
    cdef gsl_sf_result result
    cdef int status
    cdef double i0
    status = gsl_sf_exp_e(-2.*x, &result)
    if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
        status = gsl_sf_log_e(0.5*(1.+result.val), &result)
        if status == GSL_SUCCESS or status == GSL_EUNDRFLW:
            return x+result.val
    raise ValueError(gsl_strerror(status))
