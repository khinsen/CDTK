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

    cdef double exp(double x)
    cdef double log(double x)
    cdef double tanh(double x)

cdef extern from "gsl/gsl_sf_bessel.h":

    ctypedef struct gsl_sf_result:
        double val
        double err

    int gsl_sf_bessel_I0_scaled_e(double x, gsl_sf_result *result)
    int gsl_sf_bessel_I1_scaled_e(double x, gsl_sf_result *result)
    int gsl_sf_log_e(double x, gsl_sf_result *result)

cdef extern from "gsl/gsl_errno.h":

    ctypedef void gsl_error_handler_t
    int GSL_SUCCESS
    int GSL_EUNDRFLW
    char *gsl_strerror(int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()

gsl_set_error_handler_off()

## def _llkwd(p, q, dp, dq, dllk, rc, ra):
##     llk = 0.
##     for ri in rc:
##         llk -= 0.5*p[ri]+logcosh(q[ri])
##         dllk[ri] = -dp[ri]-N.tanh(q[ri])*dq[ri]
##     for ri in ra:
##         llk -= p[ri]+logI0(2.*q[ri])
##         dllk[ri] = -2.*(dp[ri]+I1divI0(2*q[ri])*dq[ri])
##     nwreflections = len(rc) + len(ra)
##     return llk/nwreflections, dllk/nwreflections

def _llkwd(array_type p, array_type q, array_type dp, array_type dq,
           array_type dllk, array_type rc, array_type ra):
    cdef gsl_sf_result result
    cdef int status
    cdef int nc, na, nwr, nr
    cdef int *rc_p, *ra_p
    cdef double *p_p, *p_q, *p_dp, *p_dq, *p_dllk
    cdef double llk, pv, qv, dpv, dqv
    cdef double i0, i1, logI0
    cdef int i

    assert PyArray_ISCONTIGUOUS(p)
    assert PyArray_ISCONTIGUOUS(q)
    assert PyArray_ISCONTIGUOUS(dp)
    assert PyArray_ISCONTIGUOUS(dq)
    assert PyArray_ISCONTIGUOUS(dllk)
    assert PyArray_ISCONTIGUOUS(rc)
    assert PyArray_ISCONTIGUOUS(ra)
    assert p.descr.elsize == sizeof(double)
    assert q.descr.elsize == sizeof(double)
    assert dp.descr.elsize == sizeof(double)
    assert dq.descr.elsize == sizeof(double)
    assert dllk.descr.elsize == sizeof(double)
    assert rc.descr.elsize == sizeof(int)
    assert ra.descr.elsize == sizeof(int)
    nc = rc.dimensions[0]
    rc_p = <int *>rc.data
    na = ra.dimensions[0]
    ra_p = <int *>ra.data
    nwr = nc + na
    nr = p.dimensions[0]
    assert nr == q.dimensions[0]
    assert nr == dp.dimensions[0]
    assert nr == dq.dimensions[0]
    p_p = <double *>p.data
    p_q = <double *>q.data
    p_dp = <double *>dp.data
    p_dq = <double *>dq.data
    p_dllk = <double *>dllk.data

    llk = 0.
    for 0 <= i < nc:
        pv = p_p[rc_p[i]]
        qv = p_q[rc_p[i]]
        dpv = p_dp[rc_p[i]]
        dqv = p_dq[rc_p[i]]
        llk = llk - 0.5*pv - qv - log(0.5*(1.+exp(-2.*qv)))
        p_dllk[rc_p[i]] = -(dpv + dqv*tanh(qv))/nwr
    for 0 <= i < na:
        pv = p_p[ra_p[i]]
        qv = 2.*p_q[ra_p[i]]
        dpv = p_dp[ra_p[i]]
        dqv = p_dq[ra_p[i]]
        status = gsl_sf_bessel_I1_scaled_e(qv, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i1 = result.val
        status = gsl_sf_bessel_I0_scaled_e(qv, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i0 = result.val
        status = gsl_sf_log_e(i0, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        logI0 = qv+result.val
        llk = llk - pv - logI0
        p_dllk[ra_p[i]] = -2.*(dpv + dqv*i1/i0)/nwr
    llk = llk/nwr
    return llk, dllk

## def _llkdf(p, q, rc, ra):
##     d = 0.
##     for ri in rc:
##         d -= p[ri] + q[ri]*N.tanh(q[ri])
##     for ri in ra:
##         d -= 2.*p[ri] + 2.*q[ri]*I1divI0(2.*q[ri])
##     return d/(len(rc)+len(ra))

def _llkdf(array_type p, array_type q, array_type rc, array_type ra):
    cdef gsl_sf_result result
    cdef int status
    cdef int nc, na
    cdef int *rc_p, *ra_p
    cdef double *p_p, *p_q
    cdef double d, pv, qv
    cdef double i0, i1
    cdef int i

    assert PyArray_ISCONTIGUOUS(p)
    assert PyArray_ISCONTIGUOUS(q)
    assert PyArray_ISCONTIGUOUS(rc)
    assert PyArray_ISCONTIGUOUS(ra)
    assert p.descr.elsize == sizeof(double)
    assert q.descr.elsize == sizeof(double)
    assert rc.descr.elsize == sizeof(int)
    assert ra.descr.elsize == sizeof(int)
    nc = rc.dimensions[0]
    rc_p = <int *>rc.data
    na = ra.dimensions[0]
    ra_p = <int *>ra.data
    p_p = <double *>p.data
    p_q = <double *>q.data

    d = 0.
    for 0 <= i < nc:
        pv = p_p[rc_p[i]]
        qv = p_q[rc_p[i]]
        d = d - pv - qv*tanh(qv)
    for 0 <= i < na:
        pv = p_p[ra_p[i]]
        qv = 2.*p_q[ra_p[i]]
        status = gsl_sf_bessel_I1_scaled_e(qv, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i1 = result.val
        status = gsl_sf_bessel_I0_scaled_e(qv, &result)
        if status != GSL_SUCCESS and status != GSL_EUNDRFLW:
            raise ValueError(gsl_strerror(status))
        i0 = result.val
        d = d - 2.*pv - qv*i1/i0
    return d / (nc + na)

## def _l(t, p, rsc, rsa):
##     lv = 0.
##     for ri in rsc:
##         lv += p[ri]*N.tanh(t*p[ri])
##     for ri in rsa:
##         p2 = 2.*p[ri]
##         lv += p2*I1divI0(t*p2)
##     return lv / (len(rsc) + 2.*len(rsa))

def _l(double t, array_type p, array_type rsc, array_type rsa):
    cdef gsl_sf_result result
    cdef int status
    cdef int nc, na
    cdef int *rsc_p, *rsa_p
    cdef double *p_p
    cdef double lv, pv
    cdef double i0, i1
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
    for 0 <= i < nc:
        pv = p_p[rsc_p[i]]
        lv = lv + pv*tanh(t*pv)
    for 0 <= i < na:
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
