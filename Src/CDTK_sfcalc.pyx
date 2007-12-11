# Core routines for structure factor calculation
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

include "numeric.pxi"

cdef extern from "math.h": 

    cdef double cos(double x)
    cdef double sin(double x)
    cdef double exp(double x)


from Scientific import N

cdef double twopi
cdef double twopisq
twopi = 2.*N.pi
twopisq = -2.*N.pi**2

def sfTerm(array_type result, array_type s, array_type f_atom,
           array_type r, array_type u, double u_scalar, int use_u):
    cdef int ns
    cdef int i
    cdef double *resp
    cdef double *sp
    cdef double *rp
    cdef double *up
    cdef double *fap
    cdef double dot_sr
    cdef double f_r
    cdef double f_i
    cdef double sus
    cdef double dwf

    if PyArray_ISCONTIGUOUS(result) and PyArray_ISCONTIGUOUS(s) \
           and PyArray_ISCONTIGUOUS(f_atom) \
           and PyArray_ISCONTIGUOUS(r) and PyArray_ISCONTIGUOUS(u):

        ns = result.dimensions[0]
        assert result.nd == 1
        assert s.nd == 2 and s.dimensions[0] == ns and s.dimensions[1] == 3
        assert f_atom.nd == 1 and f_atom.dimensions[0] == ns
        assert r.nd == 1 and r.dimensions[0] == 3
        resp = <double *>result.data
        sp = <double *>s.data
        rp = <double *>r.data
        up = <double *>u.data
        fap = <double *>f_atom.data

        for i from 0 <= i < ns:
            dot_sr = twopi*(rp[0]*sp[3*i] + rp[1]*sp[3*i+1] + rp[2]*sp[3*i+2])
            f_r = fap[i]*cos(dot_sr)
            f_i = fap[i]*sin(dot_sr)
            if use_u > 0:
                if use_u == 1:
                    sus = u_scalar * (sp[3*i+0] * sp[3*i+0]
                                      + sp[3*i+1] * sp[3*i+1]
                                      + sp[3*i+2] * sp[3*i+2])
                else:
                    sus = up[3*0+0] * sp[3*i+0] * sp[3*i+0] + \
                          up[3*1+1] * sp[3*i+1] * sp[3*i+1] + \
                          up[3*2+2] * sp[3*i+2] * sp[3*i+2] + \
                          2. * up[3*0+1] * sp[3*i+0] * sp[3*i+1] + \
                          2. * up[3*0+2] * sp[3*i+0] * sp[3*i+2] + \
                          2. * up[3*1+2] * sp[3*i+1] * sp[3*i+2]
                dwf = exp(twopisq*sus)
                f_r = f_r*dwf
                f_i = f_i*dwf
            resp[2*i] = resp[2*i] + f_r
            resp[2*i+1] = resp[2*i+1] + f_i

    else:
        # If any of the input arrays is not contiguous (which is
        # highly unlikely), fall back to Python version.
        sf = N.exp(twopi*1j*(N.dot(s, r)))
        if use_u == 1:
            dwf = N.exp(twopisq*u_scalar*N.sum(sv*sv, axis=-1))
            N.add(result, f_atom*dwf*sf, result)
        if use_u == 2:
            dwf = N.exp(twopisq*N.sum(N.dot(s, u)*s, axis=-1))
            N.add(result, f_atom*dwf*sf, result)
        else:
            N.add(result, f_atom*sf, result)
