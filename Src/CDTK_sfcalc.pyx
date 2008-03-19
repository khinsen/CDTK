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

#
# Calculate the structure factor for a single atom.
#
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
           and PyArray_ISCONTIGUOUS(r):

        ns = result.dimensions[0]
        assert result.nd == 1
        assert s.nd == 2 and s.dimensions[0] == ns and s.dimensions[1] == 3
        assert f_atom.nd == 1 and f_atom.dimensions[0] == ns
        assert r.nd == 1 and r.dimensions[0] == 3
        assert result.descr.elsize == 2*sizeof(double)
        assert s.descr.elsize == sizeof(double)
        assert f_atom.descr.elsize == sizeof(double)
        assert r.descr.elsize == sizeof(double)
        if use_u == 2:
            assert PyArray_ISCONTIGUOUS(u)
            assert u.descr.elsize == sizeof(double)
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

#
# Calculate structure factor and position/ADP derivatives
# (used by RefinementEngine)
#
def sfDeriv(array_type element_indices, array_type f_atom, array_type positions,
            array_type adps, array_type occupancies, array_type sv,
            array_type p,
            array_type sf, array_type pd, array_type adpd,
            array_type deriv, array_type sf_in, array_type a_in):
    cdef int natoms, ns, nsg
    cdef int do_sf, do_pd, do_adpd
    cdef int *element_indices_d
    cdef double *f_atom_d, *positions_d, *adps_d, *occupancies_d, *sv_d, *p_d
    cdef double *sf_d, *pd_d, *adpd_d, *deriv_d, *sf_in_d, *a_in_d
    cdef double *f_atom_p, *positions_p, *adps_p, occupancy, *sv_p, *p_p
    cdef double *pd_p, *adpd_p
    cdef double dwf, pf_arg, pf_r, pf_i, sf_abs, sf_r, sf_i, pd_f, adpd_f
    cdef int i, j, k

    assert PyArray_ISCONTIGUOUS(element_indices)
    assert element_indices.descr.elsize == sizeof(int)
    assert PyArray_ISCONTIGUOUS(f_atom)
    assert f_atom.descr.elsize == sizeof(double)
    assert PyArray_ISCONTIGUOUS(positions)
    assert positions.descr.elsize == sizeof(double)
    assert PyArray_ISCONTIGUOUS(adps)
    assert adps.descr.elsize == sizeof(double)
    assert PyArray_ISCONTIGUOUS(occupancies)
    assert occupancies.descr.elsize == sizeof(double)
    assert PyArray_ISCONTIGUOUS(sv)
    assert sv.descr.elsize == sizeof(double)
    assert PyArray_ISCONTIGUOUS(p)
    assert p.descr.elsize == 2*sizeof(double)

    natoms = positions.dimensions[0]
    ns = p.dimensions[1]
    nsg = p.dimensions[0]

    do_sf = sf.dimensions[0] > 0
    do_pd = pd.dimensions[0] > 0
    do_adpd = adpd.dimensions[0] > 0
    if do_sf:
        assert PyArray_ISCONTIGUOUS(sf)
        assert sf.descr.elsize == 2*sizeof(double)
    if do_pd or do_adpd:
        assert deriv.dimensions[0] > 0
        assert PyArray_ISCONTIGUOUS(deriv)
        assert deriv.descr.elsize == sizeof(double)
        assert PyArray_ISCONTIGUOUS(sf_in)
        assert sf_in.descr.elsize == 2*sizeof(double)
        assert PyArray_ISCONTIGUOUS(a_in)
        assert a_in.descr.elsize == sizeof(double)
    if do_pd:
        assert PyArray_ISCONTIGUOUS(pd)
        assert pd.descr.elsize == sizeof(double)
    if do_adpd:
        assert PyArray_ISCONTIGUOUS(adpd)
        assert adpd.descr.elsize == sizeof(double)

    element_indices_d = <int *>element_indices.data
    f_atom_d = <double *>f_atom.data
    positions_d = <double *>positions.data
    adps_d = <double *>adps.data
    occupancies_d = <double *>occupancies.data
    sv_d = <double *>sv.data
    p_d = <double *>p.data
    sf_d = <double *>sf.data
    pd_d = <double *>pd.data
    adpd_d = <double *>adpd.data
    deriv_d = <double *>deriv.data
    sf_in_d = <double *>sf_in.data
    a_in_d = <double *>a_in.data

    for i from 0 <= i < natoms:
        f_atom_p = f_atom_d + ns*element_indices_d[i]
        positions_p = positions_d + 3*i
        adps_p = adps_d + 6*i
        occupancy = occupancies_d[i]
        pd_p = pd_d + 3*i
        adpd_p = adpd_d + 6*i
        for j from 0 <= j < nsg:
            sv_p = sv_d + 3*ns*j
            p_p = p_d + 2*ns*j
            for k from 0 <= k < ns:
                dwf = exp(twopisq*(adps_p[0]*sv_p[3*k]*sv_p[3*k] +
                                   adps_p[1]*sv_p[3*k+1]*sv_p[3*k+1] +
                                   adps_p[2]*sv_p[3*k+2]*sv_p[3*k+2] +
                                   2.*adps_p[3]*sv_p[3*k+1]*sv_p[3*k+2] +
                                   2.*adps_p[4]*sv_p[3*k]*sv_p[3*k+2] +
                                   2.*adps_p[5]*sv_p[3*k]*sv_p[3*k+1]))
                pf_arg = twopi*(sv_p[3*k]*positions_p[0] +
                                sv_p[3*k+1]*positions_p[1] +
                                sv_p[3*k+2]*positions_p[2])
                pf_r = cos(pf_arg)
                pf_i = sin(pf_arg)
                sf_abs = occupancy*f_atom_p[k]*dwf
                sf_r = sf_abs*(pf_r*p_p[2*k]-pf_i*p_p[2*k+1])
                sf_i = sf_abs*(pf_i*p_p[2*k]+pf_r*p_p[2*k+1])
                if do_sf:
                    sf_d[2*k] = sf_d[2*k] + sf_r
                    sf_d[2*k+1] = sf_d[2*k+1] + sf_i
                if do_pd:
                    pd_f = twopi*deriv_d[k] * \
                         (sf_in_d[2*k+1]*sf_r - sf_in_d[2*k]*sf_i)/a_in_d[k]
                    pd_p[0] = pd_p[0] + pd_f*sv_p[3*k]
                    pd_p[1] = pd_p[1] + pd_f*sv_p[3*k+1]
                    pd_p[2] = pd_p[2] + pd_f*sv_p[3*k+2]
                if do_adpd:
                    adpd_f = (sf_in_d[2*k]*sf_r + sf_in_d[2*k+1]*sf_i) \
                               * deriv_d[k] / a_in_d[k]
                    adpd_p[0] = adpd_p[0] + adpd_f*sv_p[3*k]*sv_p[3*k]
                    adpd_p[1] = adpd_p[1] + adpd_f*sv_p[3*k+1]*sv_p[3*k+1]
                    adpd_p[2] = adpd_p[2] + adpd_f*sv_p[3*k+2]*sv_p[3*k+2]
                    adpd_p[3] = adpd_p[3] + 2.*adpd_f*sv_p[3*k+1]*sv_p[3*k+2]
                    adpd_p[4] = adpd_p[4] + 2.*adpd_f*sv_p[3*k+0]*sv_p[3*k+2]
                    adpd_p[5] = adpd_p[5] + 2.*adpd_f*sv_p[3*k+0]*sv_p[3*k+1]
