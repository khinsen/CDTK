# Calculate math functions using GSL
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

cdef extern from "gsl/gsl_sf_bessel.h":

    double gsl_sf_bessel_I0(double x)
    double gsl_sf_bessel_I1(double x)
    double gsl_sf_bessel_I0_scaled(double x)
    double gsl_sf_bessel_I1_scaled(double x)

cdef extern from "gsl/gsl_sf_log.h":

    double gsl_sf_log(double x)

cdef extern from "gsl/gsl_sf_exp.h":

    double gsl_sf_exp(double x)


def I0(double x):
    return gsl_sf_bessel_I0(x)

def I1(double x):
    return gsl_sf_bessel_I1(x)

def I1divI0(double x):
    return gsl_sf_bessel_I1_scaled(x)/gsl_sf_bessel_I0_scaled(x)

def logI0(double x):
    return x+gsl_sf_log(gsl_sf_bessel_I0_scaled(x))

def logcosh(double x):
    return x+gsl_sf_log(0.5*(1.+gsl_sf_exp(-2.*x)))

