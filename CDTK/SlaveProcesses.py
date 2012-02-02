# Structure refinement compute slave
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
This module contains the compute slave code for the module CDTK.Refinement.

.. moduleauthor:: Konrad Hinsen <konrad.hinsen@cnrs-orleans.fr>

"""

from Scientific import N
from CDTK_sfcalc import sfDeriv

dummy_array = N.zeros((0,), N.Int)

def do_sf(i1, i2, positions, adps, occupancies, element_indices, f_atom, sv, p):
   sf = N.zeros((sv.shape[1],), N.Complex)
   sfDeriv(element_indices[i1:i2], f_atom, positions,
            adps, occupancies, sv, p, sf,
            dummy_array, dummy_array, dummy_array, dummy_array, dummy_array)
   return sf

def do_sf_pd(i1, i2, positions, adps, occupancies, deriv, sf_in, a_in,
             element_indices, f_atom, sv, p):
   pd = N.zeros(positions.shape, N.Float)
   sfDeriv(element_indices[i1:i2], f_atom, positions,
           adps, occupancies, sv, p, dummy_array, pd, dummy_array,
           deriv, sf_in, a_in)
   return i1, i2, pd

def do_sf_adpd(i1, i2, positions, adps, occupancies, deriv, sf_in, a_in,
               element_indices, f_atom, sv, p):
   adpd = N.zeros(adps.shape, N.Float)
   sfDeriv(element_indices[i1:i2], f_atom, positions,
           adps, occupancies, sv, p, dummy_array, dummy_array, adpd,
           deriv, sf_in, a_in)
   return i1, i2, adpd
