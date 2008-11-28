# Generate the module CDTK.AtomicScatteringFactors
# The parameters are taken from the CCP4 distribution.
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

from CDTK import Units

lines = file('/Users/hinsen/Temp/ccp4-6.0.2/lib/data/atomsf.lib').readlines()
lines = lines[31:]

print '''
# Atomic scattering factors in Five-Gaussian approximation.
# The formula for the atomic scattering factor is
#   f(s) = N.sum(a*N.exp(-b*s^2))
# where (a, b) are the two arrays stored in atomic_scattering_factors
# and s is the length of the scattering vector (in 1/nm).
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
Atomic scattering factors for all elements in the Five-Gaussian approximation.

The dictionary atomic_scattering_factors maps lower-case chemical element
symbols or (element, ionic_charge) tuples to tuples (a, b) containing
the parameters of the five-Gaussian approximation. a and b are arrays
of five elements each. The scattering factor for the atom is given by

f(s) = N.sum( a * exp(-b * s**2) )

where s is the length of the scattering vector.
"""

from Scientific import N

atomic_scattering_factors = {
'''

while lines:

    atom_type = lines[0].strip()
    w, n_el, c = [eval(s) for s in lines[1].split()]
    a = [float(s) for s in lines[2].split()]
    b = [float(s) for s in lines[3].split()]
    lines = lines[5:]
    
    if ' ' in atom_type:
        continue
    if '-' in atom_type:
        element, charge = atom_type.split('-')
        charge = -int(charge)
    elif '+' in atom_type:
        element, charge = atom_type.split('+')
        charge = int(charge)
    else:
        element = atom_type
        charge = 0

    if w-charge != n_el:
        #print "!!!"
        #print element, charge, w, n_el
        #print "!!!"
        continue

    array1 = "N.array([%12.8f, %12.8f, %12.8f, %12.8f, %12.8f])" \
             % (a[0], a[1], a[2], a[3], c)
    array2 = "N.array([%12.8f, %12.8f, %12.8f, %12.8f, %12.8f])" \
             % (b[0]*Units.Ang**2/4., b[1]*Units.Ang**2/4.,
                b[2]*Units.Ang**2/4., b[3]*Units.Ang**2/4., 0.)
    print "('%s', %d): (%s," % (element.lower(), charge, array1)
    print "             %s)," % array2

print """('x', 0): (N.array([ 0., 0., 0., 0., 0.]),
             N.array([0., 0., 0., 0., 0.])),
}

for (element, charge), value in atomic_scattering_factors.items():
    if charge == 0:
       atomic_scattering_factors[element] = value

del element
del charge
del value
"""
