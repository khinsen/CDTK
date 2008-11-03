# This script generates the module CDTK.SpaceGroups.
#
# The space group information is taken from cctbx, which must be
# available to run this script.
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

from cctbx.sgtbx import space_group_info

def format_numerator(r):
    s = str(r)
    if '/' in s:
        return s.split('/')[0]
    else:
        return s

def format_denominator(r):
    s = str(r)
    if '/' in s:
        return s.split('/')[1]
    else:
        return "1"

def space_group_table_entry(number, symbol, sgi):
    group = sgi.group()
    print "transformations = []"
    for symmetry_transformation in group:
        rot =  symmetry_transformation.as_rational().r
        trans =  symmetry_transformation.as_rational().t
        print 'rot = N.array([' + \
              ','.join([str(x) for x in rot]) + '])'
        print 'rot.shape = (3, 3)'
        print 'trans_num = N.array([' + \
              ','.join([format_numerator(x) for x in trans]) + '])'
        print 'trans_den = N.array([' + \
              ','.join([format_denominator(x) for x in trans]) + '])'
        print 'transformations.append((rot, trans_num, trans_den))'
    print 'sg = SpaceGroup(%d, %s, transformations)' % (number, repr(symbol))
    print "space_groups[%d] = sg" % number
    print "space_groups[%s] = sg" % repr(symbol)
    print


print '''
# This module has been generated automatically from space group information
# obtained from the Computational Crystallography Toolbox
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
Space groups in crystallography

This module contains a list of all the 230 space groups that can occur in
a crystal. The variable space_groups contains a dictionary that maps
space group numbers and space group names to the corresponding space
group objects.
"""

from Scientific import N

class SpaceGroup(object):

    """
    Space group

    All possible space group objects are created in this module. Other
    modules should access these objects through the dictionary
    space_groups rather than create their own space group objects.
    """

    def __init__(self, number, symbol, transformations):
        """
        @param number: the number assigned to the space group by
                       international convention
        @type number: C{int}
        @param symbol: the Hermann-Mauguin space-group symbol as used
                       in PDB and mmCIF files
        @type symbol: C{str}
        @param transformations: a list of space group transformations,
                                each consisting of a tuple of three
                                integer arrays (rot, tn, td), where
                                rot is the rotation matrix and tn/td
                                are the numerator and denominator of the
                                translation vector. The transformations
                                are defined in fractional coordinates.
        @type transformations: C{list}
        """
        self.number = number
        self.symbol = symbol
        self.transformations = transformations
        self.transposed_transformations = [(N.transpose(t[0]), t[1], t[2])
                                           for t in self.transformations]

    def __repr__(self):
        return "SpaceGroup(%d, %s)" % (self.number, repr(self.symbol))

    def __len__(self):
        """
        @return: the number of space group transformations
        @rtype: C{int}
        """
        return len(self.transformations)

    def symmetryEquivalentMillerIndices(self, hkl):
        """
        @param hkl: a set of Miller indices
        @type hkl: C{Scientific.N.array_type}
        @return: a tuple (miller_indices, phase_factor) of two lists
                 of length equal to the number of space group
                 transformations. miller_indices contains the Miller
                 indices of each reflection equivalent by symmetry to the
                 reflection hkl (inclduing hkl itself as the first element).
                 phase_factor contains the phase factors that must be applied
                 to the structure factor of reflection hkl to obtain the
                 structure factor of the symmetry equivalent reflection.
        @rtype: C{tuple}
        """
        hkl_list = []
        phase_factor_list = []
        for rot, tn, td in self.transposed_transformations:
            hkl_list.append(N.dot(rot, hkl))
            t = (tn*1.)/td
            phase_factor_list.append(N.exp(-2j*N.pi*N.dot(hkl, t)))
        return hkl_list, phase_factor_list

space_groups = {}
'''

for number in range(1, 231):
    sgi = space_group_info(number)
    unique_name = sgi.symbol_and_number() 
    symbol = unique_name[:unique_name.find('(No.')].strip()
    space_group_table_entry(number, symbol, sgi)

print """

# Add alternative symbols
space_groups['R 3'] = space_groups['H 3'] = space_groups[146]
space_groups['R 3 2'] = space_groups['H 3 2'] = space_groups[155]

space_groups['C 2'] = space_groups['C 1 2 1']
space_groups['I 1 2 1'] = space_groups['C 1 2 1']
space_groups['B 2'] = space_groups['C 1 2 1']

space_groups['P 21'] = space_groups['P 1 21 1']
space_groups['P 1 1 21'] = space_groups['P 1 21 1']

space_groups['P 2 21 21'] = space_groups['P 21 21 2']

# Make it possible to use space group symbols without spaces, which occur
# in several PDB entries.
for k, sg in space_groups.items():
    if not isinstance(k, int):
        compacted = ''.join(k.split())
        if compacted not in space_groups:
            space_groups[compacted] = sg

del transformations
del rot
del trans_num
del trans_den
del sg
"""
