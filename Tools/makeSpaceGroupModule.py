# This script generates the module SpaceGroups.
#
# The space group information is taken from cctbx, which must be
# available to run this script.
#
# Written by Konrad Hinsen
# last revision: 2007-9-21
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


print """
# This module has been generated automatically from space group information
# obtained from the Computational Crystallography Toolbox
#

from Scientific import N

class SpaceGroup(object):

    def __init__(self, number, symbol, transformations):
        self.number = number
        self.symbol = symbol
        self.transformations = transformations
        self.transposed_rotations = [N.transpose(t[0])
                                     for t in self.transformations]

    def __repr__(self):
        return 'SpaceGroup(%d, %s)' % (self.number, repr(self.symbol))

    def __len__(self):
        return len(self.transformations)

    def symmetryEquivalentMillerIndices(self, hkl):
        hkl_list = []
        for rot in self.transposed_rotations:
            hkl_list.append(N.dot(rot, hkl))
        return hkl_list

space_groups = {}
"""

for number in range(1, 231):
    sgi = space_group_info(number)
    unique_name = sgi.symbol_and_number() 
    symbol = unique_name[:unique_name.find('(No.')].strip()
    space_group_table_entry(number, symbol, sgi)

print """
del transformations
del rot
del trans_num
del trans_den
del sg
"""
