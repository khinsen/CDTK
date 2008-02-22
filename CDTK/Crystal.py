# Description of a crystal
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Description of a crystal
"""

from Scientific.Geometry import Vector, isVector
from Scientific.Geometry.Transformation import Rotation, Translation, Shear
from Scientific import N, LA

class UnitCell(object):

    """
    Unit cell
    """

    def __init__(self, *parameters):
        """
        @param parameters: one of 1) three lattice vectors or
            2) six numbers: the lengths of the three lattice vectors (a, b, c)
            followed by the three angles (alpha, beta, gamma).
        """
        if len(parameters) == 6:
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma = \
                    parameters
            e1 = Vector(self.a, 0, 0)
            e2 = self.b*Vector(N.cos(self.gamma), N.sin(self.gamma), 0.)
            e3_x = N.cos(self.beta)
            e3_y = (N.cos(self.alpha)-N.cos(self.beta)*N.cos(self.gamma)) \
                   / N.sin(self.gamma)
            e3_z = N.sqrt(1.-e3_x**2-e3_y**2)
            e3 = self.c*Vector(e3_x, e3_y, e3_z)
            self.basis = [e1, e2, e3]
        elif len(parameters) == 3:
            assert isVector(parameters[0])
            assert isVector(parameters[1])
            assert isVector(parameters[2])
            self.basis = list(parameters)
            e1, e2, e3 = self.basis
            self.a = e1.length()
            self.b = e2.length()
            self.c = e3.length()
            self.alpha = N.arccos(e2*e3/(self.b*self.c))
            self.beta  = N.arccos(e1*e3/(self.a*self.c))
            self.gamma = N.arccos(e1*e2/(self.a*self.b))
        else:
            raise ValueError("Parameter list incorrect")

        r = LA.inverse(N.transpose([e1, e2, e3]))
        self.reciprocal_basis = [Vector(r[0]), Vector(r[1]), Vector(r[2])]

    def basisVectors(self):
        """
        @return: a list containing the three lattice vectors
        @rtype: C{list} of C{Scientific.Geometry.Vector}
        """
        return self.basis

    def reciprocalBasisVectors(self):
        """
        @return: a list containing the three basis vectors
                 of the reciprocal lattice
        @rtype: C{list} of C{Scientific.Geometry.Vector}
        """
        return self.reciprocal_basis

    def cellVolume(self):
        """
        @return: the volume of the unit cell
        @rtype: C{float}
        """
        e1, e2, e3 = self.basis
        return e1*e2.cross(e3)

    def cartesianToFractional(self, vector):
        """
        @param vector: a vector in real Cartesian space
        @type vector: C{Scientific.Geometry.Vector}
        @return: the vector in fractional coordinates
        @rtype: C{Scientific.N.array_type}
        """
        r1, r2, r3 = self.reciprocal_basis
        return N.array([r1*vector, r2*vector, r3*vector])

    def cartesianToFractionalMatrix(self):
        """
        @return: the 3x3 conversion matrix from real Cartesian space
                 coordinates to fractional coordinates
        """
        return N.array(self.reciprocal_basis)

    def fractionalToCartesian(self, array):
        """
        @param array: a vector in fractional coordinates
        @type array: C{Scientific.N.array_type}
        @return: the vector in real Cartesian space
        @rtype: C{Scientific.Geometry.Vector}
        """
        e1, e2, e3 = self.basis
        return array[0]*e1 + array[1]*e2 + array[2]*e3

    def fractionalToCartesianMatrix(self):
        """
        @return: the 3x3 conversion matrix from fractional
                 coordinates to real Cartesian space coordinates
        """
        return N.transpose(self.basis)

    def minimumImageDistanceVector(self, point1, point2):
        """
        @param point1: a point in the unit cell
        @type point1: C{Scientific.Geometry.Vector}
        @param point2: a point in the unit cell
        @type point2: C{Scientific.Geometry.Vector}
        @return: the minimum-image vector from point1 to point2
        @rtype: C{Scientific.Geometry.Vector}
        """
        d = self.cartesianToFractional(point2-point1)
        d = d - (d > 0.5) + (d <= -0.5)
        return self.fractionalToCartesian(d)
        
    def isCompatibleWith(self, other_cell, precision=1.e-5):
        """
        @param other_cell: a unit cell
        @type other_cell: L{UnitCell} or C{MMTK.Universe}
        @param precision: the absolute precision of the comparison
        @type precision: C{float}
        @return: C{True} if the lattice vectors of the two unit cells differ
                 by a vector of length < precision
        @rtype: C{bool}
        """
        other_basis = other_cell.basisVectors()
        for i in range(3):
            if (other_basis[i]-self.basis[i]).length() > precision:
                return False
        return True

    def cartesianCoordinateSymmetryOperations(self, space_group):
        """
        @param space_group: a space group
        @type space_group: L{CDTK.SpaceGroups.SpaceGroup}
        @return: a list of transformation objects representing the symmetry
                 operations of the space group in the Cartesian coordinates
                 of the unit cell
        @rtype: C{list} of C{Scientific.Geometry.Transformation.Transformation}
        """
        transformations = []
        to_fract = Shear(self.cartesianToFractionalMatrix())
        from_fract = Shear(self.fractionalToCartesianMatrix())
        for rot, trans_num, trans_den in space_group.transformations:
            trans = Vector((1.*trans_num)/trans_den)
            tr_fract = Translation(trans)*Rotation(rot)
            transformations.append(from_fract*tr_fract*to_fract)
        return transformations
