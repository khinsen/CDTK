# Test class UnitCell
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest
from CDTK.Crystal import UnitCell
from CDTK.SpaceGroups import space_groups
from CDTK.Utility import largestAbsoluteElement
from Scientific.Geometry import Vector, delta
from Scientific import N

deg = N.pi/180.

# The datasets are taken from the CRYST1 and SCALE records of
# PDB files.
datasets = [
    # 2PKV
    ((45.030, 45.030, 41.750, 90.00*deg, 90.00*deg, 120.00*deg),
     [Vector(0.022207, 0.012821, 0.000000),
      Vector(0.000000, 0.025643, 0.000000),
      Vector(0.000000, 0.000000, 0.023952)],
      "P 61"),

    # 1IEE
    ((77.061,  77.061,  37.223, 90.00*deg, 90.00*deg, 90.00*deg),
     [Vector(0.012977, 0.000000, 0.000000),
      Vector(0.000000, 0.012977, 0.000000),
      Vector(0.000000, 0.000000, 0.026865)],
      "P 43 21 2"),

    # 1BFF
    ((30.800,  33.300,  36.500, 64.10*deg, 73.00*deg, 76.10*deg),
     [Vector(0.032468,-0.008035,-0.007351),
      Vector(0.000000, 0.030936,-0.013296),
      Vector(0.000000, 0.000000, 0.031183)],
      "P 1"),

    # 1SLI
    ((46.443,  69.516,  72.514,113.14*deg, 95.39*deg, 106.87*deg),
     [Vector(0.021532, 0.006530, 0.005473),
      Vector(0.000000, 0.015032, 0.007388),
      Vector(0.000000, 0.000000, 0.015434)],
      "P 1"),

    # 3LZT
    ((26.650,  30.800,  33.630, 88.30*deg, 107.40*deg, 112.20*deg),
     [Vector(0.037523, 0.015313, 0.013262),
      Vector(0.000000, 0.035067, 0.003322),
      Vector(0.000000, 0.000000, 0.031301)],
      "P 1"),

    # 1XVM
    ((33.277,  37.037,  40.100, 102.22*deg, 103.93*deg, 102.59*deg),
     [Vector(0.030051, 0.006712, 0.009710),
      Vector(0.000000, 0.027665, 0.008033),
      Vector(0.000000, 0.000000, 0.026755)],
      "P 1"),

    # 135L
    ((38.070, 33.200, 46.120, 90.00*deg, 110.06*deg, 90.00*deg),
     [Vector(0.026267, 0.000000, 0.009592),
      Vector(0.000000, 0.030120, 0.000000),
      Vector(0.000000, 0.000000, 0.023083)],
     "P 1 21 1"),

    # 1K6L
    ((141.500, 141.500, 187.200, 90.00*deg, 90.00*deg, 120.00*deg),
     [Vector(0.007067, 0.004080, 0.000000),
      Vector(0.000000, 0.008160, 0.000000),
      Vector(0.000000, 0.000000, 0.005342)],
     "P 31 2 1"),
 ]

fractional_coordinates = [N.array([0., 0.1, 0.2]),
                          N.array([-0.5, 1., 0.3]),
                          N.array([2., -1., 3.]),
                          N.array([-1.1, 5., -2.3])]

class UnitCellTests(unittest.TestCase):

    def test_geometry(self):
        for params, rb, sg in datasets:
            cell1 = UnitCell(*params)
            basis = cell1.basisVectors()
            reciprocal_basis = cell1.reciprocalBasisVectors()
            for i in range(3):
                self.assert_((reciprocal_basis[i]-rb[i]).length()
                             < 1.e-6)
            for i in range(3):
                for j in range(3):
                    p = basis[i]*reciprocal_basis[j]
                    if i == j:
                        self.assertAlmostEqual(p, 1., 10)
                    else:
                        self.assertAlmostEqual(p, 0., 10)
            cell2 = UnitCell(*tuple(basis))
            self.assertAlmostEqual(cell1.a, cell2.a, 5)
            self.assertAlmostEqual(cell1.b, cell2.b, 5)
            self.assertAlmostEqual(cell1.c, cell2.c, 5)
            self.assertAlmostEqual(cell1.alpha, cell2.alpha, 5)
            self.assertAlmostEqual(cell1.beta, cell2.beta, 5)
            self.assertAlmostEqual(cell1.gamma, cell2.gamma, 5)

    def test_conversions(self):
        for params, rb, sg in datasets:
            cell = UnitCell(*params)
            m_fc = cell.fractionalToCartesianMatrix()
            m_cf = cell.cartesianToFractionalMatrix()
            for x in fractional_coordinates:
                r = cell.fractionalToCartesian(x)
                rr = N.dot(m_fc, x)
                self.assert_(largestAbsoluteElement(r.array-rr) < 1.e-10)
                xx = cell.cartesianToFractional(r)
                self.assert_(largestAbsoluteElement(x-xx) < 1.e-15)
                xx = N.dot(m_cf, rr)
                self.assert_(largestAbsoluteElement(x-xx) < 1.e-15)

    def test_symmetry_ops(self):
        for params, rb, sg in datasets:
            cell = UnitCell(*params)
            sg = space_groups[sg]
            transformations = cell.cartesianCoordinateSymmetryTransformations(sg)
            for t in transformations:
                # Check that the transformation is a rotation-translation.
                error = N.absolute(N.multiply.reduce(t.tensor.eigenvalues())-1.)
                self.assert_(error < 1.e-12)
                # Check that applying the transformation N times (where N is
                # the number of symmetry operations) yields a transformation
                # without rotation.
                ntimes = t
                for i in range(1, len(sg)):
                    ntimes = ntimes*t
                self.assert_(largestAbsoluteElement(
                               (ntimes.tensor-delta).array) < 1.e-12)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(UnitCellTests)

if __name__ == '__main__':
    unittest.main()
