import unittest
from CDTK.Crystal import UnitCell
from Scientific.Geometry import Vector
from Scientific import N

deg = N.pi/180.

# The datasets are taken from the CRYST1 and SCALE records of
# PDB files.
datasets = [
    ((45.030, 45.030, 41.750, 90.00*deg, 90.00*deg, 120.00*deg),
     [Vector(0.022207, 0.012821, 0.000000),
      Vector(0.000000, 0.025643, 0.000000),
      Vector(0.000000, 0.000000, 0.023952)]),

    ((77.061,  77.061,  37.223, 90.00*deg, 90.00*deg, 90.00*deg),
     [Vector(0.012977, 0.000000, 0.000000),
      Vector(0.000000, 0.012977, 0.000000),
      Vector(0.000000, 0.000000, 0.026865)]),

    ((30.800,  33.300,  36.500, 64.10*deg, 73.00*deg, 76.10*deg),
     [Vector(0.032468,-0.008035,-0.007351),
      Vector(0.000000, 0.030936,-0.013296),
      Vector(0.000000, 0.000000, 0.031183)]),

    ((46.443,  69.516,  72.514,113.14*deg, 95.39*deg, 106.87*deg),
     [Vector(0.021532, 0.006530, 0.005473),
      Vector(0.000000, 0.015032, 0.007388),
      Vector(0.000000, 0.000000, 0.015434)]),

    ((26.650,  30.800,  33.630, 88.30*deg, 107.40*deg, 112.20*deg),
     [Vector(0.037523, 0.015313, 0.013262),
      Vector(0.000000, 0.035067, 0.003322),
      Vector(0.000000, 0.000000, 0.031301)]),

    ((33.277,  37.037,  40.100, 102.22*deg, 103.93*deg, 102.59*deg),
     [Vector(0.030051, 0.006712, 0.009710),
      Vector(0.000000, 0.027665, 0.008033),
      Vector(0.000000, 0.000000, 0.026755)]),
    ]

fractional_coordinates = [N.array([0., 0.1, 0.2]),
                          N.array([-0.5, 1., 0.3]),
                          N.array([2., -1., 3.]),
                          N.array([-1.1, 5., -2.3])]

class UnitCellTests(unittest.TestCase):

    def test_geometry(self):
        for params, rb in datasets:
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
                        self.assertAlmostEqual(p, 1., 10.)
                    else:
                        self.assertAlmostEqual(p, 0., 10.)
            cell2 = UnitCell(*tuple(basis))
            self.assertAlmostEqual(cell1.a, cell2.a, 5)
            self.assertAlmostEqual(cell1.b, cell2.b, 5)
            self.assertAlmostEqual(cell1.c, cell2.c, 5)
            self.assertAlmostEqual(cell1.alpha, cell2.alpha, 5)
            self.assertAlmostEqual(cell1.beta, cell2.beta, 5)
            self.assertAlmostEqual(cell1.gamma, cell2.gamma, 5)

    def test_conversions(self):
        for params, rb in datasets:
            cell = UnitCell(*params)
            m_fc = cell.fractionalToCartesianMatrix()
            m_cf = cell.cartesianToFractionalMatrix()
            for x in fractional_coordinates:
                r = cell.fractionalToCartesian(x)
                rr = N.dot(m_fc, x)
                self.assert_(N.maximum.reduce(N.fabs(r.array-rr)) < 1.e-15)
                xx = cell.cartesianToFractional(r)
                self.assert_(N.maximum.reduce(N.fabs(x-xx)) < 1.e-15)
                xx = N.dot(m_cf, rr)
                self.assert_(N.maximum.reduce(N.fabs(x-xx)) < 1.e-15)

if __name__ == '__main__':
    unittest.main()
