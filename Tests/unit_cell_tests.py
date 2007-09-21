import unittest
from Crystallography.Crystal import UnitCell
from Scientific.Geometry import Vector
from Scientific import N

deg = N.pi/180.

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


class UnitCellTests(unittest.TestCase):

    def test_geometry(self):
        for params, rb in datasets:
            cell1 = UnitCell(*params)
            for i in range(3):
                self.assert_((cell1.reciprocal_basis[i]-rb[i]).length()
                             < 1.e-6)
            cell2 = UnitCell(*tuple(cell1.basis))
            self.assertAlmostEqual(cell1.a, cell2.a, 5)
            self.assertAlmostEqual(cell1.b, cell2.b, 5)
            self.assertAlmostEqual(cell1.c, cell2.c, 5)
            self.assertAlmostEqual(cell1.alpha, cell2.alpha, 5)
            self.assertAlmostEqual(cell1.beta, cell2.beta, 5)
            self.assertAlmostEqual(cell1.gamma, cell2.gamma, 5)


if __name__ == '__main__':
    unittest.main()
