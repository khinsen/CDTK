# Test utility functions
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest
import random
from Scientific import N
from Scientific.Geometry.Transformation import Rotation
from Scientific.Geometry import Vector
from CDTK.Crystal import UnitCell
from CDTK.SpaceGroups import space_groups
from CDTK.Units import deg
from CDTK.Utility import SymmetricTensor, symmetricTensorBasis
from CDTK.Utility import largestAbsoluteElement

from CDTK import Utility

datasets = [
    # 2PKV
    ((45.030, 45.030, 41.750, 90.00*deg, 90.00*deg, 120.00*deg),
     [Vector(0.022207, 0.012821, 0.000000),
      Vector(0.000000, 0.025643, 0.000000),
      Vector(0.000000, 0.000000, 0.023952)],
      "P 61", 2),

    # 1IEE
    ((77.061,  77.061,  37.223, 90.00*deg, 90.00*deg, 90.00*deg),
     [Vector(0.012977, 0.000000, 0.000000),
      Vector(0.000000, 0.012977, 0.000000),
      Vector(0.000000, 0.000000, 0.026865)],
      "P 43 21 2", 2),

    # 1BFF
    ((30.800,  33.300,  36.500, 64.10*deg, 73.00*deg, 76.10*deg),
     [Vector(0.032468,-0.008035,-0.007351),
      Vector(0.000000, 0.030936,-0.013296),
      Vector(0.000000, 0.000000, 0.031183)],
      "P 1", 6),

    # 135L
    ((38.070, 33.200, 46.120, 90.00*deg, 110.06*deg, 90.00*deg),
     [Vector(0.026267, 0.000000, 0.009592),
      Vector(0.000000, 0.030120, 0.000000),
      Vector(0.000000, 0.000000, 0.023083)],
     "P 1 21 1", 4),

    # 1K6L
    ((141.500, 141.500, 187.200, 90.00*deg, 90.00*deg, 120.00*deg),
     [Vector(0.007067, 0.004080, 0.000000),
      Vector(0.000000, 0.008160, 0.000000),
      Vector(0.000000, 0.000000, 0.005342)],
     "P 31 2 1", 2),
 ]

def randomArray(shape):
    return N.reshape([random.random() for i in range(N.product(shape))],
                     shape)

def randomRotationMatrix():
    axis = Vector(randomArray((3,)))
    angle = 2.*N.pi*random.random()
    return Rotation(axis, angle).tensor.array


class UtilityTests(unittest.TestCase):

    def test_symmetric_tensors(self):
        t = N.array([[ 1.,  -0.5,  0.3],
                     [-0.5,  2.5, -1.1],
                     [ 0.3, -1.1,  2.]])
        tc = N.array([1., 2.5, 2., -1.1, 0.3, -0.5])
        error = largestAbsoluteElement(SymmetricTensor(t).array-tc)
        self.assert_(error == 0.)
        for i in range(50):
            t = SymmetricTensor(randomArray((6,)))
            tf = t.array2d
            tc = SymmetricTensor(tf)
            error = largestAbsoluteElement(tf-tc.array2d)
            self.assert_(error == 0.)
        for i in range(10):
            t = randomArray((3,3))
            self.assertRaises(ValueError, SymmetricTensor, t)

    def test_symmetric_eigenvalues(self):
        for i in range(50):
            ev = randomArray((3,))
            d = randomRotationMatrix()
            t = SymmetricTensor(*(tuple(ev)+(0., 0., 0.))) \
                .applyRotationMatrix(d)
            ev_c = t.eigenvalues()
            if N.minimum.reduce(ev_c) > 0.:
                self.assert_(t.isPositiveDefinite())
            else:
                self.assert_(not t.isPositiveDefinite())
            self.assert_(largestAbsoluteElement(N.sort(ev)-ev_c) < 1.e-8)
            self.assert_(N.multiply.reduce(ev) - t.determinant() < 1.e-10)
            if N.sum(ev >= 0.) == 3 or N.sum(ev <= 0.) == 3:
                td = t.makeDefinite()
                self.assert_(largestAbsoluteElement(td.array-t.array) < 1.e-12)
                
    def test_symmetric_inversion(self):
        for i in range(100):
            t = SymmetricTensor(randomArray((6,)))
            if abs(t.trace()) > 1.e-5:
                t_inv = t.inverse()
                diff  = N.dot(t_inv.array2d, t.array2d) - N.identity(3)
                self.assert_(largestAbsoluteElement(diff) < 1.e-10)
        
    def test_tensor_rotation(self):
        for i in range(50):
            d = randomRotationMatrix()
            dc = Utility.symmetricTensorRotationMatrix(d)
            t = SymmetricTensor(randomArray((6,)))
            tc = t.array
            tf = t.array2d
            t_rot = t.applyRotationMatrix(d)
            tc_rot = N.dot(dc, tc)
            tf_rot = N.dot(N.dot(d, tf), N.transpose(d))
            diff = SymmetricTensor(tc_rot).array2d-tf_rot
            self.assert_(largestAbsoluteElement(diff) < 1.e-14)
            diff = (SymmetricTensor(tc_rot)-t_rot).array
            self.assert_(largestAbsoluteElement(diff) < 1.e-14)

    def test_basis(self):
        for params, rb, sg, nb in datasets:
            cell = UnitCell(*params)
            sg = space_groups[sg]
            trs = cell.cartesianCoordinateSymmetryTransformations(sg)
            basis = symmetricTensorBasis(cell, sg)
            assert len(basis) == nb
            for t in basis:
                for tr in trs:
                    t_rot = t.applyRotationMatrix(tr.tensor.array)
                    assert largestAbsoluteElement((t_rot-t).array) < 1.e-12

def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(UtilityTests))
    return s

if __name__ == '__main__':
    unittest.main()
