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
from CDTK.Utility import largestAbsoluteElement

from CDTK import Utility

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
        error = largestAbsoluteElement(Utility.compactSymmetricTensor(t)-tc)
        self.assert_(error == 0.)
        for i in range(50):
            t = Utility.fullSymmetricTensor(randomArray((6,)))
            tc = Utility.compactSymmetricTensor(t)
            error = largestAbsoluteElement(Utility.fullSymmetricTensor(tc)-t)
            self.assert_(error == 0.)
        for i in range(10):
            t = randomArray((3,3))
            self.assertRaises(AssertionError,
                              Utility.compactSymmetricTensor, t)

    def test_tensor_rotation(self):
        for i in range(50):
            d = randomRotationMatrix()
            dc = Utility.symmetricTensorRotationMatrix(d)
            t = Utility.fullSymmetricTensor(randomArray((6,)))
            tc = Utility.compactSymmetricTensor(t)
            tc_rot = N.dot(dc, tc)
            t_rot = N.dot(N.dot(d, t), N.transpose(d))
            diff = Utility.fullSymmetricTensor(tc_rot)-t_rot
            self.assert_(largestAbsoluteElement(diff) < 1.e-14)


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(UtilityTests))
    return s

if __name__ == '__main__':
    unittest.main()
