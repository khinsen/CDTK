# Small utility functions used in various places
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

from Scientific import N

# ADP tensors are symmetric; only six of their nine elements are
# independent. In refinement applications, CDTK uses an internal
# representation consisting of a set of independent elements arranged
# in a 1-d array of length six. The order of elements is
# [xx, yy, zz, yz, xz, xy]. The following routines convert between
# this representation and the standard representation in terms of
# a 3x3 array.

def compactSymmetricTensor(t):
    """
    @param t: an array or shape (3,3) representing a symmetric tensor
    @type t: C{Scientific.N.array_type}
    @return: an array of shape (6,) representing the same tensor by
             its independent elements [xx, yy, zz, yz, xz, xy]
    @rtype: C{Scientific.N.array_type}
    """
    assert t.shape == (3, 3)
    assert largestAbsoluteElement(t-N.transpose(t)) == 0.
    return N.take(N.reshape(t, (9,)), [0, 4, 8, 5, 2, 1])

def fullSymmetricTensor(t):
    """
    @param t: an array or shape (6,) representing a symmetric tensor in
              compact storage
    @type t: C{Scientific.N.array_type}
    @return: an array of shape (3,3) representing the same tensor
    @rtype: C{Scientific.N.array_type}
    """
    assert t.shape == (6,)
    return N.reshape(N.take(t, [0, 5, 4, 5, 1, 3, 4, 3, 2]), (3, 3))


# Symmetry operations can be applied to ADP tensors in compact storage
# by a matrix multiplication with a "tensor rotation matrix" of shape (6,6).
# The following function constructs this matrix from a standard
# (3,3) rotation matrix.

def symmetricTensorRotationMatrix(d):
    """
    @param d: an array of shape (3,3) representing a rotation matrix
    @type d: C{Scientific.N.array_type}
    @return: an array of shape (6,6) representing the equivalent rotation
             matrix for a symmetric tensor in compact storage
    @rtype: C{Scientific.N.array_type}
    """
    assert d.shape == (3, 3)
    dd = d[:, :, N.NewAxis, N.NewAxis]*d[N.NewAxis, N.NewAxis, :, :]
    assert largestAbsoluteElement(dd[:, 0, :, 0]+dd[:, 1, :, 1]+dd[:, 2, :, 2]
                                  -N.identity(3, N.Float)) < 1.e-10
    dd = N.reshape(N.transpose(dd, [0, 2, 1, 3]), (9, 9))
    dd = N.take(dd, [0, 4, 8, 5, 2, 1], axis=0)
    dd = N.take(dd, [0, 4, 8, 5, 2, 1, 7, 6, 3], axis=1)
    dd[:, 3:6] += dd[:, 6:9]
    dd = dd[:, :6]
    return dd


# A utility function used in argument checking and unit tests.

def largestAbsoluteElement(a):
    """
    @param a: a numerical array
    @type a: C{Scientific.N.array_type}
    @return: the largest absolute value over all array elements
    @rtype: C{float}
    """
    return N.maximum.reduce(N.absolute(N.ravel(a)))
