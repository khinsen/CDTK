# Small utility functions used in various places
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

from Scientific import N, LA
from CDTK_symtensor import SymmetricTensor, delta

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

# Return the independent symmetric tensors (in compact storage) for a given
# space group and unit cell. Every tensor that is invariant under the rotations
# of the space group can be written as a linear superposition of these vectors.

def symmetricTensorBasis(cell, space_group):
    from CDTK.Crystal import UnitCell
    subspace = 1.*N.equal.outer(N.arange(6), N.arange(6))
    for tr in cell.cartesianCoordinateSymmetryOperations(space_group):
        rot = symmetricTensorRotationMatrix(tr.tensor.array)
        ev, axes = LA.eigenvectors(rot)
        new_subspace = []
        for i in range(6):
            if abs(ev[i]-1.) < 1.e-12:
                p = N.dot(N.transpose(subspace), N.dot(subspace, axes[i].real))
                new_subspace.append(p)
        m, s, subspace = LA.singular_value_decomposition(N.array(new_subspace))
        nb = N.sum(s/s[0] > 1.e-12)
        subspace = subspace[:nb]
    return [SymmetricTensor(a) for a in subspace]

# A utility function used in argument checking and unit tests.

def largestAbsoluteElement(a):
    """
    @param a: a numerical array
    @type a: C{Scientific.N.array_type}
    @return: the largest absolute value over all array elements
    @rtype: C{float}
    """
    return N.maximum.reduce(N.absolute(N.ravel(a)))
