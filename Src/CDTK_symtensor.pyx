# Symmetric tensor object
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#


# ADP tensors are symmetric; only six of their nine elements are
# independent. This module defines a representation of symmetric tensors
# consisting of a set of independent elements arranged in a 1-d array 
# of length six. The order of elements is [xx, yy, zz, yz, xz, xy].


include "numeric.pxi"

cdef extern from "math.h":

    double sqrt(double x)
    double cos(double x)
    double acos(double x)
    double M_PI

from Scientific import Geometry
from Scientific import N, LA

#
# For efficiency reasons (calling __init__ makes the creation of a tensor
# rather expensive), most of the operations happen in class
# "symmetric_tensor", which is not meant to be used directly in application
# code. Objects of this class are initialized with zero values and then
# initialized explicitly by calling the method "set".
# Application code should use the derived class "SymmetricTensor", which adds
# only the __init__ method.
#
cdef class symmetric_tensor:

    cdef double xx, yy, zz, yz, xz, xy

    property array:
        def __get__(self):
            return N.array([self.xx, self.yy, self.zz,
                            self.yz, self.xz, self.xy])

    property array2d:
        def __get__(self):
            return N.array([[self.xx, self.xy, self.xz],
                            [self.xy, self.yy, self.yz],
                            [self.xz, self.yz, self.zz]])

    # __array_priority__ and __array_wrap__ are needed to permit
    # multiplication with numpy scalar types.
    property __array_priority__:
        def __get__(self):
            return 10.0

    def __array_wrap__(self, array):
        result = symmetric_tensor()
        symmetric_tensor.set(result, array[0], array[1], array[2],
                             array[3], array[4], array[5])
        return result

    def __copy__(self, memo = None):
        return self

    def __deepcopy__(self, memo = None):
        return self

    def __getstate__(self):
        return (self.xx, self.yy, self.zz, self.yz, self.xz, self.xy)

    def __setstate__(self, state):
        self.xx, self.yy, self.zz, self.yz, self.xz, self.xy = state

    def __reduce__(self):
        return (SymmetricTensor, (self.xx, self.yy, self.zz,
                                  self.yz, self.xz, self.xy))

    cdef void set(self, double xx, double yy, double zz,
                        double yz, double xz, double xy):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.yz = yz
        self.xz = xz
        self.xy = xy

    def __repr__(self):
        return 'SymmetricTensor(%f,%f,%f,%f,%f,%f)' \
               % (self.xx, self.yy, self.zz, self.yz, self.xz, self.xy)

    def __str__(self):
        return repr(self)

    def __add__(symmetric_tensor self, symmetric_tensor other):
        result = symmetric_tensor()
        symmetric_tensor.set(result,
                             self.xx+other.xx, self.yy+other.yy,
                             self.zz+other.zz, self.yz+other.yz,
                             self.xz+other.xz, self.xy+other.xy)
        return result

    def __neg__(symmetric_tensor self):
        result = symmetric_tensor()
        symmetric_tensor.set(result, -self.xx, -self.yy, -self.zz,
                                     -self.yz, -self.xz, -self.xy)
        return result

    def __sub__(symmetric_tensor self, symmetric_tensor other):
        result = symmetric_tensor()
        symmetric_tensor.set(result,
                             self.xx-other.xx, self.yy-other.yy,
                             self.zz-other.zz, self.yz-other.yz,
                             self.xz-other.xz, self.xy-other.xy)
        return result

    def __mul__(x, y):
        if isinstance(y, symmetric_tensor):
            if Geometry.isVector(x):
                return Geometry.Vector(N.dot(x.array, y.array2d))
            else:
                return y.__smul(x)
        elif isinstance(x, symmetric_tensor):
            if Geometry.isVector(y):
                return Geometry.Vector(N.dot(x.array2d, y.array))
            else:
                return x.__smul(y)

    def __smul(self, double s):
        result = symmetric_tensor()
        symmetric_tensor.set(result, s*self.xx, s*self.yy, s*self.zz,
                                     s*self.yz, s*self.xz, s*self.xy)
        return result

    def __richcmp__(symmetric_tensor self, other, int op):
        cdef int eq
        if op != 2 and op != 3:
            return NotImplemented
        if isinstance(other, symmetric_tensor):
            eq = self.xx == other[0] and self.yy == other[1] \
                 and self.zz == other[2] and self.yz == other[3] \
                 and self.xz == other[4] and self.xy == other[5]
        else:
            eq = False
        if op == 2:
            return eq
        else:
            return not eq

    def __getitem__(self, int index):
        if index == 0 or index == -6:
            return self.xx
        elif index == 1 or index == -5:
            return self.yy
        elif index == 2 or index == -4:
            return self.zz
        elif index == 3 or index == -3:
            return self.yz
        elif index == 4 or index == -2:
            return self.xz
        elif index == 5 or index == -1:
            return self.xy
        raise IndexError

    def asTensor(self):
        "Returns an equivalent rank-2 tensor object"
        return Geometry.Tensor(N.array([[self.xx, self.xy, self.xz],
                                        [self.xy, self.yy, self.yz],
                                        [self.xz, self.yz, self.zz]]), 1)

    def trace(self):
        return self.xx + self.yy + self.zz

    def determinant(self):
        return self.xx*(self.yy*self.zz - self.yz*self.yz) \
               - self.xy*(self.xy*self.zz-self.xz*self.yz) \
               + self.xz*(self.xy*self.yz-self.yy*self.xz)

    def isPositiveDefinite(self):
        cdef double d1, d2, d3
        d1 = self.zz
        d2 = self.yy*self.zz - self.yz*self.yz
        d3 = self.xx*d2 \
             - self.xy*(self.xy*self.zz-self.xz*self.yz) \
             + self.xz*(self.xy*self.yz-self.yy*self.xz)
        return d1 > 0. and d2 > 0. and d3 > 0.

    def eigenvalues(self):
        cdef double b, c, d
        cdef double q, r, theta
        cdef double e1, e2, e3, temp
        b = self.xx + self.yy + self.zz
        c = self.xy*self.xy + self.xz*self.xz + self.yz*self.yz \
            - (self.xx*self.yy + self.xx*self.zz + self.yy*self.zz)
        d = self.xx*(self.yy*self.zz - self.yz*self.yz) \
            - self.xy*(self.xy*self.zz-self.xz*self.yz) \
            + self.xz*(self.xy*self.yz-self.yy*self.xz)
        q = sqrt(b*b+3.*c)/3.
        r = (9.*b*c + 27.*d + 2.*b*b*b)/54.
        if q*q*q-r <= 1.e-10*r:
            theta = 0.
        else:
            theta = acos(r/(q*q*q))
        e1 = 2.*q*cos(theta/3.)+b/3.
        e2 = 2.*q*cos((theta+2.*M_PI)/3.)+b/3.
        e3 = 2.*q*cos((theta+4.*M_PI)/3.)+b/3.
        if e2 < e1:
            temp = e2
            e2 = e1
            e1 = temp
        if e3 < e2:
            temp = e3
            e3 = e2
            e2 = temp
        if e2 < e1:
            temp = e2
            e2 = e1
            e1 = temp
        return N.array([e1, e2, e3])

    def makeDefinite(self):
        ev, rot = LA.eigenvectors(self.array2d)
        if N.sum(ev) >= 0.:
            a = N.dot(N.transpose(rot)*N.maximum(ev, 0.), rot)
        else:
            a = N.dot(N.transpose(rot)*N.minimum(ev, 0.), rot)
        return SymmetricTensor(a[0,0], a[1,1], a[2,2], a[1,2], a[0,2], a[0,1])

    def inverse(self):
        cdef double d, dinv
        d = self.xx*(self.yy*self.zz-self.yz*self.yz) \
            - self.xy*(self.xy*self.zz-self.xz*self.yz) \
            + self.xz*(self.xy*self.yz-self.yy*self.xz)
        dinv = 1./d
        result = symmetric_tensor()
        symmetric_tensor.set(result,
                             dinv*(self.yy*self.zz-self.yz*self.yz),
                             dinv*(self.xx*self.zz-self.xz*self.xz),
                             dinv*(self.xx*self.yy-self.xy*self.xy),
                             dinv*(self.xz*self.xy-self.xx*self.yz),
                             dinv*(self.xy*self.yz-self.xz*self.yy),
                             dinv*(self.xz*self.yz-self.xy*self.zz))
        return result
        
    def applyRotationMatrix(self, array_type d):
        cdef double *dp
        cdef double dxx, dxy, dxz
        cdef double dyx, dyy, dyz
        cdef double dzx, dzy, dzz
        cdef double xx, xy, xz
        cdef double yx, yy, yz
        cdef double zx, zy, zz

        assert d.nd == 2 and d.dimensions[0] == 3 and d.dimensions[1] == 3
        if PyArray_ISCONTIGUOUS(d):
            dp = <double *>d.data
            dxx = dp[0]
            dxy = dp[1]
            dxz = dp[2]
            dyx = dp[3]
            dyy = dp[4]
            dyz = dp[5]
            dzx = dp[6]
            dzy = dp[7]
            dzz = dp[8]
        else:
            dxx, dxy, dxz = d[0]
            dyx, dyy, dyz = d[1]
            dzx, dzy, dzz = d[2]

        xx = self.xx*dxx + self.xy*dxy + self.xz*dxz
        xy = self.xx*dyx + self.xy*dyy + self.xz*dyz
        xz = self.xx*dzx + self.xy*dzy + self.xz*dzz

        yx = self.xy*dxx + self.yy*dxy + self.yz*dxz
        yy = self.xy*dyx + self.yy*dyy + self.yz*dyz
        yz = self.xy*dzx + self.yy*dzy + self.yz*dzz

        zx = self.xz*dxx + self.yz*dxy + self.zz*dxz
        zy = self.xz*dyx + self.yz*dyy + self.zz*dyz
        zz = self.xz*dzx + self.yz*dzy + self.zz*dzz

        result = symmetric_tensor()
        symmetric_tensor.set(result,
                             dxx*xx + dxy*yx + dxz*zx,
                             dyx*xy + dyy*yy + dyz*zy,
                             dzx*xz + dzy*yz + dzz*zz,
                             dyx*xz + dyy*yz + dyz*zz,
                             dxx*xz + dxy*yz + dxz*zz,
                             dxx*xy + dxy*yy + dxz*zy)
        return result


cdef class SymmetricTensor(symmetric_tensor):

    property __safe_for_unpickling__:
        def __get__(self):
            return 1

    def __init__(self, xx=None, yy=None, zz=None, yz=None, xz=None, xy=None):
        cdef int rank
        if xx is None:
            pass
        elif yy is None and zz is None and yz is None \
                 and xz is None and xy is None:
            try:
                array = xx.array
            except AttributeError:
                array = N.array(xx)
            rank = len(array.shape)
            if rank == 0:
                v = float(array)  # assignment in two steps to work around
                self.xx = self.yy = self.zz = v # a Cython bug
                self.yz = self.xz = self.xy = 0.
            elif rank == 1:
                self.xx, self.yy, self.zz, self.yz, self.xz, self.xy = array
            elif rank == 2:
                assert array.shape == (3, 3)
                self.xx = array[0,0]
                self.yy = array[1,1]
                self.zz = array[2,2]
                self.yz = array[1,2]
                self.xz = array[0,2]
                self.xy = array[0,1]
                if self.yz != array[2, 1] or self.xz != array[2, 0] \
                       or self.xy != array[1, 0]:
                    raise ValueError("tensor not symmetric")
            else:
                raise ValueError("can't convert to symmetric tensor")
                
        else:
            self.xx = xx
            self.yy = yy
            self.zz = zz
            self.yz = yz
            self.xz = xz
            self.xy = xy



delta = SymmetricTensor(1.)
