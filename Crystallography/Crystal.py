from Scientific.Geometry import Vector, isVector
from Scientific import N, LA

class UnitCell(object):

    def __init__(self, *parameters):
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
        return self.basis

    def reciprocalBasisVectors(self):
        return self.reciprocal_basis

    def cellVolume(self):
        e1, e2, e3 = self.basis
        return e1*e2.cross(e3)

    def cartesianToFractional(self, vector):
        r1, r2, r3 = self.reciprocal_basis
        return N.array([r1*vector, r2*vector, r3*vector])

    def cartesianToFractionalMatrix(self):
        return N.array(self.reciprocal_basis)

    def fractionalToCartesian(self, array):
        e1, e2, e3 = self.basis
        return array[0]*e1 + array[1]*e2 + array[2]*e3

    def fractionalToCartesianMatrix(self):
        return N.transpose(self.basis)

    def isCompatibleWith(self, other_cell, precision=1.e-5):
        other_basis = other_cell.basisVectors()
        for i in range(3):
            if (other_basis[i]-self.basis[i]).length() > precision:
                return False
        return True
