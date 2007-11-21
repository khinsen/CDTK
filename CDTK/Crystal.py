from CDTK import Units
from Scientific.Geometry import Vector, isVector, delta
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


class Map(object):

    def __init__(self, cell, n1, n2, n3):
        self.cell = cell
        self.array = N.zeros((n1, n2, n3), N.Float)
        self.shape = (n1, n2, n3)
        self.x1 = N.arange(n1)/float(n1)
        self.x2 = N.arange(n2)/float(n2)
        self.x3 = N.arange(n3)/float(n3)
        self.vmd_origin = Vector(0., 0., 0.)

    def makePositive(self):
        smallest = N.minimum.reduce(N.ravel(self.array))
        self.array -= smallest

    def writeToVMDScript(self, filename, label=None):
        if label is None:
            label = self.default_label
        factor = 1./N.maximum.reduce(N.ravel(self.array))
        vmd_script = file(filename, 'w')
        vmd_script.write('mol new\n')
        vmd_script.write('mol volume top "%s" \\\n' % label)
        e1, e2, e3 = self.cell.basisVectors()
        vmd_script.write('  {%f %f %f} \\\n' % tuple(self.vmd_origin/Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple(e1/Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple(e2/Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple(e3/Units.Ang))
        vmd_script.write('  %d %d %d \\\n' % self.shape)
        vmd_script.write('  {')
        for iz in range(self.shape[2]):
            for iy in range(self.shape[1]):
                for ix in range(self.shape[0]):
                    vmd_script.write(str(factor*self.array[ix, iy, iz]) + ' ')
        vmd_script.write('}\n')
        vmd_script.write('mol addrep top\nmol modstyle 0 top isosurface\n')
        vmd_script.close()


class ElectronDensityMap(Map):

    default_label = "Electron density"

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        if cell is None:
            cell = self.cell
        m_fc = cell.fractionalToCartesianMatrix()
        from AtomicStructureFactors import atomic_structure_factors
        for element, position, adp, occupancy in atom_iterator:
            a, b = atomic_structure_factors[element.lower()]
            bdiv = b / (2.*N.pi**2)
            xa = cell.cartesianToFractional(position)
            xa = xa-N.floor(xa) # map to interval [0..1)
            dx1 = self.x1-xa[0]
            dx1 += (dx1 < -0.5).astype(N.Int) - (dx1 >= 0.5).astype(N.Int)
            dx2 = self.x2-xa[1]
            dx2 += (dx2 < -0.5).astype(N.Int) - (dx2 >= 0.5).astype(N.Int)
            dx3 = self.x3-xa[2]
            dx3 += (dx3 < -0.5).astype(N.Int) - (dx3 >= 0.5).astype(N.Int)
            for i in range(5):
                if isinstance(adp, float):
                    sigma = (adp + bdiv[i])*delta
                else:
                    sigma = adp + bdiv[i]*delta
                sigma_inv = LA.inverse(sigma.array)
                weight = a[i] * N.sqrt(LA.determinant(sigma_inv)) * occupancy
                m = -0.5*N.dot(N.transpose(m_fc), N.dot(sigma_inv, m_fc))
                e = N.zeros(self.shape, N.Float)
                N.add(e, m[0, 0]*(dx1*dx1)[:, N.NewAxis, N.NewAxis], e)
                N.add(e, m[1, 1]*(dx2*dx2)[N.NewAxis, :, N.NewAxis], e)
                N.add(e, m[2, 2]*(dx3*dx3)[N.NewAxis, N.NewAxis, :], e)
                N.add(e, (2.*m[0, 1]) *
                   dx1[:, N.NewAxis, N.NewAxis]*dx2[N.NewAxis, :, N.NewAxis], e)
                N.add(e, (2.*m[0, 2]) *
                   dx1[:, N.NewAxis, N.NewAxis]*dx3[N.NewAxis, N.NewAxis, :], e)
                N.add(e, (2.*m[1, 2]) *
                   dx2[N.NewAxis, :, N.NewAxis]*dx3[N.NewAxis, N.NewAxis, :], e)
                N.add(self.array, weight*N.exp(e), self.array)

    def calculateFromUniverse(self, universe, adps, conf=None):
        if conf is None:
            conf = universe.configuration()
        cell = universe.__class__()
        cell.setCellParameters(conf.cell_parameters)
        self.calculateFromUnitCellAtoms(((atom.symbol, conf[atom],
                                          adps[atom], 1.)
                                         for atom in universe.atomList()),
                                        cell)

    def calculateFromStructureFactor(self, sf):
        from CDTK_sf_fft import reflections_to_map
        from CDTK.Reflections import StructureFactor
        if not isinstance(sf, StructureFactor):
            raise TypeError("%s is not a StructureFactor instance" % str(sf))
        m_cf = self.cell.cartesianToFractionalMatrix()
        det_m_cf = LA.determinant(m_cf)
        n1, n2, n3 = self.shape
        self.array += reflections_to_map(sf, n1, n2, n3, det_m_cf)


class PattersonMap(Map):

    default_label = "Patterson map"

    def __init__(self, cell, n1, n2, n3):
        Map.__init__(self, cell, n1, n2, n3)
        e1, e2, e3 = self.cell.basisVectors()
        # display Patterson maps centered on (0, 0, 0) to
        # facilitate comparisons
        self.vmd_origin = -0.5*(e1+e2+e3)

    def calculateFromIntensities(self, intensities):
        from CDTK_sf_fft import reflections_to_map
        from CDTK.Reflections import IntensityData
        if not isinstance(intensities, IntensityData):
            raise TypeError("%s is not an IntensityData instance"
                            % str(intensities))
        m_cf = self.cell.cartesianToFractionalMatrix()
        det_m_cf = LA.determinant(m_cf)
        n1, n2, n3 = self.shape
        array = reflections_to_map(intensities, n1, n2, n3, det_m_cf)
        array = N.concatenate([array[n1/2:, :, :], array[:n1/2, :, :]], axis=0)
        array = N.concatenate([array[:, n2/2:, :], array[:, :n2/2, :]], axis=1)
        array = N.concatenate([array[:, :, n3/2:], array[:, :, :n3/2]], axis=2)
        self.array += array
