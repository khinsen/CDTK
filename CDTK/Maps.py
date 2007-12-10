# Electron density maps and Patterson maps
#

"""
Electron density maps and Patterson maps
"""

from CDTK import Units
from Scientific.Geometry import Vector, isVector
import Scientific.Geometry
from Scientific import N, LA

class Map(object):

    """
    Map base class
    """

    def __init__(self, cell, n1, n2, n3):
        """
        @param cell: the unit cell for which the map is defined
        @type cell: L{CDTK.Crystal.UnitCell}
        @param n1: the number of points in the grid along the
                   first lattice vector
        @type n1: C{int}
        @param n2: the number of points in the grid along the
                   second lattice vector
        @type n2: C{int}
        @param n3: the number of points in the grid along the
                   third lattice vector
        @type n3: C{int}
        """
        self.cell = cell
        self.array = N.zeros((n1, n2, n3), N.Float)
        self.shape = (n1, n2, n3)
        self.x1 = N.arange(n1)/float(n1)
        self.x2 = N.arange(n2)/float(n2)
        self.x3 = N.arange(n3)/float(n3)
        self.vmd_origin = Vector(0., 0., 0.)

    def makePositive(self):
        """
        Subtract the smallest map value from all other map values,
        such that the smallest value becomes 0.
        """
        smallest = N.minimum.reduce(N.ravel(self.array))
        self.array -= smallest

    def writeToVMDScript(self, filename, label=None):
        """
        @param filename: the name of the generated VMD script
        @type filename: C{string}
        @param label: the label of the map as displayed by VMD
        @type label: C{string}
        """
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

    """
    Electron density map

    An electron density map can be calculated from a StructureFactor
    by Fourier transform or directly from an atomic model.
    """

    default_label = "Electron density"

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        """
        @param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        @type atom_iterator: iterable
        @param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        @type cell: L{CDTK.Crystal.UnitCell}
        """
        if cell is None:
            cell = self.cell
        m_fc = cell.fractionalToCartesianMatrix()
        from AtomicScatteringFactors import atomic_scattering_factors
        for element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
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
                    sigma = (adp + bdiv[i])*Scientific.Geometry.delta
                else:
                    sigma = adp + bdiv[i]*Scientific.Geometry.delta
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
        """
        @param universe: a periodic MMTK universe
        @type universe: C{MMTK.Periodic3DUniverse}
        @param adps: the anisotropic displacement parameters for all atoms
        @type adps: C{MMTK.ParticleTensor}
        @param conf: a configuration for the universe, defaults to the
                     current configuration
        @type conf: C{MMTK.Configuration}
        """
        if conf is None:
            conf = universe.configuration()
        cell = universe.__class__()
        cell.setCellParameters(conf.cell_parameters)
        self.calculateFromUnitCellAtoms(((atom.symbol, conf[atom],
                                          adps[atom], 1.)
                                         for atom in universe.atomList()),
                                        cell)

    def calculateFromStructureFactor(self, sf):
        """
        @param sf: a structure factor set
        @type sf: L{CDTK.Reflections.StructureFactor}
        """
        from CDTK_sf_fft import reflections_to_map
        from CDTK.Reflections import StructureFactor
        if not isinstance(sf, StructureFactor):
            raise TypeError("%s is not a StructureFactor instance" % str(sf))
        m_cf = self.cell.cartesianToFractionalMatrix()
        det_m_cf = LA.determinant(m_cf)
        n1, n2, n3 = self.shape
        self.array += reflections_to_map(sf, n1, n2, n3, det_m_cf)


class PattersonMap(Map):

    """
    Patterson map

    A Patterson map can be calculated from reflection intensities.
    """

    default_label = "Patterson map"

    def __init__(self, cell, n1, n2, n3):
        Map.__init__(self, cell, n1, n2, n3)
        e1, e2, e3 = self.cell.basisVectors()
        # display Patterson maps centered on (0, 0, 0) to
        # facilitate comparisons
        self.vmd_origin = -0.5*(e1+e2+e3)

    def calculateFromIntensities(self, intensities):
        """
        @param intensities: a set of reflection intensities
        @type intensities: L{CDTK.Reflections.IntensityData}
        """
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
