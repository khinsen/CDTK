# Electron density maps and Patterson maps
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
Electron density maps and Patterson maps

.. moduleauthor:: Konrad Hinsen <konrad.hinsen@cnrs-orleans.fr>

"""

from CDTK import Units
from CDTK.Utility import SymmetricTensor, delta
from Scientific.Geometry import Vector, isVector
from Scientific import N, LA

class Map(object):

    """
    Map base class
    """

    def __init__(self, cell, n1, n2, n3):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        """
        self.cell = cell
        self.array = N.zeros((n1, n2, n3), N.Float)
        self.shape = (n1, n2, n3)
        self.x1 = N.arange(n1)/float(n1)-0.5
        self.x2 = N.arange(n2)/float(n2)-0.5
        self.x3 = N.arange(n3)/float(n3)-0.5
        e1, e2, e3 = self.cell.basisVectors()
        self.vmd_origin = -0.5*(e1+e2+e3)

    def makePositive(self):
        """
        Subtract the smallest map value from all other map values,
        such that the smallest value becomes 0.
        """
        smallest = N.minimum.reduce(N.ravel(self.array))
        self.array -= smallest

    def writeToVMDScript(self, filename, label=None):
        """
        :param filename: the name of the generated VMD script
        :type filename: string
        :param label: the label of the map as displayed by VMD
        :type label: string
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

    @classmethod
    def fromUnitCellAtoms(cls, cell, n1, n2, n3, atom_iterator):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        :param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        :type atom_iterator: iterable
        """
        obj = cls(cell, n1, n2, n3)
        obj.calculateFromUnitCellAtoms(atom_iterator, None)
        return obj

    @classmethod
    def fromUniverse(cls, cell, n1, n2, n3, universe, adps, conf=None):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        :param universe: a periodic MMTK universe
        :type universe: MMTK.Periodic3DUniverse
        :param adps: the anisotropic displacement parameters for all atoms
        :type adps: MMTK.ParticleTensor
        :param conf: a configuration for the universe, defaults to the
                     current configuration
        :type conf: MMTK.Configuration
        """
        obj = cls(cell, n1, n2, n3)
        obj.calculateFromUniverse(universe, adps, conf)
        return obj

    @classmethod
    def fromAsymmetricUnitAtoms(cls, cell, n1, n2, n3,
                                atom_iterator, space_group):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        :param atom_iterator: an iterator or sequence that yields
                              for each atom in the asymmetric unit a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        :type atom_iterator: iterable
        :param space_group: the space group of the crystal
        :type space_group: CDTK.SpaceGroups.SpaceGroup
        """
        obj = cls(cell, n1, n2, n3)
        obj.calculateFromAsymmetricUnitAtoms(atom_iterator, space_group, None)
        return obj

    @classmethod
    def fromStructureFactor(cls, cell, n1, n2, n3, sf):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        :param sf: a structure factor set
        :type sf: CDTK.Reflections.StructureFactor
        """
        obj = cls(cell, n1, n2, n3)
        obj.calculateFromStructureFactor(sf)
        return obj

    @classmethod
    def fromIntensities(cls, cell, n1, n2, n3, intensities):
        """
        :param cell: the unit cell for which the map is defined
        :type cell: CDTK.Crystal.UnitCell
        :param n1: the number of points in the grid along the
                   first lattice vector
        :type n1: int
        :param n2: the number of points in the grid along the
                   second lattice vector
        :type n2: int
        :param n3: the number of points in the grid along the
                   third lattice vector
        :type n3: int
        :param intensities: a set of reflection intensities
        :type intensities: CDTK.Reflections.IntensityData
        """
        obj = cls(cell, n1, n2, n3)
        obj.calculateFromIntensities(intensities)
        
    def calculateFromAsymmetricUnitAtoms(self, atom_iterator, space_group,
                                         cell=None):
        """
        :param atom_iterator: an iterator or sequence that yields
                              for each atom in the asymmetric unit a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        :param space_group: the space group of the crystal
        :type space_group: CDTK.SpaceGroups.SpaceGroup
        :type atom_iterator: iterable
        :param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        :type cell: CDTK.Crystal.UnitCell
        """
        if cell is None:
            cell = self.cell
        st = cell.cartesianCoordinateSymmetryTransformations(space_group)
        def check(adp):
            # Only isotropic B factors are handled for now
            if not isinstance(adp, float):
                raise NotImplementedError("Symmetry transformation of ADPs"
                                          " not yet implemented")
            return adp
        it = ((atom_id, element, tr(position), check(adp), occupancy)
              for atom_id, element, position, adp, occupancy in atom_iterator
              for tr in st)
        self.calculateFromUnitCellAtoms(it, cell)

    def calculateFromUniverse(self, universe, adps, conf=None):
        """
        :param universe: a periodic MMTK universe
        :type universe: MMTK.Periodic3DUniverse
        :param adps: the anisotropic displacement parameters for all atoms
        :type adps: MMTK.ParticleTensor
        :param conf: a configuration for the universe, defaults to the
                     current configuration
        :type conf: MMTK.Configuration
        """
        if conf is None:
            conf = universe.configuration()
        cell = universe.__class__()
        cell.setCellParameters(conf.cell_parameters)
        self.calculateFromUnitCellAtoms(((atom, atom.symbol, conf[atom],
                                          adps[atom], 1.)
                                         for atom in universe.atomList()),
                                        cell)


class ElectronDensityMap(Map):

    """
    Electron density map

    An electron density map can be calculated from a StructureFactor
    by Fourier transform or directly from an atomic model.
    """

    default_label = "Electron density"

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        """
        :param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        :type atom_iterator: iterable
        :param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        :type cell: CDTK.Crystal.UnitCell
        """
        if cell is None:
            cell = self.cell
        m_fc = cell.fractionalToCartesianMatrix()
        from AtomicScatteringFactors import atomic_scattering_factors
        for atom_id, element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
            bdiv = b / (2.*N.pi**2)
            xa = cell.cartesianToFractional(position)+0.5
            xa -= N.floor(xa)+0.5
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
                    sigma = SymmetricTensor(adp) + bdiv[i]*delta
                sigma_inv = sigma.inverse()
                weight = a[i] * N.sqrt(sigma_inv.determinant()) * occupancy
                m = -0.5*N.dot(N.transpose(m_fc),
                               N.dot(sigma_inv.array2d, m_fc))
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

    def calculateFromStructureFactor(self, sf):
        """
        :param sf: a structure factor set
        :type sf: CDTK.Reflections.StructureFactor
        """
        from CDTK_sf_fft import reflections_to_map
        from CDTK.ReflectionData import StructureFactor
        if not isinstance(sf, StructureFactor):
            raise TypeError("%s is not a StructureFactor instance" % str(sf))
        m_cf = self.cell.cartesianToFractionalMatrix()
        det_m_cf = LA.determinant(m_cf)
        n1, n2, n3 = self.shape
        self.array += reflections_to_map(sf, n1, n2, n3, det_m_cf)


class SolventMap(Map):

    """
    Solvent map

    A solvent map is 0 in the areas occupied by explicitly
    modelled atoms and 1 outside, with a smooth transition between
    the two zones.

    The solvent model is the polynomial switch model described
    in Fenn et al., Acta Cryst. D66, 1024-1031 (2010)
    """

    default_label = "Solvent"

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        """
        :param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        :type atom_iterator: iterable
        :param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        :type cell: CDTK.Crystal.UnitCell
        """
        if cell is None:
            cell = self.cell
        from ChemicalData import vdW_radii
        m_fc = cell.fractionalToCartesianMatrix()
        m = N.dot(N.transpose(m_fc), m_fc)
        w = 0.08 # window width
        self.array[...] = 1.
        for atom_id, element, position, adp, occupancy in atom_iterator:
            a = vdW_radii[element.lower()]

            xa = cell.cartesianToFractional(position)+0.5
            xa -= N.floor(xa)+0.5
            dx1 = self.x1-xa[0]
            dx1 += (dx1 < -0.5).astype(N.Int) - (dx1 >= 0.5).astype(N.Int)
            dx2 = self.x2-xa[1]
            dx2 += (dx2 < -0.5).astype(N.Int) - (dx2 >= 0.5).astype(N.Int)
            dx3 = self.x3-xa[2]
            dx3 += (dx3 < -0.5).astype(N.Int) - (dx3 >= 0.5).astype(N.Int)

            rsq = N.zeros(self.shape, N.Float)
            N.add(rsq, m[0,0]*(dx1*dx1)[:, N.NewAxis, N.NewAxis], rsq)
            N.add(rsq, m[1,1]*(dx2*dx2)[N.NewAxis, :, N.NewAxis], rsq)
            N.add(rsq, m[2,2]*(dx3*dx3)[N.NewAxis, N.NewAxis, :], rsq)
            N.add(rsq, (2.*m[0, 1]) *
                  dx1[:, N.NewAxis, N.NewAxis]*dx2[N.NewAxis, :, N.NewAxis],
                  rsq)
            N.add(rsq, (2.*m[0, 2]) *
                  dx1[:, N.NewAxis, N.NewAxis]*dx3[N.NewAxis, N.NewAxis, :],
                  rsq)
            N.add(rsq, (2.*m[1, 2]) *
                  dx2[N.NewAxis, :, N.NewAxis]*dx3[N.NewAxis, N.NewAxis, :],
                  rsq)
            dw = (N.sqrt(rsq)-a+w)/w
            rho = N.where(dw <= 0., 0.,
                          N.where(dw >= 2., 1., (0.75-0.25*dw)*dw*dw))
            self.array *= rho


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
        :param intensities: a set of reflection intensities
        :type intensities: CDTK.Reflections.IntensityData
        """
        from CDTK_sf_fft import reflections_to_map
        from CDTK.ReflectionData import IntensityData
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

    def calculateFromStructureFactor(self, sf):
        """
        :param sf: a structure factor set
        :type sf: CDTK.Reflections.StructureFactor
        """
        return self.calculateFromIntensities(self, sf.intensities())
