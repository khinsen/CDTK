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

import CDTK
from CDTK import Units
from CDTK.Utility import SymmetricTensor, delta, \
                         cartesianCoordinateSymmetryTransformations
from Scientific.Geometry import Vector, isVector
import numpy as np
import numpy.linalg as la

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
        self.array = np.zeros((n1, n2, n3), np.float)
        self.shape = (n1, n2, n3)
        self.x1 = np.arange(n1)/float(n1)-0.5
        self.x2 = np.arange(n2)/float(n2)-0.5
        self.x3 = np.arange(n3)/float(n3)-0.5
        e1, e2, e3 = self.cell.basisVectors()
        self.volume_origin = -0.5*(e1+e2+e3)

    def makePositive(self):
        """
        Subtract the smallest map value from all other map values,
        such that the smallest value becomes 0.
        """
        smallest = self.array.min()
        self.array -= smallest

    def writeMRC(self, filename, labels=None):
        """
        Write the map to a file in MRC format.
        :param filename: the name of the file
        :type filename: str
        :param labels: a list of max. 10 labels of max. 80 characters
        :type labels: sequence
        """
        volume = self.array
        dtype = self._closestMRCType(volume.dtype)
        with file(filename, 'wb') as f:
            f.write(self._MRCHeader(volume, dtype, labels))
            for k in range(volume.shape[2]):
                for j in range(volume.shape[1]):
                    line = volume[:, j, k].astype(dtype)
                    f.write(line.tostring())

    def _closestMRCType(self, dtype):
        if dtype in (np.float32, np.float64, np.float,
                     np.int32, np.int, np.uint32, np.uint, np.uint16):
            return np.float32
        elif dtype in (np.int16, np.uint8):
            return np.int16
        elif dtype in (np.int8, np.int0, np.character):
            return np.int8
        else:
            raise TypeError('Volume data of unknown type %s' % dtype)

    def _MRCHeader(self, volume, map_dtype, labels=None):

        cell_size = (self.cell.a/Units.Ang,
                     self.cell.b/Units.Ang,
                     self.cell.c/Units.Ang)
        cell_angles = (self.cell.alpha/Units.deg,
                       self.cell.beta/Units.deg,
                       self.cell.gamma/Units.deg,)

        if labels is None:
            import time
            labels = ["Created by CDTK " + CDTK.__version__,
                      time.asctime()]

        nlabels = len(labels)
        labels = labels + (10-nlabels) * [""]
        labels = [l + (80-len(l))*'\0' for l in labels]

        def bytes(values, dtype):
            return np.array(values, dtype).tostring()

        return ''.join([
            bytes(volume.shape, np.int32),  # nc, nr, ns
            bytes({np.int8: 0, np.int16: 1, np.float32: 2}[map_dtype],
                  np.int32),  # mode
            bytes((0, 0, 0), np.int32), # ncstart, nrstart, nsstart
            bytes(volume.shape, np.int32),  # nx, ny, nz
            bytes(cell_size, np.float32), # x_length, y_length, z_length
            bytes(cell_angles, np.float32), # alpha, beta, gamma
            bytes((1, 2, 3), np.int32), # mapc, mapr, maps
            bytes((volume.min(), volume.max(), volume.mean()),
                  np.float32), # dmin, dmax, dmean
            bytes(0, np.int32), # ispg
            bytes(0, np.int32), # nsymbt
            bytes([0]*25, np.int32), # extra
            bytes(self.volume_origin/Units.Ang,
                  np.float32), # origin (MRC extension to CCP4)
            'MAP ', # map
            bytes(0x00004144 if np.little_endian else 0x11110000,
                  np.int32), # machst
            bytes(np.sqrt(volume.var()), np.float32), # rms
            bytes(nlabels, np.int32), # nlabl
            ''.join(labels), # label
        ])

    def writeToVMDScript(self, filename, label=None):
        """
        :param filename: the name of the generated VMD script
        :type filename: string
        :param label: the label of the map as displayed by VMD
        :type label: string
        """
        if label is None:
            label = self.default_label
        factor = 1./self.array.max()
        vmd_script = file(filename, 'w')
        vmd_script.write('mol new\n')
        vmd_script.write('mol volume top "%s" \\\n' % label)
        e1, e2, e3 = self.cell.basisVectors()
        vmd_script.write('  {%f %f %f} \\\n' % tuple(self.volume_origin/Units.Ang))
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
        st = cartesianCoordinateSymmetryTransformations(cell, space_group)
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
            bdiv = b / (2.*np.pi**2)
            xa = cell.cartesianToFractional(position)+0.5
            xa -= np.floor(xa)+0.5
            dx1 = self.x1-xa[0]
            dx1 += (dx1 < -0.5).astype(np.int) - (dx1 >= 0.5).astype(np.int)
            dx2 = self.x2-xa[1]
            dx2 += (dx2 < -0.5).astype(np.int) - (dx2 >= 0.5).astype(np.int)
            dx3 = self.x3-xa[2]
            dx3 += (dx3 < -0.5).astype(np.int) - (dx3 >= 0.5).astype(np.int)
            for i in range(5):
                if isinstance(adp, float):
                    sigma = (adp + bdiv[i])*delta
                else:
                    sigma = SymmetricTensor(adp) + bdiv[i]*delta
                sigma_inv = sigma.inverse()
                weight = a[i] * np.sqrt(sigma_inv.determinant()) * occupancy
                m = -0.5*np.dot(m_fc.T, np.dot(sigma_inv.array2d, m_fc))
                e = np.zeros(self.shape, np.float)
                np.add(e, m[0, 0]*(dx1*dx1)[:, np.newaxis, np.newaxis], e)
                np.add(e, m[1, 1]*(dx2*dx2)[np.newaxis, :, np.newaxis], e)
                np.add(e, m[2, 2]*(dx3*dx3)[np.newaxis, np.newaxis, :], e)
                np.add(e,
                       (2.*m[0, 1]) *
                         dx1[:, np.newaxis, np.newaxis] *
                         dx2[np.newaxis, :, np.newaxis], e)
                np.add(e,
                       (2.*m[0, 2]) *
                         dx1[:, np.newaxis, np.newaxis] *
                         dx3[np.newaxis, np.newaxis, :], e)
                np.add(e,
                       (2.*m[1, 2]) *
                         dx2[np.newaxis, :, np.newaxis] *
                         dx3[np.newaxis, np.newaxis, :], e)
                np.add(self.array, weight*np.exp(e), self.array)

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
        det_m_cf = la.det(m_cf)
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
        m = np.dot(m_fc.T, m_fc)
        w = 0.08 # window width
        self.array[...] = 1.
        for atom_id, element, position, adp, occupancy in atom_iterator:
            a = vdW_radii[element.lower()]

            xa = cell.cartesianToFractional(position)+0.5
            xa -= np.floor(xa)+0.5
            dx1 = self.x1-xa[0]
            dx1 += (dx1 < -0.5).astype(np.int) - (dx1 >= 0.5).astype(np.int)
            dx2 = self.x2-xa[1]
            dx2 += (dx2 < -0.5).astype(np.int) - (dx2 >= 0.5).astype(np.int)
            dx3 = self.x3-xa[2]
            dx3 += (dx3 < -0.5).astype(np.int) - (dx3 >= 0.5).astype(np.int)

            rsq = np.zeros(self.shape, np.float)
            np.add(rsq, m[0,0]*(dx1*dx1)[:, np.newaxis, np.newaxis], rsq)
            np.add(rsq, m[1,1]*(dx2*dx2)[np.newaxis, :, np.newaxis], rsq)
            np.add(rsq, m[2,2]*(dx3*dx3)[np.newaxis, np.newaxis, :], rsq)
            np.add(rsq,
                   (2.*m[0, 1]) *
                     dx1[:, np.newaxis, np.newaxis] *
                     dx2[np.newaxis, :, np.newaxis],
                   rsq)
            np.add(rsq,
                   (2.*m[0, 2]) *
                     dx1[:, np.newaxis, np.newaxis] * 
                     dx3[np.newaxis, np.newaxis, :],
                   rsq)
            np.add(rsq,
                   (2.*m[1, 2]) *
                     dx2[np.newaxis, :, np.newaxis] *
                     dx3[np.newaxis, np.newaxis, :],
                   rsq)
            dw = (np.sqrt(rsq)-a+w)/w
            rho = np.where(dw <= 0., 0.,
                           np.where(dw >= 2., 1., (0.75-0.25*dw)*dw*dw))
            self.array *= rho

    # reimplementation that ignores ADPs
    def calculateFromAsymmetricUnitAtoms(self, atom_iterator, space_group,
                                         cell=None):
        if cell is None:
            cell = self.cell
        st = cartesianCoordinateSymmetryTransformations(cell, space_group)
        it = ((atom_id, element, tr(position), 0., occupancy)
              for atom_id, element, position, adp, occupancy in atom_iterator
              for tr in st)
        self.calculateFromUnitCellAtoms(it, cell)


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
        self.volume_origin = -0.5*(e1+e2+e3)

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
        det_m_cf = la.det(m_cf)
        n1, n2, n3 = self.shape
        array = reflections_to_map(intensities, n1, n2, n3, det_m_cf)
        array = np.concatenate([array[n1/2:, :, :], array[:n1/2, :, :]], axis=0)
        array = np.concatenate([array[:, n2/2:, :], array[:, :n2/2, :]], axis=1)
        array = np.concatenate([array[:, :, n3/2:], array[:, :, :n3/2]], axis=2)
        self.array += array

    def calculateFromStructureFactor(self, sf):
        """
        :param sf: a structure factor set
        :type sf: CDTK.Reflections.StructureFactor
        """
        return self.calculateFromIntensities(self, sf.intensities())
