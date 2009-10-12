# Description of a crystal
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Description of a crystal
"""

from CDTK import Units
from CDTK.SpaceGroups import space_groups
from CDTK.Utility import SymmetricTensor
from Scientific.Geometry import Vector, isVector
from Scientific.Geometry.Transformation import Rotation, Translation, Shear
from Scientific import N, LA
import copy

class UnitCell(object):

    """
    Unit cell
    """

    def __init__(self, *parameters):
        """
        @param parameters: one of 1) three lattice vectors or
            2) six numbers: the lengths of the three lattice vectors (a, b, c)
            followed by the three angles (alpha, beta, gamma).
        """
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
        """
        @return: a list containing the three lattice vectors
        @rtype: C{list} of C{Scientific.Geometry.Vector}
        """
        return self.basis

    def reciprocalBasisVectors(self):
        """
        @return: a list containing the three basis vectors
                 of the reciprocal lattice
        @rtype: C{list} of C{Scientific.Geometry.Vector}
        """
        return self.reciprocal_basis

    def cellVolume(self):
        """
        @return: the volume of the unit cell
        @rtype: C{float}
        """
        e1, e2, e3 = self.basis
        return e1*e2.cross(e3)

    def cartesianToFractional(self, vector):
        """
        @param vector: a vector in real Cartesian space
        @type vector: C{Scientific.Geometry.Vector}
        @return: the vector in fractional coordinates
        @rtype: C{Scientific.N.array_type}
        """
        r1, r2, r3 = self.reciprocal_basis
        return N.array([r1*vector, r2*vector, r3*vector])

    def cartesianToFractionalMatrix(self):
        """
        @return: the 3x3 conversion matrix from real Cartesian space
                 coordinates to fractional coordinates
        """
        return N.array(self.reciprocal_basis)

    def fractionalToCartesian(self, array):
        """
        @param array: a vector in fractional coordinates
        @type array: C{Scientific.N.array_type}
        @return: the vector in real Cartesian space
        @rtype: C{Scientific.Geometry.Vector}
        """
        e1, e2, e3 = self.basis
        return array[0]*e1 + array[1]*e2 + array[2]*e3

    def fractionalToCartesianMatrix(self):
        """
        @return: the 3x3 conversion matrix from fractional
                 coordinates to real Cartesian space coordinates
        """
        return N.transpose(self.basis)

    def minimumImageDistanceVector(self, point1, point2):
        """
        @param point1: a point in the unit cell
        @type point1: C{Scientific.Geometry.Vector}
        @param point2: a point in the unit cell
        @type point2: C{Scientific.Geometry.Vector}
        @return: the minimum-image vector from point1 to point2
        @rtype: C{Scientific.Geometry.Vector}
        """
        d = self.cartesianToFractional(point2-point1)
        d = d - (d > 0.5) + (d <= -0.5)
        return self.fractionalToCartesian(d)
        
    def isCompatibleWith(self, other_cell, precision=1.e-5):
        """
        @param other_cell: a unit cell
        @type other_cell: L{UnitCell} or C{MMTK.Universe}
        @param precision: the absolute precision of the comparison
        @type precision: C{float}
        @return: C{True} if the lattice vectors of the two unit cells differ
                 by a vector of length < precision
        @rtype: C{bool}
        """
        other_basis = other_cell.basisVectors()
        for i in range(3):
            if (other_basis[i]-self.basis[i]).length() > precision:
                return False
        return True

    def cartesianCoordinateSymmetryOperations(self, space_group):
        """
        @param space_group: a space group
        @type space_group: L{CDTK.SpaceGroups.SpaceGroup}
        @return: a list of transformation objects representing the symmetry
                 operations of the space group in the Cartesian coordinates
                 of the unit cell
        @rtype: C{list} of C{Scientific.Geometry.Transformation.Transformation}
        """
        transformations = []
        to_fract = Shear(self.cartesianToFractionalMatrix())
        from_fract = Shear(self.fractionalToCartesianMatrix())
        for rot, trans_num, trans_den in space_group.transformations:
            trans = Vector((1.*trans_num)/trans_den)
            tr_fract = Translation(trans)*Rotation(rot)
            transformations.append(from_fract*tr_fract*to_fract)
        return transformations


class Atom(object):

    """
    An Atom object stores the relevant data about an atom for
    the purpose of calculating scattering factors and maps:
    the chemical element, the position, the fluctuation
    parameter (scalar or tensor), and the occupancy. It also
    stores an atom_id attribute whose value is not used in the
    calculations. It can be used as a unique atom identifier to
    establish a link to other atomic models.
    """

    def __init__(self, atom_id, element, position, fluctuation, occupancy):
        """
        @param atom_id: not used in any calculation. This value can be used
                        as a unique atom identifier to establish a link
                        to other atomic models.
        @param element: the chemical element
        @type element: C{str}
        @param position: the position vector
        @type position: C{Scientific.Geometry.Vector}
        @param fluctuation: the fluctuation parameter (B factor)
        @type fluctuation: C{float} or L{CDTK.Utility.SymmetricTensor}
        @param occupancy: the occupancy
        @type occupancy: C{float}
        """
        self.atom_id = atom_id
        self.element = element
        self.position = position
        self.fluctuation = fluctuation
        self.occupancy = occupancy


class Crystal(object):

    """
    A Crystal object defines a unit cell, a space group, and a list
    of atoms in the unit cell. Its iterator interface permits it
    to be used directly as the asymmetric unit iterator in the calculation
    of model amplitudes, maps, and in the definition of refinement
    engines.
    """

    def __init__(self, cell, space_group):
        """
        @param cell: the unit cell
        @type cell: L{CDTK.Crystal.UnitCell}
        @param space_group: the space group
        @type space_group: L{CDTK.SpaceGroups.SpaceGroup}
        """
        self.cell = cell
        self.space_group = space_group
        self.atoms = []

    def __len__(self):
        """
        @return: the number of atoms in the asymmetric unit
        @rtype: C{int}
        """
        return len(self.atoms)

    def __iter__(self):
        """
        @return: a generator yielding the atoms in the asymmetric unit
        @rtype: generator
        """
        for a in self.atoms:
            yield (a.atom_id, a.element, a.position, a.fluctuation, a.occupancy)

    def updateAtomParametersFromRefinementEngine(self, refinement_engine):
        for a in self.atoms:
            a.position = refinement_engine.getPosition(a.atom_id)
            a.fluctuation = refinement_engine.getADP(a.atom_id)
            a.occupancy = refinement_engine.getOccupancy(a.atom_id)

    def countAtomsByElement(self):
        """
        @return: a dictionary mapping chemical element symbols
                 to the number of atoms of that element in the
                 crystal. Note that the number of atoms can be
                 non-integer because the occupancies don't
                 necessarily add up to integer values.
        @rtype: C{dict}
        """
        counts = {}
        for atom_id, element, position, fluctuation, occupancy in self:
            counts[element] = counts.get(element, 0.) + occupancy
        for element in counts.keys():
            n_float = counts[element]
            n_int = int(n_float)
            if float(n_int) == n_float:
                counts[element] = n_int
            else:
                counts[element] = n_float
        return counts

    def atomsWithCorrectedFluctuations(self):
        """
        Correct fluctuations using two rules: (1) If there is an
        anisotropic fluctuation tensor with at least one negative
        eigenvalue, but with a positive trace, replace the negative
        eigenvalues by zero. If the trace is negative, set the fluctuation
        tensor to zero. (2) If there is a scalar fluctuation that is
        negative, replace it by zero.
        
        @return: a generator yielding the atoms in the asymmetric unit
                 after correction of the fluctuation tensor
        @rtype: generator
        """
        for a in self.atoms:
            adp = a.fluctuation
            if adp is None:
                adp = 0.
            if not isinstance(adp, float):
                if not adp.isPositiveDefinite():
                    fixed = adp.makeDefinite()
                    if fixed.isPositiveDefinite():
                        adp = fixed
                    else:
                        adp = adp.trace()
            if isinstance(adp, float) and adp < 0.:
                adp = 0.
            yield (a.atom_id, a.element, a.position, adp, a.occupancy)

    def atomsWithNegativeFluctuations(self):
        """
        @return: a generator yielding the atoms in the asymmetric unit
                 whose fluctuation parameter has negative eigenvalues
        @rtype: generator
        """
        for a in self.atoms:
            adp = a.fluctuation
            if adp is not None and \
               ((isinstance(adp, float) and adp < 0.)
                or ((not isinstance(adp, float))
                    and N.logical_or.reduce(adp.eigenvalues() < 0.))):
                yield (a.atom_id, a.element, a.position,
                       a.fluctuation, a.occupancy)
        

class PDBCrystal(Crystal):

    """
    A PDBCrystal object is a Crystal object generated from a PDB
    file.
    """

    def __init__(self, file_or_filename,
                 peptide_chains=True, nucleotide_chains=True,
                 residue_filter = None):
        """
        @param file_or_filename: the name of a PDB file, or a file object
        @type file_or_filename: C{str} or C{file}
        @param peptide_chains: if True, include the peptide chains from
                               the PDB file, otherwise discard them
        @type peptide_chains: C{Boolean}
        @param nucleotide_chains: if True, include the nucleotide chains from
                                  the PDB file, otherwise discard them
        @type nucleotide_chains: C{Boolean}
        @param residue_filter: a function called for each non-peptide and
                               non-nucleotide residue with the residue name
                               and residue number as argument. If it returns
                               True, the residue is included, otherwise it
                               is discarded. If no residue_filter is given,
                               all residues are includes.
        """
        from Scientific.IO.PDB import Structure
        s = Structure(file_or_filename)
        self.pdb_structure = s
        self.atom_dict = {}

        cell = UnitCell(s.a*Units.Ang, s.b*Units.Ang, s.c*Units.Ang,
                        s.alpha*Units.deg, s.beta*Units.deg, s.gamma*Units.deg)
        Crystal.__init__(self, cell, space_groups[s.space_group])

        remaining_residues = [r for r in s.residues]
        for residue in remaining_residues:
            residue.chain_id = ''
        residues = []
        for chain_list, flag in [(s.peptide_chains, peptide_chains),
                                 (s.nucleotide_chains, nucleotide_chains)]:
            for chain in chain_list:
                for residue in chain:
                    residue.chain_id = chain.chain_id
                    remaining_residues.remove(residue)
                    if flag:
                        residues.append(residue)
        if residue_filter is None:
            residues.extend(remaining_residues)
        else:
            for residue in remaining_residues:
                if residue_filter(residue.name, residue.number):
                    residues.append(residue)

        for residue in residues:
            for atom in residue:
                fluctuation = atom['temperature_factor'] \
                              * Units.Ang**2/(8.*N.pi**2)
                try:
                    fluctuation = SymmetricTensor(atom['u']*Units.Ang**2)
                except KeyError:
                    pass
                a = Atom(atom, atom['element'], atom['position']*Units.Ang,
                         fluctuation, atom['occupancy'])
                self.atoms.append(a)
                self.atom_dict[atom] = a

    def writeToFile(self, filename):
        from Scientific.IO.PDB import PDBFile
        pdb = PDBFile(filename, 'w')
        pdb.writeLine('CRYST1',
                      {'a': self.cell.a/Units.Ang,
                       'b': self.cell.b/Units.Ang,
                       'c': self.cell.c/Units.Ang,
                       'alpha': self.cell.alpha/Units.deg,
                       'beta': self.cell.beta/Units.deg,
                       'gamma': self.cell.gamma/Units.deg,
                       'space_group': self.space_group.symbol,
                       'z': len(self.space_group),
                       })
        m = self.cell.cartesianToFractionalMatrix()*Units.Ang
        m = N.where(N.fabs(m) >= 1.e-6, m, 0.)
        pdb.writeLine('SCALE1',
                      {'s1': m[0, 0],
                       's2': m[0, 1],
                       's3': m[0, 2],
                       'u': 0.,
                       })
        pdb.writeLine('SCALE2',
                      {'s1': m[1, 0],
                       's2': m[1, 1],
                       's3': m[1, 2],
                       'u': 0.,
                       })
        pdb.writeLine('SCALE3',
                      {'s1': m[2, 0],
                       's2': m[2, 1],
                       's3': m[2, 2],
                       'u': 0.,
                       })

        for residue in self.pdb_structure.residues:
            for atom in residue:
                u = self.atom_dict[atom].fluctuation
                if isinstance(u, SymmetricTensor):
                    b = u.trace()*(8.*N.pi**2/3.)
                else:
                    b = u*(8.*N.pi**2)
                    u = None
                pdb.writeLine('ATOM',
                              {'position': self.atom_dict[atom].position \
                                             / Units.Ang,
                               'occupancy': self.atom_dict[atom].occupancy,
                               'temperature_factor': b/Units.Ang**2,
                               'element': atom['element'],
                               'serial_number': atom['serial_number'],
                               'name': atom.name,
                               'residue_name': residue.name,
                               'residue_number': residue.number,
                               'chain_id': residue.chain_id,
                               })
                if u is not None:
                    pdb.writeLine('ANISOU',
                                  {'u': u.array2d/Units.Ang**2,
                                    'element': atom['element'],
                                   'serial_number': atom['serial_number'],
                                   'name': atom.name,
                                   'residue_name': residue.name,
                                   'residue_number': residue.number,
                                   'chain_id': residue.chain_id,
                                   })

        pdb.close()
