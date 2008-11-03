# mmCIF parser
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

# The first stage of this parser (tokenizer) was inspired by the mmCIF parser
# in PyMMLIB (http://pymmlib.sourceforge.net/). The regular expression for
# identifying tokens was taken directly from there. The higher-level part of
# the parser is completely different, however. The PyMMLIB parser loads
# all the file contents into memory. This turned out to consume too much
# memory for large structure factor file. The parser in this module stores
# nothing at all. It is written as an iterator yielding mmCIF data items
# to the calling routine. The highest-level interface provides a filter
# for selecting data items and feeds tables to the calling program line
# by line through a callback object. This ensures that only the required
# data is stored, and directly in the form that the application will use.

"""
mmCIF parser

The functions in this module parse mmCIF files and return the contained
data items. There are four different routines depending on the level of
access that the application requires.
"""

from CDTK import Units
from Scientific.IO.TextFile import TextFile
from Scientific import N
import re

# Identifiers used in the return values of the parsers.
TOKEN = 'token'
DATA_LABEL = 'data_label'
DATA_VALUE = 'data_value'
KEYWORD = 'keyword'

LABEL_OR_KEYWORD = 'label_or_keyword'
VALUE = 'value'
LOOP_LABELS = 'loop_labels'
LOOP_VALUES = 'loop_values'

DATA = 'data'
TABLE_HEADER = 'table_header'
TABLE_DATA = 'table_data'

# A regular expression that encodes mmCIF syntax.
_token_regexp = re.compile(
    r"(?:"
     "(?:_(.+?)[.](\S+))"               "|"  # _section.subsection
     "(?:['\"](.*?)(?:['\"]\s|['\"]$))" "|"  # quoted strings
     "(?:\s*#.*$)"                      "|"  # comments
     "(\S+)"                                 # unquoted tokens
     ")")


# mmCIF-specific Exception classes
class MMCIFError(Exception):
    pass

class MMCIFSyntaxError(MMCIFError):

    def __init__(self, text, line_number=None):
        if line_number is None:
            # This is a workaround for the __setstate__ method in Exception
            # that effectively requires exceptions to accept a single
            # argument, otherwise a bug occurs during unpickling.
            MMCIFError.__init__(self, text)
            return
        self.line_number = line_number
        MMCIFError.__init__(self, "Line %d: %s" % (line_number, text))

#
# The parser object. It provides interfaces at different levels.
# 
class MMCIFParser(object):

    """
    Parser for mmCIF files

    The file to be parsed is specified when the parser object is created.
    One of the parser functions/generators may then be called to access
    the file contents.
    """
    
    def __init__(self, file_name=None, file_object=None,
                 pdb_code=None, pdb_sf_code=None):
        """
        Specify the mmCIF file to be loaded. Only one of the four
        keyword parameters may be given a value.
        @param file_name: the name of a file. Compressed or gzipped files
               can be handled directly.
        @type file_name: C{str}
        @param file_object: a file object
        @type file_object: C{file}
        @param pdb_code: the PDB code for a structure file, which is
               taken from a public or local PDB repository
               (see L{CDTK.PDBRepository}.
        @type pdb_code: C{str}
        @param pdb_sf_code: the PDB code for a structure factor file, which
               is taken from a public or local PDB repository
               (see L{CDTK.PDBRepository}.
        @type pdb_sf_code: C{str}
        """
        self.line_number = 0
        if file_name is not None:
            assert file_object is None
            assert pdb_code is None
            assert pdb_sf_code is None
            self.file_object = TextFile(file_name)
        elif file_object is not None:
            assert pdb_code is None
            assert pdb_sf_code is None
            self.file_object = file_object
        elif pdb_code is not None:
            assert pdb_sf_code is None
            from CDTK.PDBRepository import mmcif_files
            self.file_object = mmcif_files.getFile(pdb_code)
        elif pdb_sf_code is not None:
            from CDTK.PDBRepository import sf_files
            self.file_object = sf_files.getFile(pdb_sf_code)
        else:
            raise ValueError("No input file given")

    def parseLowLevel(self):
        """
        An iterator that yields the contents of the mmCIF file in the
        form of (type, data) pairs. The type can be KEYWORD, DATA_LABEL,
        or DATA_VALUE.
        """
        file_iter = iter(self.file_object)
        while True:
            line = file_iter.next()
            self.line_number += 1

            ## skip comments
            if line.startswith("#"):
                continue

            ## semi-colon multi-line strings
            if line.startswith(";"):
                lmerge = [line[1:]]
                while True:
                    line = file_iter.next()
                    self.line_number += 1
                    if line.startswith(";"):
                        break
                    lmerge.append(line)

                lmerge[-1] = lmerge[-1].rstrip()
                yield (DATA_VALUE, "".join(lmerge))
                continue

            ## split line into tokens
            for match in _token_regexp.finditer(line):
                label1, label2, string, token = match.groups()
                if label1 is not None and label2 is not None:
                    yield DATA_LABEL, (label1, label2)
                elif string is not None:
                    yield DATA_VALUE, string
                elif token is not None:
                    token_parts = token.split('_')
                    if len(token_parts) == 1 \
                       or token_parts[0].lower() not in ("data", "loop",
                                                         "global", "save",
                                                         "stop"):
                        yield DATA_VALUE, token
                    else:
                        yield KEYWORD, (token_parts[0].lower(), token_parts[1])

    def parse(self):
        """
        An iterator that yields the contents of the mmCIF file in the
        form of (type, data) pairs. The type can be KEYWORD (data
        is an optional label), DATA (data is a triple of (category label,
        item label, value)), TABLE_HEADER (data is a list of the category and
        item labels in the table) or TABLE_DATA (data is a list containing the
        data items for one row of the table).
        """
        iterator = self.parseLowLevel()
        while True:
            item_type, item = iterator.next()
            if item_type is KEYWORD and item[0] == "data":
                yield item_type, item
                break

        state = LABEL_OR_KEYWORD
        while True:
            item_type, item = iterator.next()

            if state is LABEL_OR_KEYWORD:
                if item_type is DATA_LABEL:
                    label1, label2 = item
                    state = VALUE
                elif item_type is KEYWORD:
                    if item[0] == "loop":
                        loop_labels = []
                        state = LOOP_LABELS
                    else:
                        yield item_type, item
                else:
                    raise MMCIFSyntaxError("Expected data label or keyword",
                                           self.line_number)

            elif state is VALUE:
                if item_type is DATA_VALUE:
                    state = LABEL_OR_KEYWORD
                    yield DATA, (label1, label2, item)
                else:
                    raise MMCIFSyntaxError("Expected data value for label %s.%s"
                                           % (label1, label2),
                                           self.line_number)

            elif state is LOOP_LABELS:
                if item_type is DATA_LABEL:
                    if loop_labels and loop_labels[0][0] != item[0]:
                        # The label does not belong to the loop category.
                        # meaning that the loop is empty and terminated.
                        label1, label2 = item
                        state = VALUE
                    else:
                        loop_labels.append(item)
                elif item_type is DATA_VALUE:
                    loop_data = [item]
                    state = LOOP_VALUES
                    yield TABLE_HEADER, loop_labels
                    if len(loop_labels) == 1:
                        yield TABLE_DATA, loop_data
                        loop_data = []
                else:
                    raise MMCIFSyntaxError("Expected label or value in loop",
                                           self.line_number)

            elif state is LOOP_VALUES:
                if item_type is DATA_VALUE:
                    loop_data.append(item)
                    if len(loop_data) == len(loop_labels):
                        yield TABLE_DATA, loop_data
                        loop_data = []
                else:
                    if len(loop_data) > 0:
                        raise MMCIFSyntaxError("Extraneous data in loop:" +
                                               str(loop_data),
                                               self.line_number)
                    if item_type is DATA_LABEL:
                        label1, label2 = item
                        state = VALUE
                    elif item_type is KEYWORD:
                        if item[0] == "loop":
                            loop_labels = []
                            state = LOOP_LABELS
                        else:
                            yield item_type, item
                    else:
                        raise MMCIFSyntaxError("Expected data label or loop",
                                               self.line_number)

    def parseAndSelect(self, categories, data=0):
        """
        An iterator that yields a subset of the contents of the mmCIF file
        in the form of (type, data) pairs. The return values are the same
        as for L{parse}. However, only items corresponding to the selection
        are yielded. The selection consists of a list of categories and of
        the specification of a data set by number (0 = first) or name.
        Note that most mmCIF files contain only a single data set.

        @param categories: a sequence of category names
        @type categories: sequence of C{str}
        @param data: the selected data set, either by number (0 being the
               first data set) or by name
        @type data: C{int} or C{str}
        """
        dataset = -1
        dataset_name = None
        for item_type, item in self.parse():

            if item_type is KEYWORD:
                if item[0] == 'data':
                    dataset += 1
                    dataset_name = item[1]
                    return_data = data == dataset or data == dataset_name
                else:
                    raise MMCIFError("Keyword %s not yet implemented" % item[0])

            elif item_type is DATA:
                if item[0] in categories and return_data:
                    yield DATA, item

            elif item_type is TABLE_HEADER:
                keep_table = False
                for label1, label2 in item:
                    if label1 in categories:
                        keep_table = True
                if keep_table and return_data:
                    yield TABLE_HEADER, item

            elif item_type is TABLE_DATA:
                if keep_table and return_data:
                    yield TABLE_DATA, item

            else:
                raise MMCIFSyntaxError("Unexpected item type %s"
                                       % str(item_type),
                                       self.line_number)

    def parseToObjects(self, data = 0, **categories):
        """
        Parse the file and store the selected data using the provided
        keyword parameters. Each keyword argument specifies one category
        of data items to be processed; categories for which there is no
        keyword parameter are ignored. The value of the keyword argument
        should be a dictionary for standard data items (which are stored
        in the dictionary) and a callable object for tables. The object
        is called once for each row in the table. The two arguments
        given are 1) a dictionary mapping item labels to indices into
        the data list and 2) the items in the row as a list.

        @param data: the selected data set, either by number (0 being the
               first data set) or by name
        @type data: C{int} or C{str}
        @param categories: the category-specific handlers
        @type categories: C{dict}
        """
        for item_type, item in self.parseAndSelect(categories.keys(), data):

            if item_type is DATA:
                label1, label2, value = item
                categories[label1][label2] = value

            elif item_type is TABLE_HEADER:
                indices = {}
                for i, (label1, label2) in enumerate(item):
                    indices[label2] = i
                handler = categories[label1]

            elif item_type is TABLE_DATA:
                handler(indices, item)

            else:
                raise MMCIFSyntaxError("Unexpected item type %s"
                                       % str(item_type),
                                       self.line_number)


class MMCIFStructureFactorData(object):

    """
    Structure factor data from an mmCIF file

    Loads reflections with structure factor amplitudes or intensities
    from the mmCIF structure factor file in the PDB. Since many
    structure factor files in the PDB do not completely respect the
    mmCIF format and dictionary, this class is generous about common
    mistakes. Nevertheless, it fails for some PDB entries.

    After successful initialization, the following attributes of
    a MMCIFStructureFactorData object contain the data:

      - reflections: a L{CDTK.Reflections.ReflectionSet} object

      - data: the experimental data stored in a
        L{CDTK.ReflectionData.ExperimentalAmplitudes} or
        L{CDTK.ReflectionData.ExperimentalIntensities} object.

      - model: the calculated structure factor stored in a
        L{CDTK.ReflectionData.StructureFactor} object. If no calculated
        data is contained in the file, the value is C{None}.
    """

    def __init__(self, sf_file=None, structure_file=None, pdb_code=None,
                 fill = False, load_model_sf = True,
                 require_sigma = True):
        """
        Specify the data to be loaded. The following combinations
        are valid:

         - pdb_code only: the data is taken from a public or local
           PDB repository

         - sf_file only: the data is taken from the structure
           factor file

         - sf_file and structure_file: cell and/or symmetry information
           is read from structure_file if it is missing in sf_file

        @param sf_file: the name of the structure factor mmCIF file
        @type sf_file: C{str}
        @param structure_file: the name of the structure mmCIT file
        @type structure_file: C{str}
        @param pdb_code: a four-letter PDB code
        @type pdb_code: C{str}
        @param fill: C{True} if the reflection set should contain all
               reflections in the resolution sphere. With the default value
               of C{False}, only the reflections listed in the mmCIF file
               will be present in the reflection set.
        @type fill: C{bool}
        @param load_model_sf: C{True} if model structure factors (F_calc)
                              should be loaded if present in the file
        @type load_model_sf: C{bool}
        @param require_sigma: if C{True}, ignore experimental data points
                              without sigma. If C{False}, set sigma to zero
                              if it is not given.
        @type require_sigma: C{bool}
        """
        if pdb_code is not None:
            self.pdb_code = pdb_code
            assert sf_file is None
            assert structure_file is None
        elif sf_file is not None:
            self.pdb_code = None
            if isinstance(sf_file, str):
                sf_file = TextFile(sf_file)
            self.sf_file = sf_file
            if structure_file is not None:
                if isinstance(structure_file, str):
                    structure_file = TextFile(structure_file)
            self.structure_file = structure_file
        else:
            raise MMCIFError("No structure factor data given")
        self.load_model_sf = load_model_sf
        self.require_sigma = require_sigma
        self.cell = {}
        self.symmetry = {}
        self.reflections = None
        self.parseSFFile()
        self.finalize(fill)

    def parseSFFile(self):
        if self.pdb_code is not None:
            parser = MMCIFParser(pdb_sf_code=self.pdb_code)
        else:
            parser = MMCIFParser(file_object=self.sf_file)
        parser.parseToObjects(cell = self.cell,
                              symmetry = self.symmetry,
                              refln = self.addReflection)
        if self.reflections is None:
            raise MMCIFError("reflection data missing")

    def parseStructureFile(self):
        if self.pdb_code is not None:
            parser = MMCIFParser(pdb_code=self.pdb_code)
        else:
            if self.structure_file is None:
                return
            parser = MMCIFParser(file_object=self.structure_file)
        parser.parseToObjects(cell = self.cell,
                              symmetry = self.symmetry)

    def setupReflectionSet(self):
        from CDTK.Crystal import UnitCell
        from CDTK.SpaceGroups import space_groups
        from CDTK.Reflections import ReflectionSet
        from CDTK.ReflectionData import ExperimentalAmplitudes, \
                                        ExperimentalIntensities
        from CDTK import Units

        if len(self.cell) == 0 or len(self.symmetry) == 0:
            self.parseStructureFile()
        if len(self.cell) == 0:
            raise MMCIFError("cell parameters missing")
        if len(self.symmetry) == 0:
            raise MMCIFError("symmetry parameters missing")

        cell = UnitCell(float(self.cell['length_a'])*Units.Ang,
                        float(self.cell['length_b'])*Units.Ang,
                        float(self.cell['length_c'])*Units.Ang,
                        float(self.cell['angle_alpha'])*Units.deg,
                        float(self.cell['angle_beta'])*Units.deg,
                        float(self.cell['angle_gamma'])*Units.deg)
        try:
            sg_key = int(self.symmetry['Int_Tables_number'])
        except (KeyError, ValueError):
            sg_key = self.symmetry['space_group_name_H-M'].strip()
        space_group = space_groups[sg_key]
        self.reflections = ReflectionSet(cell, space_group)

    def addReflection(self, indices, data):
        from Scientific import N
        if self.reflections is None:
            # First call
            from CDTK.ReflectionData import ExperimentalAmplitudes, \
                                            ExperimentalIntensities, \
                                            StructureFactor
            self.setupReflectionSet()
            self.data = None
            for label1, label2 in [('F_meas', 'F_meas_sigma'),
                                   ('F_meas_au', 'F_meas_sigma_au')]:
                if label1 in indices and label2 in indices:
                    self.data_index = indices[label1]
                    self.sigma_index = indices[label2]
                    self.data = ExperimentalAmplitudes
            for label1, label2 in [('intensity_meas', 'intensity_sigma'),
                                   ('intensity_meas_au', 'intensity_sigma_au'),
                                   ('F_squared_meas', 'F_squared_sigma')]:
                if label1 in indices and label2 in indices:
                    self.data_index = indices[label1]
                    self.sigma_index = indices[label2]
                    self.data = ExperimentalIntensities
            if self.data is None:
                if self.require_sigma:
                    raise MMCIFError("no experimental data with sigma"
                                     " found in %s" % str(indices.keys()))
                else:
                    for label1 in ['F_meas', 'F_meas_au']:
                        if label1 in indices:
                            self.data_index = indices[label1]
                            self.sigma_index = None
                            self.data = ExperimentalAmplitudes
                    for label1 in ['intensity_meas', 'intensity_meas_au',
                                   'F_squared_meas']:
                        if label1 in indices:
                            self.data_index = indices[label1]
                            self.sigma_index = None
                            self.data = ExperimentalIntensities
                    if self.data is None:
                        raise MMCIFError("no experimental data found in %s"
                                         % str(indices.keys()))
            self.model = None
            if self.load_model_sf and \
                   'F_calc' in indices and 'phase_calc' in indices:
                self.model = StructureFactor
            self.reflection_data = []

        h = int(data[indices['index_h']])
        k = int(data[indices['index_k']])
        l = int(data[indices['index_l']])
        if h*h+k*k+l*l > 0:
            ri = self.reflections.getReflection((h, k, l))
            # The status field is often used incorrectly.
            # Accept a 0/1 flag (presumably meant as "r_free flag")
            # in addition to what mmCIF defines.
            try:
                status = data[indices['status']]
            except KeyError:
                status = 'o'
            value = data[self.data_index]
            if self.sigma_index is None:
                sigma = '0.'
            else:
                sigma = data[self.sigma_index]
            observed = status in 'ofhl01' and value != '?'
            if self.require_sigma:
                observed = observed and sigma != '?'
            if observed:
                value = float(value)
                sigma = float(sigma)
            else:
                value = sigma = 0.
            if self.model is not None:
                model_amplitude = float(data[indices['F_calc']])
                model_phase = float(data[indices['phase_calc']])
            else:
                model_amplitude = model_phase = 0.
            self.reflection_data.append((h, k, l,
                                         value, sigma, observed,
                                         model_amplitude, model_phase))


    def finalize(self, fill):
        if fill:
            self.reflections.fillResolutionSphere()
        self.data = self.data(self.reflections)
        if self.model is not None:
            self.model = self.model(self.reflections)
        for h, k, l, value, sigma, observed, model_amplitude, model_phase \
            in self.reflection_data:
            ri = self.reflections[(h, k, l)]
            if observed:
                try:
                    self.data[ri] = N.array([value, sigma])
                except ValueError:
                    # Systematically absent reflections that is not marked
                    # as such in the status field. This is too common to
                    # raise an exception. Continue to the next reflection,
                    # as setting the model sf is likely to cause the same
                    # exception again.
                    continue
            if self.model is not None:
                self.model[ri] = model_amplitude*N.exp(1j*model_phase*Units.deg)

