# Access to files from the PDB
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Access to files in a local PDB repository or by download from the PDB

This module defines the class PDBFileCollection and three instances for
the three main PDB database directories:

 - pdb_files for the collection of files in the PDB format
 - mmcif_files for the collection of files in mmCIF format
 - sf_files for the collection of structure factor files in mmCIF format

The database directories are taken from the environment variables
PDB_PATH, PDB_MMCIF_PATH, and PDB_SF_PATH. If a directory is
undefined, files are downloaded by HTTP access to the PDB Web site
(http://www.rcsb.org/pdb/). Only the getFile() method will work in
this case.
"""

import cStringIO
import gzip
import os
import urllib

pdb_path = os.environ.get('PDB_PATH', None)
pdb_mmcif_path = os.environ.get('PDB_MMCIF_PATH', None)
pdb_sf_path = os.environ.get('PDB_SF_PATH', None)


class PDBFileCollection(object):

    """
    A PDBFileCollection describes a database directory in a local PDB
    repository, containing either PDB files, mmCIF files, or structure
    factor files.
    """

    def __init__(self, base_path, filename_pattern, url_pattern):
        """
        @param base_path: the path to the database directory, i.e. the
                          path containing the two-letter subdirectories
        @type base_path: C{str}
        @param filename_pattern: the pattern of the file names in the
                                 database, containing %s at the place of
                                 the four-letter PDB code.
        @type filename_pattern: C{str}
        @param url_pattern: the pattern of the URLs for download from the
                            PDB server, containing %s at the place of
                            the four-letter PDB code.
        @type url_pattern: C{str}
        """
        self.base_path = base_path
        self.is_local = base_path is not None
        self.filename_pattern = filename_pattern
        self.url_pattern = url_pattern
        self.pdb_code_index = self.filename_pattern.find('%s')

    def getFilename(self, pdb_code):
        """
        @param pdb_code: the four-letter PDB code
        @type pdb_code: C{str}
        @return: the corresponding file name
        @rtype: C{str}
        """
        assert len(pdb_code) == 4, "Invalid PDB code " + repr(pdb_code)
        if self.base_path is None:
            raise IOError("Directory path undefined")
        pdb_code = pdb_code.lower()
        subdir = pdb_code[1:3]
        return os.path.join(self.base_path, subdir,
                            self.filename_pattern % pdb_code)

    def fileExists(self, pdb_code):
        """
        @param pdb_code: the four-letter PDB code
        @type pdb_code: C{str}
        @return: C{True} if there is a corresponding file
        @rtype: C{bool}
        """
        return os.path.exists(self.getFilename(pdb_code))

    def getFile(self, pdb_code):
        """
        @param pdb_code: the four-letter PDB code
        @type pdb_code: C{str}
        @return: the corresponding file
        @rtype: C{file}
        """
        if not self.is_local:
            assert len(pdb_code) == 4, "Invalid PDB code " + repr(pdb_code)
            pdb_code = pdb_code.lower()
            if self.url_pattern is None:
                raise IOError("No URL pattern for PDB repository")
            url = self.url_pattern % pdb_code
            filename, headers = urllib.urlretrieve(url)
            if filename[-2:] == 'gz':
                return gzip.GzipFile(filename)
            else:
                return file(filename)
        filename = self.getFilename(pdb_code)
        if filename[-2:] == 'gz':
            return gzip.GzipFile(filename)
        else:
            return file(filename)

    def __iter__(self):
        """
        @return: a generator yielding the PDB codes for all the files
                 in the collection
        @rtype: generator
        """
        for dirpath, dirname, filenames in os.walk(self.base_path):
            for filename in filenames:
                pdb_code = filename[self.pdb_code_index:
                                    self.pdb_code_index+4]
                if self.filename_pattern % pdb_code == filename:
                    yield pdb_code


pdb_files = PDBFileCollection(pdb_path, 'pdb%s.ent.gz',
                              'http://www.rcsb.org/pdb/files/%s.pdb.gz')
mmcif_files = PDBFileCollection(pdb_mmcif_path, '%s.cif.gz',
                                'http://www.rcsb.org/pdb/files/%s.cif.gz')
sf_files = PDBFileCollection(pdb_sf_path, 'r%ssf.ent.gz',
                             'http://www.rcsb.org/pdb/files/r%ssf.ent.gz')
