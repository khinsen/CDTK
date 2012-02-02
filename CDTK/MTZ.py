# MTZ file reader
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
Structure factor files in MTZ (CCP4) format

This module contains an interface (read-only for the moment) to
the MTZ library of the CCP4 distribution.

Usage::

  from CDTK.MTZ import MTZFile
  sf_file = MTZFile(filename)

  print 'Title:', sf_file.title
  print
  print 'Symmetry:'
  print '  Space group name:', sf_file.symmetry.space_group_name
  print '  Space group number:', sf_file.symmetry.space_group_number
  print '  Point group name:', sf_file.symmetry.point_group_name
  print '  Symmetry operations:'
  for symop in sf_file.symmetry.symmetry_operations:
      print symop

  for crystal in sf_file.crystals:
      print
      print 'Crystal ', crystal.id
      print '  Name: ', crystal.name
      print '  Project name: ', crystal.project_name
      print '  Resolution range: ', crystal.resolution_range
      print '  Cell: ', crystal.cell
      for dataset in crystal.datasets:
          print '  Dataset', dataset.id
          print '    Name:', dataset.name
          print '    Wavelength:', dataset.wavelength
          for label, column in dataset.columns.items():
              print '    Column: ', label
              print '      Type: ', repr(column.type)
              print '      Range: ', column.range
              print '      Values (first five): ', column.values[:5]
              print '      Missing number flag(first five): ', \\
                                          column.missing_number_flag[:5]

.. moduleauthor:: Konrad Hinsen <konrad.hinsen@cnrs-orleans.fr>

"""

from CDTK_MTZ import MTZFile
