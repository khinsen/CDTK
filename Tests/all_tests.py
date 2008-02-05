# Run the complete test suite
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest

import unit_cell_tests
import reflection_set_tests
import structure_factor_tests
import map_tests
import refinement_tests
import mmtk_tests

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(unit_cell_tests.suite())
    test_suite.addTests(reflection_set_tests.suite())
    test_suite.addTests(structure_factor_tests.suite())
    test_suite.addTests(map_tests.suite())
    test_suite.addTests(refinement_tests.suite())
    test_suite.addTests(mmtk_tests.suite())
    return test_suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

