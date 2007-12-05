import unittest

import unit_cell_tests
import reflection_set_tests
import structure_factor_tests
import map_tests
import mmtk_tests

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(unit_cell_tests.suite())
    test_suite.addTests(reflection_set_tests.suite())
    test_suite.addTests(structure_factor_tests.suite())
    test_suite.addTests(map_tests.suite())
    test_suite.addTests(mmtk_tests.suite())
    return test_suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

