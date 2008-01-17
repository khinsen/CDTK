# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
@undocumented: mmCIF
"""

#
# Package information
#
from __pkginfo__ import __version__

#
# Add shared library path to sys.path
#
import os, sys
sys.path.append(os.path.join(os.path.split(__file__)[0], sys.platform))
del os
del sys
