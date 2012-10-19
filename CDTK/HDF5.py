# Persistent storage in HDF5 files
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
HDF5 storage

Implements persistent storage of CDTK object in HDF5 files.
All HDF5 I/O of CDTK objects should pass through this module,
which ensures that references between objects are correctly
translated to and from HDF5 object references.
"""

import h5py

# For now, import all modules directly for hard-wired handlers
from CDTK.Reflections import FrozenReflectionSet
from CDTK.ReflectionData import ReflectionData

class HDF5Store(object):

    handlers = {'ReflectionSet': FrozenReflectionSet.fromHDF5,
                'ReflectionData': ReflectionData.fromHDF5}

    def __init__(self, hdf5_group):
        assert isinstance(hdf5_group, h5py.Group)
        self.root = hdf5_group
        self.cache = {}

    def relativePath(self, path):
        if path[0] == '/':
            if path.startswith(self.root.name):
                path = path[len(self.root.name):]
                if path[0] == '/':
                    path = path[1:]
            else:
                raise ValueError("path name not inside root group")
        return path

    def store(self, path, obj):
        path = self.relativePath(path)
        ret = obj.storeHDF5(path)
        self.return_value_cache[obj] = ret
        self._register(path, obj)
        return node

    def retrieve(self, path_or_node_or_ref):
        if isinstance(path_or_node_or_ref, h5py.h5r.Reference):
            path_or_node_or_ref = self.root[path_or_node_or_ref]
        if isinstance(path_or_node_or_ref, (h5py.highlevel.Group,
                                            h5py.highlevel.Dataset)):
            path = path_or_node_or_ref.name
        elif isinstance(path_or_node_or_ref, str):
            path = path_or_node_or_ref
        else:
            raise TypeError('path_or_node_or_ref must be a string, '
                            'an HDF5 node, or an HDF5 reference')
        path = self.relativePath(path)
        data = self.cache.get(path, None)
        if data is not None:
            return data
        node = self.root[path]
        data_class = node.attrs.get('DATA_CLASS', None)
        if data_class is None:
            raise ValueError("unknown node type at " + path)
        obj = self.handlers[data_class](self, node)
        self.cache[path] = obj
        self.cache[id(obj)] = path
        return obj
