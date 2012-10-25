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
import weakref

# For now, import all modules directly for hard-wired handlers
from CDTK.Reflections import FrozenReflectionSet
from CDTK.ReflectionData import ReflectionData

#
# Data model versioning 
#
# Current version
DATA_MODEL_MAJOR_VERSION = 0
DATA_MODEL_MINOR_VERSION = 1
#
# Minimal version that can be handled
DATA_MODEL_MIN_MAJOR_VERSION = 0
DATA_MODEL_MIN_MINOR_VERSION = 1


class HDF5Store(object):

    handlers = {'ReflectionSet': FrozenReflectionSet.fromHDF5,
                'ReflectionData': ReflectionData.fromHDF5}

    def __init__(self, hdf5_group_or_filename, file_mode=None):
        if isinstance(hdf5_group_or_filename, h5py.Group):
            assert file_mode is None
            self.root = hdf5_group_or_filename
        else:
            assert isinstance(hdf5_group_or_filename, basestring)
            if file_mode is None:
                file_mode = 'r'
            self.root = h5py.File(hdf5_group_or_filename, file_mode)
        self.path_cache = weakref.WeakKeyDictionary()
        self.obj_cache = weakref.WeakValueDictionary()
        self.info_cache = weakref.WeakKeyDictionary()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root.file.close()
        return False

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
        if path == '':
            # unnamed data
            path = "_UNNAMED_/%d" % id(obj)
        path = self.relativePath(path)

        cached_path = self.path_cache.get(obj, None)
        if cached_path is not None:
            if path != cached_path:
                if not path.startswith("_UNNAMED_"):
                    # add a soft link
                    self.root[path] = h5py.SoftLink(cached_path)
            node = self.root[cached_path]
            info = self.info_cache.get(obj, None)
            return node, info

        ret = obj.storeHDF5(self, path)
        if isinstance(ret, tuple):
            assert len(ret) == 2
            node, info = ret
        else:
            node = ret
            info = None
        self.obj_cache[path] = obj
        self.path_cache[obj] = path
        self.info_cache[obj] = info
        return node, info

    def stamp(self, node, data_class):
        node.attrs['DATA_MODEL'] = 'CDTK'
        node.attrs['DATA_MODEL_MAJOR_VERSION'] = DATA_MODEL_MAJOR_VERSION
        node.attrs['DATA_MODEL_MINOR_VERSION'] = DATA_MODEL_MINOR_VERSION
        node.attrs['DATA_CLASS'] = data_class
        
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
        data = self.obj_cache.get(path, None)
        if data is not None:
            return data
        link_obj = self.root.get(path, getlink=True)
        if isinstance(link_obj, h5py.SoftLink):
            return self.retrieve(link_obj.path)
        node = self.root[path]
        data_class = node.attrs.get('DATA_CLASS', None)
        if data_class is None:
            raise ValueError("unknown node type at " + path)

        if node.attrs.get('DATA_MODEL', None) != 'CDTK' \
           or node.attrs.get('DATA_MODEL_MAJOR_VERSION', None) \
                            > DATA_MODEL_MIN_MAJOR_VERSION \
           or node.attrs.get('DATA_MODEL_MINOR_VERSION', None) \
                            > DATA_MODEL_MIN_MINOR_VERSION:
            raise ValueError("Unknow data model or invalid version number")

        obj = self.handlers[data_class](self, node)
        self.obj_cache[path] = obj
        self.path_cache[obj] = path
        return obj
