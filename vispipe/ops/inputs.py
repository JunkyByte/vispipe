from vispipe import block
from glob import iglob
import numpy as np
import pickle
import os
import json
"""
Input generators for your pipeline.
"""


@block(tag='input')
class np_iter_file:
    """
    Yields the raw buffer, line by line from a numpy file.
    This is similar to stacking a ``numpy_flow`` into an ``iterator``.
    This is a finite generator and will stop the pipeline on end.

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded array line by line.
    """
    def __init__(self, path: str = ''):
        self.file = iter(np.load(path))

    def run(self):
        yield next(self.file)


@block(tag='input', max_queue=1)
class np_file:
    """
    Yields the full raw buffer once from a numpy file.

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded array.
    """
    def __init__(self, path: str = ''):
        self.file = np.load(path)

    def run(self):
        if self.file is StopIteration:
            raise StopIteration
        x = self.file
        self.file = StopIteration
        yield x


@block(tag='input', max_queue=1)
def np_flow(path):
    """
    Yields the raw buffer from the path provided as input.
    This is an infinite generator and will never raise a StopIteration

    Yields
    ------
        The full content of the loaded array from the path you provided.
    """
    yield np.load(path)


@block(tag='input')
class pickle_iter_file:
    """
    Yields the raw buffer, line by line from a pickle file.
    This is similar to stacking a ``numpy_flow`` into an ``iterator``.
    This is a finite generator and will stop the pipeline on end.

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded pickle line by line.
    """
    def __init__(self, path: str = ''):
        self.file = iter(pickle.load(open(path, 'rb')))

    def run(self):
        yield next(self.file)


@block(tag='input')
class pickle_file:
    """
    Yields the full raw buffer once from a pickle file.

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded pickle.
    """
    def __init__(self, path: str = ''):
        self.file = pickle.load(open(path, 'rb'))

    def run(self):
        if self.file is StopIteration:
            raise StopIteration
        x = self.file
        self.file = StopIteration
        yield x


@block(tag='input', max_queue=1)
def pickle_flow(path):
    """
    Yields the raw buffer from the path provided as input.
    This is an infinite generator and will never raise a StopIteration.

    Yields
    ------
        The full content of the loaded pickle from the path you provided.
    """
    yield pickle.load(open(path, 'rb'))


@block(tag='input')
class json_file:
    """
    Yields the full raw buffer once from a json file.

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded json.
    """
    def __init__(self, path: str = ''):
        self.file = json.load(open(path))

    def run(self):
        if self.file is StopIteration:
            raise StopIteration
        x = self.file
        self.file = StopIteration
        yield x


@block(tag='input', max_queue=1)
def json_flow(path):
    """
    Yields the raw buffer from the path provided as input.
    This is an infinite generator and will never raise a StopIteration.

    Yields
    ------
        The full content of the loaded json from the path you provided.
    """
    yield json.load(open(path))


@block(tag='input', max_queue=10)
class iter_folders:
    def __init__(self, root_dir: str = '', extension: str = '', recursive: bool = False):
        """
        Yields the absolute path of all the files with a specified extension in a folder.
        It can be recursive, the paths will be relative to the folder specified.
        (e.g. if you want absolute paths specify an absolute root_dir)

        Parameters
        ----------
        root_dir : str
            The main directory from which the search will start.
        extension : str
            The extension of the files, if not specified all extensions will be returned.
        recursive : bool
            Whether the search should be recursive or not.
        """
        root_dir = os.path.join(root_dir, '**') if recursive else root_dir
        self.files = (f for f in iglob(root_dir, recursive=recursive) if
                f.endswith(extension) and os.path.isfile(f))

    def run(self):
        yield next(self.files)
