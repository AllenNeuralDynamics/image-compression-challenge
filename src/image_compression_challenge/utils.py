"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

import io
import os
import tifffile
import zipfile


def is_file_in_zip(zip_path, filename):
    """
    Checks if the given filename is contained in the ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to ZIP archive to be checked.
    filename : str
        Filename to be searched for in ZIP archive.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = [os.path.basename(n) for n in z.namelist()]
        return filename in namelist
