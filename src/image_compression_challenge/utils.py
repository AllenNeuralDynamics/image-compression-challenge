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


# --- OS utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at the given path.

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.
    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes the given directory and all of its subdirectories.

    Parameters
    ----------
    path : str
        Path to directory to be removed if it exists.
    """
    if os.path.exists(path):
        shutil.rmtree(path)


# --- ZIP utils ---
def move_zip_in_zip(outer_zip_path, inner_zip_name, output_path):
    """
    Extracts a nested ZIP file from within a parent ZIP archive and saves it
    to a specified output path.

    Parameters
    ----------
    outer_zip_path : str
        Path to the parent ZIP file containing the nested ZIP.
    inner_zip_name : str
        Name (or suffix) of the inner ZIP file to extract.
    output_path : str
        Destination path where the extracted inner ZIP should be saved.
    """
    with zipfile.ZipFile(outer_zip_path, 'r') as outer_zip:
        # Find file path
        matches = [f for f in outer_zip.namelist() if f.endswith(inner_zip_name)]
        if not matches:
            raise FileNotFoundError(f"{self.inner_tiff} not found in ZIP")
        filename = matches[0]

        # Move file
        inner_zip_bytes = outer_zip.read(filename)
        with open(output_path, 'wb') as f_out:
            f_out.write(inner_zip_bytes)


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
