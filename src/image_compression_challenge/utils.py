"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from skimage.metrics import structural_similarity as ssim

import numpy as np
import io
import os
import shutil
import tensorstore as ts
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
def find_compressed_path(zip_path, filename):
    """
    Finds the path to the specified compressed image.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submission ZIP archive.
    filename : str
        Name of compressed file.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in [n for n in z.namelist()]:
            if filename in name and "decompressed" not in name:
                return name
    raise Exception(f"Compressed file {filename} not found!")


def find_decompressed_path(zip_path, filename):
    """
    Finds the path to the specified decompressed image.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submission ZIP archive.
    filename : str
        Name of compressed file.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in [n for n in z.namelist()]:
            if filename in name:
                return name
    raise Exception(f"Decompressed file {filename} not found!")


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
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = [os.path.basename(n) for n in z.namelist()]
        return filename in namelist


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
    with zipfile.ZipFile(outer_zip_path, "r") as outer_zip:
        # Find file path
        name_list = outer_zip.namelist()
        matches = [f for f in name_list if f.endswith(inner_zip_name)]
        if not matches:
            raise FileNotFoundError(f"{inner_zip_name} not found in ZIP")
        filename = matches[0]

        # Move file
        inner_zip_bytes = outer_zip.read(filename)
        with open(output_path, "wb") as f_out:
            f_out.write(inner_zip_bytes)


# --- Miscellaneous ---
def compute_ssim(img1, img2, axis=0, win_size=7):
    """
    Computes the structural similarity (SSIM) between two 3D images by
    averaging the SSIM between 2D slices.

    Parameters
    ----------
    img1 : numpy.ndarray
        Image to be evaluated.
    img2 ; numpy.ndarray
        Image to be evaluated.
    axis : int, optional
        Axis to compute SSIM along. Default is 0.
    win_size : int, optional
        Size of convolutional kernel used to compute SSIM.

    Returns
    -------
    ssim : float
        Structural similarity between the two given images.
    """
    # Initializations
    assert img1.shape == img2.shape, "Images must have the same shape"
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    # Main
    ssim_values = []
    for i in range(img1.shape[0]):
        val = ssim(
            img1[i, ...],
            img2[i, ...],
            data_range=data_range,
            win_size=win_size,
        )
        ssim_values.append(val)
    return np.mean(ssim_values)


def get_tensorstore_args(img_path):
    """
    Gets the arguments needed to use tensorstore to read the given zarr image.

    Parameters
    ----------
    img_path : str
        Path to image to be read.

    Returns
    -------
    tensorstore_args : dict
        Arguments needed to use tensorstore to read the given zarr image.
    """
    if img_path.startswith("s3://"):
        bucket_name, path = parse_cloud_path(img_path)
        tensorstore_args = {
            "driver": "zarr",
            "kvstore": {"driver": "s3", "bucket": bucket_name, "path": path},
        }
    else:
        tensorstore_args = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": img_path},
        }
    return tensorstore_args


def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and key/prefix. Supports
    paths of the form: "s3://bucket_name/prefix" or without a scheme.

    Parameters
    ----------
    path : str
        Path to be parsed.

    Returns
    -------
    bucket_name : str
        Name of the bucket.
    prefix : str
        Cloud prefix.
    """
    path = path[len("s3://"):]
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def read_zarr(img_path):
    """
    Reads a Zarr volume from S3.

    Parameters
    ----------
    img_path : str
        Path to Zarr directory.

    Returns
    -------
    img : numpy.ndarray
        Image volume.
    """
    args = get_tensorstore_args(img_path)
    img = ts.open(args, open=True).result()
    img = img.read().result()[:]
    return img


def read_zipped_tiff(zip_path, filename):
    """
    Reads an TIFF file contained within a ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    filename : str
        Name of image to be read.

    Returns
    -------
    img : numpy.ndarray
        Image volume.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        # Collect only valid TIFF files, ignoring __MACOSX junk
        tiff_files = [
            f
            for f in z.namelist()
            if f.lower().endswith((".tif", ".tiff"))
            and not os.path.basename(f).startswith("._")
        ]

        # Choose file
        matches = [f for f in tiff_files if f.endswith(filename)]
        if not matches:
            raise FileNotFoundError(f"{filename} not found in ZIP")
        filename = matches[0]

        # Load TIFF
        with z.open(filename) as f:
            img = tifffile.imread(io.BytesIO(f.read()))
        return img
