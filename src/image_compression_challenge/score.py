"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TiffReader
from segmentation_skeleton_metrics.utils.util import compute_weighted_avg
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import zipfile

from image_compression_challenge import utils

BLOCK_NUMS = ["005", "006", "007", "008", "009"]
ERROR_TOLS = {
    "% Omit Edges": 10,
    "Split Rate": 1000,
    "Merge Rate": 1000
}


def score(zip_path):
    # Check submission is valid
    print("\nStep 1: Check Submission")
    check_required_submission_files(zip_path)
    check_ssim(zip_path)
    #check_segmentation_consistency(zip_path)

    # Score submission
    print("\nStep 2: Score Submission")
    compression_score = compute_compressed_size(zip_path)
    return compression_score


# --- Check Submission ---
def check_required_submission_files(zip_path):
    """
    Checks if a participant's submission contains the required compressed
    image, segmentation, and SWC files.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    """
    # Subroutines
    def check_file(filename):
        err_msg = f"{filename} is missing from submitted ZIP archive!"
        assert utils.is_file_in_zip(zip_path, filename), err_msg

    # Main
    for num in tqdm(BLOCK_NUMS, desc="Checking Required Files"):
        utils.find_compressed_path(zip_path, f"compressed_{num}")
        check_file(f"decompressed_{num}.tiff")
        check_file(f"segmentation_{num}.tiff")
        check_file(f"skeletons_{num}.zip")


def check_ssim(zip_path):
    # Load original images
    img_root = "s3://aind-benchmark-data/3d-image-compression/blocks"

    # Check metric
    for num in tqdm(BLOCK_NUMS, desc="Checking SSIM"):
        # Set paths
        decompressed_filename = f"decompressed_{num}.tiff"
        original_path = f"{img_root}/block_{num}/input.zarr/0"

        # Read images
        decompressed = utils.read_zipped_tiff(zip_path, decompressed_filename)
        original = utils.read_zarr(original_path)[:]

        # Compute metric
        t0 = time()
        ssim = utils.compute_ssim(decompressed[0, 0], original[0, 0])
        print("runtime:", time() - t0)
        print("ssim:", ssim)
        assert ssim > 0.7, f"Failed with SSIM={ssim} on block {num}"


def check_segmentation_consistency(zip_path):
    move_skeleton_zips(zip_path)
    for num in tqdm(BLOCK_NUMS, desc="Checking Segmentation"):
        # Load segmentation results
        result_baseline = load_baseline_segmentation_result(num)
        result_submission = compute_segmentation_metrics(zip_path, num)

        # Compare segmentation results
        for metric in ERROR_TOLS:
            avg_baseline = compute_weighted_avg(result_baseline, metric)
            avg_sumission = compute_weighted_avg(result_submission, metric)
            error = avg_sumission - avg_baseline
            if error > ERROR_TOLS[metric] and error:
                raise ValueError(f"Failed with {metric}={error}")
    utils.rmdir("./temp")


# --- Compute Score ---
def compute_compressed_size(zip_path):
    """
    Compute the average compressed file size (in GBs) across all blocks in a
    ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.

    Returns
    -------
    score : float
        Average compressed file size (in GBs) across all blocks.
    """
    # Compute score
    compressed_size = list()
    for num in tqdm(BLOCK_NUMS, "Compute Compressed Size"):
        # Find path to compressed image
        name = f"compressed_{num}"
        compressed_img_path = utils.find_compressed_path(zip_path, name)

        # Compute compressed size
        size_gb = get_file_size(zip_path, compressed_img_path)
        compressed_size.append(size_gb)

    # Report score
    score = np.mean(compressed_size)
    print(f"Score: {score} GBs")
    return score


def get_file_size(zip_path, filename):
    """
    Gets the size (in GBs) of the given file contained in the provided
    zip.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    filename : str
        Name of file to be checked.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        info = zf.getinfo(filename)
        return info.file_size / 1024 ** 3


# --- Helpers ---
def compute_segmentation_metrics(zip_path, num):
    # Paths
    gt_path = f"s3://aind-benchmark-data/3d-image-compression/swcs/block_{num}/"
    segmentation_filename = f"segmentation_{num}.tiff"
    skeletons_path = f"./temp/skeletons_{num}.zip"
    output_dir = "./temp"

    # Read segmentation
    segmentation = TiffReader(
        zip_path, inner_tiff=segmentation_filename
    )

    # Run evaluation
    evaluate(
        gt_path,
        segmentation,
        output_dir,
        anisotropy=(0.748, 0.748, 1.0),
        fragments_pointer=skeletons_path,
        verbose=False
    )

    # Process result
    result = pd.read_csv("./temp/results.csv")
    return fill_nan_results(result)


def fill_nan_results(df):
    df["Merge Rate"] = df["Merge Rate"].fillna(df["SWC Run Length"])
    df["Split Rate"] = df["Split Rate"].fillna(df["SWC Run Length"])
    return df


def load_baseline_segmentation_result(num):
    """
    Loads the skeleton-based metric results for the baseline segmentation for
    the image block corresponding to "num".

    Parameters
    ----------
    num : str
        Unique identifier for an image block.

    Returns
    -------
    pandas.DataFrame
        Skeleton-based metric results for the baseline segmentation.
    """
    url = f"https://raw.githubusercontent.com/AllenNeuralDynamics/image-compression-challenge/main/baseline_segmentation_results/results_{num}.csv"
    result = pd.read_csv(url)
    return fill_nan_results(result)


def move_skeleton_zips(zip_path):
    """
    Extracts specific skeleton ZIP archives from a parent ZIP file and moves
    them into a temporary directory. This extraction ensures that the SWC
    files are stored in a format compatible with the SWC readers used in
    segmentation-skeleton-metrics.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    """
    # Initialize temp directory
    output_dir = "./temp/"
    utils.mkdir(output_dir)

    # Iterate over skeletons
    for num in BLOCK_NUMS:
        source_filename = f"skeletons_{num}.zip"
        destination_path = f"{output_dir}/skeletons_{num}.zip"
        utils.move_zip_in_zip(zip_path, source_filename, destination_path)
