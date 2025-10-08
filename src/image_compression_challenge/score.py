"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import shutil

from image_compression_challenge import utils

BLOCK_NUMS = ["005", "006", "007", "008", "009"]
ERROR_TOLS = {
    "% Omit Edges": 10,
    "Split Rate": 500,
    "Merge Rate": 500
}


def score(zip_path, read_fn):
    # Check submission is valid
    print("Check Submission...")
    check_required_submission_files(zip_path)
    check_segmentation_consistency(zip_path)

    # Score submission
    print("Score Submission...")
    compression_score = compute_compressed_size(zip_path)
    decompression_score = compute_decompression_cost(zip_path, read_fn)
    return compression_score, decompression_score


# --- Check Submission ---
def check_required_submission_files(zip_path):
    """
    Checks if a participant's submission contains the required compressed
    image, segmentation, and SWC files.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submission ZIP archive.
    """
    # Subroutines
    def check_file(filename):
        err_msg = f"{filename} is missing from submitted ZIP archive!"
        assert utils.is_file_in_zip(zip_path, filename), err_msg

    # Main
    for num in tqdm(BLOCK_NUMS, desc="Checking Required Files"):
        #check_file(f"compressed_{num}")
        check_file(f"segmentation_{num}.tiff")
        check_file(f"skeletons_{num}.zip")


def check_segmentation_consistency(zip_path):
    # Initializations
    move_skeleton_zips(zip_path)

    # Evaluation
    for num in BLOCK_NUMS:
        # Load segmentation results
        result_baseline = load_baseline_segmentation_result(num)
        result_submission = compute_segmentation_metrics(zip_path, num)

        # Compare segmentation results
        for metric in tqdm(ERROR_TOLS, desc="Checking Segmentation"):
            error = submission_result[metric] - baseline_result[metric]
            if error > ERROR_TOLS[metric]:
                raise ValueError(f"Failed with {metric}={submission_result[metric]}")
    #utils.rmdir("./temp")


# --- Compute Score ---
def compute_compressed_size(zip_path):
    return np.inf


def compute_decompression_cost(zip_path, read_fn):
    decompression_cost = list()
    for num in BLOCK_NUMS:
        # Find path to compressed image
        compressed_img_path = find_compressed_path(zip_path, num)

        # Decompress image
        t0 = time()
        read_fn(compressed_img_path)
        decompression_cost.append(time() - t0)
    return np.mean(decompression_cost)


# --- Helpers ---
def compute_segmentation_metrics(zip_path, num):
    # Paths
    gt_path = f"s3://aind-benchmark-data/3d-image-compression/swcs/block_{num}/"
    segmentation_filename = f"segmentation_{num}.tiff"
    skeletons_path = f"./temp/skeletons_{num}.zip"
    output_dir = "./temp"

    # Read segmentation
    segmentation = TiffReader(zip_path, inner_tiff=segmentation_filename)

    # Run evaluation
    skeleton_metric = SkeletonMetric(
        gt_path,
        segmentation,
        output_dir,
        fragments_pointer=skeletons_path,
        anisotropy=(0.748, 0.748, 1.0),
    )
    skeleton_metric.run()

    # Process result
    result = pd.read_csv("./temp/results.csv")
    return fill_nan_results(result)


def fill_nan_results(df):
    df["Merge Rate"] = df["Merge Rate"].fillna(df["SWC Run Length"])
    df["Split Rate"] = df["Split Rate"].fillna(df["SWC Run Length"])
    return df


def find_compressed_path(zip_path, num):
    """
    Finds the path for the compressed image corresponding to "num".

    Parameters
    ----------
    zip_path : str
        Path to a participant's submission ZIP archive.
    num : str
        Unique identifier for an image block.
    """
    pass


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
        Path to a participant's submission ZIP archive.
    """
    # Initialize temp directory
    output_dir = "./temp/"
    utils.mkdir(output_dir)

    # Iterate over skeletons
    for num in BLOCK_NUMS:
        source_filename = f"skeletons_{num}.zip"
        destination_path = f"{output_dir}/skeletons_{num}.zip"
        utils.move_zip_in_zip(zip_path, source_filename, destination_path)
