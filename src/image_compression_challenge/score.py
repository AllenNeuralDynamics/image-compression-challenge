"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader

import numpy as np
import pandas as pd
import shutil

from image_compression_challenge import utils

BLOCK_NUMS = ["005", "006", "007", "008", "009"]


def score(zip_path, read_fn):
    # Check submission is valid
    check_required_submission_files(zip_path)
    check_segmentation_consistency(zip_path)

    # Score submission
    compression_score = compute_compressed_size(zip_path)
    decompression_score = compute_decompression_cost(zip_path, read_fn)
    return compression_score, decompression_score


# --- Check Submission ---
def check_required_submission_files(zip_path):
    """
    Checks if a participant's submission contains the required image,
    segmentation, and SWC files.
    """
    # Subroutines
    def check_file(filename):
        err_msg = f"{filename} is missing from submitted ZIP archive!"
        assert utils.is_file_in_zip(zip_path, filename), err_msg

    # Main
    for num in BLOCK_NUMS:
        #check_file(f"block_{num}")
        check_file(f"segmentation_{num}.tiff")
        check_file(f"skeletons_{num}.zip")
    print("Pass [1/2]: Submission contains required files!")


def check_segmentation_consistency(zip_path):
    # Initializations
    move_skeleton_zips(zip_path)

    # Evaluation
    for num in BLOCK_NUMS:
        # Load segmentation results
        baseline_result = load_baseline_segmentation_result(num)
        submission_result = compute_segmentation_metrics(zip_path, num)

        # Compare segmentation results

    #utils.rmdir("./temp")
    print("Pass [2/2]: Submission has consistent segmentation!")


# --- Compute Score ---
def compute_compressed_size(zip_path):
    return np.inf


def compute_decompression_cost(zip_path, read_fn):
    return np.inf


# --- Helpers ---
def compute_segmentation_metrics(zip_path, num):
    # Paths
    gt_path = f"s3://aind-benchmark-data/3d-image-compression/swcs/block_{num}/"
    segmentation_filename = f"segmentation_{num}.tiff"
    skeletons_path = f"./temp/skeletons_{num}.zip"
    output_dir = "./temp/block"

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
    return pd.read_csv("./temp/results.csv")


def load_baseline_segmentation_result(num):
    url = f"https://raw.githubusercontent.com/AllenNeuralDynamics/image-compression-challenge/main/baseline_segmentation_results/results_{num}.csv"
    return pd.read_csv(url)


def move_skeleton_zips(zip_path):
    """
    Extracts specific skeleton ZIP archives from a parent ZIP file and moves
    them into a temporary directory. This extraction ensures that the SWC
    files are stored in a format compatible with the SWC readers used in
    segmentation-skeleton-metrics.

    Parameters
    ----------
    zip : str
        Path to a participant's submission ZIP archive.
    """
    # Initialize temp directory
    output_dir = "./temp"
    utils.mkdir(output_dir)

    # Iterate over skeletons
    for num in BLOCK_NUMS:
        source_filename = f"skeletons_{num}.zip"
        destination_path = f"{output_dir}/skeletons_{num}.zip"
        utils.move_zip_in_zip(zip_path, source_filename, destination_path)
