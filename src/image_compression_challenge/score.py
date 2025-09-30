"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader

import numpy as np
import shutil

from image_compression_challenge import utils


def score(zip_path, read_fn):
    # Check submission is valid
    check_required_submission_files(zip_path)
    check_segmentation_consistency(zip_path)

    # Score submission
    compression_score = compute_compressed_size(zip_path)
    decompression_score = compute_decompression_cost(zip_path, read_fn)
    score = compression_score + compression_score
    return score


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
    for n in range(5, 10):
        #check_file(f"block_00{n}")
        check_file(f"segmentation_00{n}.tiff")
        check_file(f"skeletons_00{n}.zip")
    print("Pass [1/2]: Submission contains required files!")


def check_segmentation_consistency(zip_path):
    # Initializations
    move_skeleton_zips(zip_path)

    # Evaluation
    for n in range(5, 10):
        compute_segmentation_metrics(zip_path, n)

    #utils.rmdir("./temp")
    print("Pass [2/2]: Submission has consistent segmentation!")


def compute_segmentation_metrics(zip_path, n):
    # Paths
    gt_path = f"s3://aind-benchmark-data/3d-image-compression/swcs/block_00{n}/"
    segmentation_filename = f"segmentation_00{n}.tiff"
    skeletons_path = f"./temp/skeletons_00{n}.zip"
    output_dir = f"./temp/block_00{n}"

    # Read segmentation
    segmentation = TiffReader(zip_path, inner_tiff=segmentation_filename)

    # Run evaluation
    skeleton_metric = SkeletonMetric(
        gt_path, segmentation, output_dir, fragments_pointer=skeletons_path
    )
    skeleton_metric.run()


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
    for n in range(5, 10):
        source_filename = f"skeletons_00{n}.zip"
        destination_path = f"{output_dir}/skeletons_00{n}.zip"
        utils.move_zip_in_zip(zip_path, source_filename, destination_path)


# --- Compute Score ---
def compute_compressed_size(zip_path):
    return np.inf


def compute_decompression_cost(zip_path, read_fn):
    return np.inf
