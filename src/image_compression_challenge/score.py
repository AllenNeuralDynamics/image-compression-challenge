"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from concurrent.futures import as_completed, ProcessPoolExecutor
from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TiffReader
from segmentation_skeleton_metrics.utils.util import compute_weighted_avg
from tqdm import tqdm

import numpy as np
import pandas as pd
import zipfile

from image_compression_challenge import utils

VALIDATE_NUMS = ["000", "001", "002", "003", "004"]
TEST_NUMS = ["005", "006", "007", "008", "009"]
ERROR_TOLS = {
    "% Omit Edges": 10,
    "Split Rate": 1000,
    "Merge Rate": 1000
}


def score(zip_path, use_test_blocks=True):
    """
    Evaluates a compressed submission file by validating its contents and
    computing its compression score.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    use_test_blocks : bool, optional
        Indication of whether to run evaluation using test blocks. Otherwise,
        the validation blocks are used. Default is True.
    """
    # Set block IDs
    block_nums = TEST_NUMS if use_test_blocks else VALIDATE_NUMS

    # Check submission is valid
    print("\nStep 1: Check Submission")
    #check_required_submission_files(zip_path, block_nums)
    check_ssim(zip_path, block_nums)
    check_segmentation_consistency(zip_path, block_nums)

    # Score submission
    print("\nStep 2: Score Submission")
    compression_score = compute_compressed_size(zip_path, block_nums)
    return compression_score


# --- Check Submission ---
def check_required_submission_files(zip_path, block_nums):
    """
    Checks if a participant's submission contains the required compressed
    image, segmentation, and SWC files.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    block_nums : List[str]
        Block numbers to use in evaluation.
    """
    # Subroutines
    def check_file(filename):
        """
        Checks that file is contained in the submitted ZIP archive.

        Parameters
        ----------
        filename : str
            Name of file to be checked.
        """
        err_msg = f"{filename} is missing from submitted ZIP archive!"
        assert utils.is_file_in_zip(zip_path, filename), err_msg

    # Main
    for num in tqdm(block_nums, desc="Checking Required Files"):
        utils.find_compressed_path(zip_path, f"compressed_{num}")
        check_file(f"decompressed_{num}.tiff")
        check_file(f"segmentation_{num}.tiff")
        check_file(f"skeletons_{num}.zip")


def check_ssim(zip_path, block_nums):
    """
    Checks the decompressed image quality for all benchmark blocks by
    computing the Structural Similarity Index (SSIM) between decompressed
    and original data.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    block_nums : List[str]
        Block numbers specifying what blocks to use in evaluation.
    """
    img_root = "s3://aind-benchmark-data/3d-image-compression/blocks"
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Assign processes
        pending = dict()
        for num in block_nums:
            # Set paths
            decompressed_filename = f"decompressed_{num}.tiff"
            original_path = f"{img_root}/block_{num}/input.zarr/0"

            # Submit thread
            thread = executor.submit(
                _compute_ssim, original_path, zip_path, decompressed_filename
            )
            pending[thread] = num

        # Process results
        pbar = tqdm(total=len(block_nums), desc="Checking SSIM")
        for thread in as_completed(pending.keys()):
            num = pending.pop(thread)
            ssim = thread.result()
            assert ssim > 0.9, f"Failed with SSIM={ssim} on block {num}"
            pbar.update(1)


def _compute_ssim(original_path, zip_path, decompressed_filename):
    """
    Computes the Structural Similarity Index (SSIM) between an image and its
    decompressed counterpart stored in a ZIP archive.

    Parameters
    ----------
    original_path : str
        Path to the original Zarr dataset containing the reference image.
    zip_path : str
        Path to the ZIP archive containing the decompressed TIFF image.
    decompressed_filename : str
        Name of the TIFF file within the ZIP archive to be compared.

    Returns
    -------
    ssim : float
        Computed SSIM value between the decompressed and original images,
        where values close to 1 indicate high similarity.
    """
    # Read images
    decompressed = utils.read_zipped_tiff(zip_path, decompressed_filename)
    original = utils.read_zarr(original_path)

    # Compute metric
    ssim = utils.compute_ssim(decompressed[0, 0], original[0, 0])
    return ssim


def check_segmentation_consistency(zip_path, block_nums):
    """
    Checks segmentation results against baseline metrics to ensure
    consistency.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    block_nums : List[str]
        Block numbers specifying what blocks to use in evaluation.
    """
    move_skeleton_zips(zip_path, block_nums)
    for num in tqdm(block_nums, desc="Checking Segmentation"):
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
def compute_compressed_size(zip_path, block_nums):
    """
    Computes the average compressed file size (in GBs) across all blocks in a
    ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    block_nums : List[str]
        Block numbers specifying what blocks to use in evaluation.

    Returns
    -------
    score : float
        Average compressed file size (in GBs) across all blocks.
    """
    # Compute score
    compressed_size = list()
    for num in tqdm(block_nums, "Compute Compressed Size"):
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

    Returns
    -------
    float
        Size of the given file.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        info = zf.getinfo(filename)
        return info.file_size / 1024 ** 3


# --- Helpers ---
def compute_segmentation_metrics(zip_path, num):
    """
    Computes skeleton-based segmentation metrics for a given image.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    num : str
        Unique identifier for an image block.

    Returns
    -------
    results : pandas.DataFrame
        Data frame containing skeleton metric results.
    """
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
    results = pd.read_csv(f"./temp/results.csv")
    return fill_nan_results(results)


def fill_nan_results(df):
    """
    Replaces NaN values in 'Merge Rate' and 'Split Rate' columns with values
    from 'SWC Run Length'.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing skeleton metric results on the baseline
        segmentation.

    Returns
    -------
    df : pandas.DataFrame
        Data frame containing skeleton metric results with NaN values
        replaced.
    """
    # Approximate actual run length

    # Fill NaNs
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


def move_skeleton_zips(zip_path, block_nums):
    """
    Extracts specific skeleton ZIP archives from a parent ZIP file and moves
    them into a temporary directory. This extraction ensures that the SWC
    files are stored in a format compatible with the SWC readers used in
    segmentation-skeleton-metrics.

    Parameters
    ----------
    zip_path : str
        Path to a participant's submitted ZIP archive.
    block_nums : List[str]
        Block numbers specifying what blocks to use in evaluation.
    """
    # Initialize temp directory
    output_dir = "./temp/"
    utils.mkdir(output_dir)

    # Iterate over skeletons
    for num in block_nums:
        source_filename = f"skeletons_{num}.zip"
        destination_path = f"{output_dir}/skeletons_{num}.zip"
        utils.move_zip_in_zip(zip_path, source_filename, destination_path)
