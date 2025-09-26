"""
Created on Thu Sep 25 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that scores submissions to the image compression data challenge.

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader

from image_compression_challenge import utils


def score(zip_path, read_fn):
    # Check submission contains required files
    check_files_in_submission(zip_path)

    # Check segmentation consistency
    if is_segmentation_consistent(zip_path):
        print("Pass [2/2]: Submission has consistent segmentation!")
    else:
        print("Failed Segmentation Consistency Check!")
        return np.inf

    # Score submission
    compression_score = compute_compressed_size(zip_path)
    decompression_score = compute_decompression_cost(zip_path, read_fn)


def check_files_in_submission(zip_path):
    for n in range(5, 6):
        segmentation_filename = f"segmentation_00{n}.tiff"
        skeletons_filename = f"skeletons_00{n}.zip"
        assert utils.is_file_in_zip(zip_path, segmentation_filename)
        assert utils.is_file_in_zip(zip_path, skeletons_filename)
    print("Pass [1/2]: Submission contains required files!")

# --- Check Segmentations ---
def is_segmentation_consistent(zip_path):
    for n in range(5, 6):
        block_num = f"00{n}"
        compute_segmentation_metrics(zip_path, n)


def compute_segmentation_metrics(zip_path, n):
    # Paths
    gt_path = f"s3://aind-benchmark-data/3d-image-compression/swcs/block_00{n}/"
    output_dir = "./temp"
    segmentation_filename = f"submission/segmentation_00{n}.tiff"
    skeletons_path = f"{zip_path}/skeletons_00{n}.zip"

    # Read segmentation
    segmentation = TiffReader(
        zip_path,
        inner_tiff=segmentation_filename,
        swap_axes=False
    )

    # Run evaluation
    skeleton_metric = SkeletonMetric(
        gt_path, segmentation, output_dir, fragments_pointer=skeletons_path
    )
    skeleton_metric.run()


# --- Compute Score ---
def compute_compressed_size(zip_path):
    return np.inf


def compute_decompression_cost(zip_path, read_fn):
    return np.inf
