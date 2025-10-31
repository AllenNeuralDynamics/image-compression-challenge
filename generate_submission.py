"""
Created on Wed Oct 7 20:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Runs image segmentation pipeline for a dataset of image blocks.

"""

from aind_exaspim_neuron_segmentation import inference
from aind_exaspim_neuron_segmentation.utils import img_util, util

import numpy as np
import os
import tifffile


def main():
    """
    Runs image segmentation pipeline for a dataset of image blocks.
    """
    # Initializations
    model = inference.load_model(model_path, affinity_mode=affinity_mode)
    mips_dir = os.path.join(output_dir, "MIPs")
    util.mkdir(output_dir)
    util.mkdir(mips_dir)

    # Main
    for n in range(5, 10):
        # Read image
        num = f"00{n}"
        img_path = "path-to-compressed-block_num"
        img = "read-compressed-image"

        output_path = os.path.join(mips_dir, f"input_{num}.png")
        img_util.plot_mips(img, output_path=output_path)

        # Predict affinities
        affinities = inference.predict(
            img,
            model,
            affinity_mode=affinity_mode,
            batch_size=batch_size,
            brightness_clip=300,
            normalization_percentiles=(1, 99.9),
            overlap=overlap,
            patch_shape=patch_shape,
            trim=trim
        )

        output_path = os.path.join(mips_dir, f"affs_{num}.png")
        img_util.plot_mips(
            affinities[0],
            output_path=output_path
        )

        img_path = f"{output_dir}/decompressed_{num}.tiff"
        tifffile.imwrite(
            img_path,
            img.astype(np.uint16),
            compression='zlib'
        )
        del img

        # Generate segmentation
        segmentation = inference.affinities_to_segmentation(
            affinities,
            agglomeration_thresholds=[0.6, 0.8, 0.9],
            min_segment_size=100
        )
        del affinities

        output_path = os.path.join(mips_dir, f"segmentation_{num}.png")
        img_util.plot_segmentation_mips(segmentation, output_path=output_path)

        # Save results
        zipped_swcs_path = f"{output_dir}/skeletons_{num}.zip"
        inference.segmentation_to_zipped_swcs(segmentation, zipped_swcs_path)

        segmentation_path = f"{output_dir}/segmentation_{num}.tiff"
        tifffile.imwrite(
            segmentation_path,
            segmentation.astype(np.uint16),
            compression='zlib'
        )


if __name__ == "__main__":
    # Parameters
    affinity_mode = True
    batch_size = 16
    device = "cuda"
    overlap = (32, 32, 32)
    patch_shape = (96, 96, 96)
    trim = 8

    # Paths
    model_name = "UNet3d-20251019-643-0.6649"
    model_path = f"/home/jupyter/models/data-challenge-segmentation/{model_name}.pth"
    output_dir = "path-to-output-dir"

    # Main
    main()
