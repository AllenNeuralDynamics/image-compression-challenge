# Image Compression Challenge

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Add description...

## Prepare Submission

After running your image compression algorithm on the test dataset: [block_005](https://open.quiltdata.com/b/aind-benchmark-data/tree/3d-image-compression/blocks/block_005/), [block_006](https://open.quiltdata.com/b/aind-benchmark-data/tree/3d-image-compression/blocks/block_006/), [block_007](https://open.quiltdata.com/b/aind-benchmark-data/tree/3d-image-compression/blocks/block_007/), [block_008](https://open.quiltdata.com/b/aind-benchmark-data/tree/3d-image-compression/blocks/block_008/), [block_009](https://open.quiltdata.com/b/aind-benchmark-data/tree/3d-image-compression/blocks/block_009/); submit your results in a single ZIP archive with the following files:

- **Compressed Images**
  - Format: Any
  - Filename: `compressed_{num}.{extension}`
 
- **Segmentations**
  - Format: `.tiff`
  - Filename: `segmentation_{num}.tiff`
  - Generate segmentations using the provided model ([download here](insert link)).  
  - Use the [aind-exaspim-neuron-segmentation](https://github.com/AllenNeuralDynamics/aind-exaspim-neuron-segmentation) repository to compute affinity maps, and convert them to segmentations by following the *Predict* section of the README with default `inference.predict` parameters.

- **SWCs**
  - Format: `.zip`
  - Filename: `skeletons_{num}.zip`
  - Must be generated from the segmentations by following the final step in the “Predict” section of the repository’s README.


**Example of Submission Layout**
```text
  submission.zip
  ├── compressed_005.zarr
  ├── segmentation_005.tiff
  ├── skeletons_005.zip
  ├── ...
  ├── compressed_009.zarr
  ├── segmentation_009.tiff
  └── skeletons_009.zip
```

## Score Submission
Add description...

```python
from image_compression_challenge.score import score

# Initializations
submission_zip_path = "path-to-submission-zip"
read_compressed = "function-to-read-compressed-image"

# Main
score(submission_zip_path, read_compressed)

```

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
image-compression-challenge is licensed under the MIT License.
