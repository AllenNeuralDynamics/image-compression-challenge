# Image Compression Challenge

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Add description...

## Prepare Submission

Your submission must be a single ZIP archive with the following structure:
- **Compressed Images**
  - Format: `.zarr`
  - Filename: `block_{block_id}.zarr`
 
- **Segmentations**
  - Format: `.tiff`
  - Filename: `segmentation_{block_id}.tiff`
  - Generate segmentations using the provided model ([download here](insert link)).  
  - Use the [aind-exaspim-neuron-segmentation](https://github.com/AllenNeuralDynamics/aind-exaspim-neuron-segmentation) repository to compute affinity maps, and convert them to segmentations following the *Predict* section of the README with default `inference.predict` parameters.

- **SWCs**
  - Format: `.zip`
  - Filename: `skeletons_{block_id}.zip`
  - Must be generated from the segmentations by following the final step in the “Predict” section of the repository’s README.


**Example of Submission Layout**
```text
  submission.zip
  ├── block_005.zarr
  ├── segmentation_005.tiff
  ├── skeletons_005.zip
  ├── ...
  ├── block_009.zarr
  ├── segmentation_009.tiff
  └── skeletons_009.zip
```

## Score Submission
Add description...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
supervoxel-loss is licensed under the MIT License.
