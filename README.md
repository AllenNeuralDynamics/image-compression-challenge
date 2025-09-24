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
  - Filename: `{block_id}.zarr`
 
- **Segmentations**
  - Format: `.tiff`
  - Filename: `{block_id}.tiff`
  - Must be generated using the provided segmentation model that can be downloaded [insert link]().
  - Use the [aind-exaspim-neuron-segmentation](https://github.com/AllenNeuralDynamics/aind-exaspim-neuron-segmentation) repository to generate segmentations. Follow the steps in the “Predict” section of its README to run the model.

- **SWCs**
  - Format: `.zip`
  - Filename: `{block_id}.zip`

## Score Submission
Add description...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
supervoxel-loss is licensed under the MIT License.
