# Multi-Band Image Stitching & Panorama Blending

This project builds a full panorama pipeline from scratch. It includes keypoint matching, homography estimation, and multi-band blending for seamless stitching.

## Features
- SIFT-based feature detection and matching
- Homography estimation with RANSAC
- Spherical weighting for pixel importance
- Multi-band Gaussian blending with 3 levels
- Reconstructs high-quality panoramic scenes

## Techniques
- Difference of Gaussians (DoG)
- Blending weight pyramids
- Seamless pixel-wise merging

## Run
```bash
python blending_assignment.py
