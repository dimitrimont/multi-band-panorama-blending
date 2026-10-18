# Panorama stitching: SIFT, hand-written RANSAC, homography

Coursework for CSC 340 (Computer Vision), spring 2025. Two overlapping photos are joined into one
by finding the points they share and solving for the homography between them, with RANSAC written
by hand rather than called from OpenCV.

**Live walkthrough:** https://dimitri-montgomery-site.pages.dev/vision/panorama-stitching/ runs the
same pipeline in the browser on a drawn scene, step by step.

## What is here

| File | What it is |
|---|---|
| `PA6StitchingMontgomeryDimitri.py` | The working pipeline: SIFT keypoints, matching with a 0.7 ratio test, and RANSAC by hand (25,000 rounds, each a 4-point homography solved by SVD), compared against OpenCV's `findHomography` |
| `PA6_Stitching_Report_MontgomeryDimitri.pdf` | Report on the matching and RANSAC stage |
| `340_FinalPA_Stiching_montgomery.pdf` | Final report, with five stitched panoramas |
| `blending_assignment.py` | The course's starter file for multi-band blending. **Not completed**: every step is still `...` |
| `blendingSteps.pdf` | The course handout describing multi-band blending |

## How the RANSAC works

Each round picks 4 random matches, builds the 8 × 9 system (two rows per match), and takes the
right singular vector for the smallest singular value as the homography. A round is scored by the
**total** reprojection distance over every match, and the lowest total wins. The program then
prints OpenCV's homography for the same matches and the difference between the two.

Scoring by total distance lets outliers vote: one badly wrong match can outweigh many good ones.
The textbook rule, counting matches that land within a few pixels, holds up better; the live
walkthrough runs both on the same samples.

## Not finished

- Multi-band blending (`blending_assignment.py` is the unfilled starter).
- The warping and stitching code behind the final report's panoramas is not in this repository.

## Run

```bash
pip install opencv-python numpy matplotlib
python PA6StitchingMontgomeryDimitri.py
```

The script expects the image pairs named in its `images` dictionary (for example `waterfall1.jpg`
and `waterfall2.jpg`) next to it; they came with the course and are not included. It prints both
homographies and their difference, and saves the matches as `final_image.png`.
