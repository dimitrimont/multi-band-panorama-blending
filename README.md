# Panorama stitching: SIFT, hand-written RANSAC, homography

Coursework for CSC 340 (Computer Vision), spring 2025. Two overlapping photos are joined into one
by finding the points they share and solving for the homography between them, with RANSAC written
by hand rather than called from OpenCV.

**Live walkthrough:** https://dimitri-montgomery-site.pages.dev/vision/panorama-stitching/ runs the
same pipeline in the browser on a drawn scene, step by step.

## What is here

| File | What it is |
|---|---|
| `PA6StitchingMontgomeryDimitri.py` | The pipeline: SIFT keypoints, matches kept by a 0.7 ratio test, RANSAC by hand (25,000 rounds), and a comparison with OpenCV's `findHomography` |
| `test_ransac.py` | Checks the RANSAC on a drawn pair whose true homography is known, including with a third of the matches replaced by random ones |
| `PA6_Stitching_Report_MontgomeryDimitri.pdf` | Report on the matching and RANSAC stage |
| `340_FinalPA_Stiching_montgomery.pdf` | Final report, with five stitched panoramas |
| `blending_assignment.py` | The course's starter file for multi-band blending. **Not completed**: every step is still `...` |
| `blendingSteps.pdf` | The course handout describing multi-band blending |

## How the RANSAC works

Each round picks 4 random matches, normalises them, builds the 8 × 9 system (two rows per match)
and takes the right singular vector for the smallest singular value as the homography. It then
counts how many of **all** the matches land within 3 px of where that homography sends them. The
round that explains the most matches wins, and its homography is refitted on all of those inliers
at once.

### What changed on 2026-10-18

The first version scored each round by the **total** distance over every match, outliers
included, and kept the round with the smallest total. Under a total, a few badly wrong matches
outweigh many good ones, so the winner was a compromise with the outliers. It also kept the best
four-point guess rather than refitting, and solved on raw pixel coordinates.

Measured on the drawn pair in `test_ransac.py`, with real SIFT matches (417 after the ratio test):

| | Off from the true homography | Run time |
|---|---|---|
| First version (total distance) | 10.06 px | 60 s |
| Now (inlier count, refit, normalised) | 0.03 px | 4 s |
| OpenCV `findHomography` | 0.05 px | |

The speed-up is from scoring all matches at once with NumPy instead of a Python loop per match.

## Not finished

- Multi-band blending (`blending_assignment.py` is the unfilled starter).
- The warping and stitching code behind the final report's panoramas is not in this repository.

## Run

```bash
pip install opencv-python numpy matplotlib pytest
python -m pytest test_ransac.py                      # no images needed
python PA6StitchingMontgomeryDimitri.py waterfall    # or mountain, florence, queenstown
```

The script expects the image pairs named in `PAIRS` (for example `waterfall1.jpg` and
`waterfall2.jpg`) next to it; they came with the course and are not included. It prints both
homographies, their difference, and how far apart they place the inliers, and saves the matches
that agree as `final_image.png`.
