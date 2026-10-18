"""
Panorama stitching, stage one: find the homography between two overlapping photos.

    SIFT keypoints  ->  matches (ratio test)  ->  RANSAC by hand  ->  compare with OpenCV

CSC 340, spring 2025. RANSAC is written out rather than called from OpenCV, then checked against
cv2.findHomography on the same matches.

Revised 2026-10-18. The first version scored each RANSAC round by the total reprojection distance
over every match, so a few badly wrong matches could outvote many good ones and drag the result
off. Now each round is scored by how many matches it explains (inliers within a few pixels), and
the winner is refitted on all of its inliers at once. Points are normalised before solving, which
keeps the SVD well conditioned when coordinates run into the hundreds.

Usage:
    python PA6StitchingMontgomeryDimitri.py [pair]        # pair: waterfall, mountain, florence, queenstown

The image pairs came with the course and are not in this repository; put them next to the script.
"""

import random
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

PAIRS = {
    "waterfall": ("waterfall1.jpg", "waterfall2.jpg"),
    "mountain": ("mountainLeft.jpg", "mountainRight.jpg"),
    "florence": ("florenceLeft.JPG", "florenceRight.JPG"),
    "queenstown": ("queenstownLeft.JPG", "queenstownRight.JPG"),
}

RATIO = 0.7  # keep a match only if it is clearly better than the runner-up
ITERATIONS = 25_000  # the first report found 20,000 to 50,000 gave a stable matrix
INLIER_PX = 3.0  # a match "agrees" with a homography if it lands within this many pixels


# ── matching ──────────────────────────────────────────────────────────────────────────────


def load_pair(name):
    """Both images of a pair, in greyscale."""
    left, right = PAIRS[name]
    img1 = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        sys.exit(f"Could not load {left} and {right}. Put them next to this script.")
    return img1, img2


def find_matches(img1, img2):
    """SIFT keypoints in both images, and the matches that pass the ratio test.

    Returns the two keypoint lists, the good matches, and the matched points as two N x 2 arrays
    (img1 coordinates, img2 coordinates).
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    good = []
    for best, second in cv2.BFMatcher().knnMatch(des1, des2, k=2):
        if best.distance < RATIO * second.distance:
            good.append(best)

    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
    return kp1, kp2, good, pts1, pts2


# ── the homography ────────────────────────────────────────────────────────────────────────


def normalise(pts):
    """Move the points' centre to the origin and scale them to an average distance of sqrt(2).

    Returns the normalised points and the 3 x 3 matrix that did it, so the result can be undone.
    """
    centre = pts.mean(axis=0)
    spread = np.sqrt(((pts - centre) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / spread if spread > 0 else 1.0
    T = np.array([[s, 0, -s * centre[0]], [0, s, -s * centre[1]], [0, 0, 1]])
    return (pts - centre) * s, T


def homography(src, dst):
    """The homography taking src to dst, from 4 or more point pairs.

    Two rows per pair in A, then the right singular vector for the smallest singular value is
    the answer: exactly the 8 x 9 system of the first version when there are four pairs, and a
    least-squares fit when there are more. None if the points are degenerate.
    """
    src_n, T1 = normalise(src)
    dst_n, T2 = normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, Vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def project(H, pts):
    """Apply H to an N x 2 array of points."""
    ones = np.ones((len(pts), 1))
    mapped = np.hstack([pts, ones]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / mapped[:, 2:3]


def ransac(pts1, pts2, iterations=ITERATIONS, seed=None):
    """RANSAC for the homography from pts1 to pts2.

    Each round fits 4 random matches and counts how many of all the matches land within
    INLIER_PX of where that homography sends them. The round that explains the most matches wins
    (ties go to the smaller error over those inliers), and the winner is refitted on all of its
    inliers. Returns the homography and a boolean mask of the inliers.
    """
    rng = random.Random(seed)
    n = len(pts1)
    if n < 4:
        sys.exit(f"Only {n} matches passed the ratio test; a homography needs at least 4.")

    best_count, best_error, best_mask = -1, np.inf, None
    for _ in range(iterations):
        pick = rng.sample(range(n), 4)
        H = homography(pts1[pick], pts2[pick])
        if H is None:
            continue
        distance = np.linalg.norm(project(H, pts1) - pts2, axis=1)
        mask = distance < INLIER_PX  # NaN (a point sent to infinity) compares False
        count = int(mask.sum())
        error = distance[mask].sum()
        if count > best_count or (count == best_count and error < best_error):
            best_count, best_error, best_mask = count, error, mask

    return homography(pts1[best_mask], pts2[best_mask]), best_mask


# ── run ───────────────────────────────────────────────────────────────────────────────────


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "waterfall"
    img1, img2 = load_pair(name)
    kp1, kp2, good, pts1, pts2 = find_matches(img1, img2)
    print(f"{name}: {len(kp1)} and {len(kp2)} keypoints, {len(good)} matches after the ratio test\n")

    mine, inliers = ransac(pts1, pts2)
    opencv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, INLIER_PX)

    np.set_printoptions(precision=4, suppress=True)
    print(f"Mine ({ITERATIONS:,} rounds, {inliers.sum()} of {len(good)} matches agree):\n{mine}\n")
    print(f"OpenCV:\n{opencv}\n")
    print(f"OpenCV - mine:\n{opencv - mine}\n")

    # Where the two disagree, in pixels, over the matches mine kept: easier to read than a matrix.
    gap = np.linalg.norm(project(mine, pts1[inliers]) - project(opencv, pts1[inliers]), axis=1)
    print(f"The two homographies place the inliers {gap.mean():.2f} px apart on average.")

    # Save rather than show: plt.show() freezes on macOS.
    kept = [m for m, keep in zip(good, inliers) if keep]
    drawn = cv2.drawMatches(img1, kp1, img2, kp2, kept, None, flags=2)
    plt.title(f"{name}: matches that agree with the homography")
    plt.imshow(drawn, cmap="gray")
    plt.axis("off")
    plt.savefig("final_image.png", bbox_inches="tight", dpi=150)
    print("Saved final_image.png")


if __name__ == "__main__":
    main()
