"""
Checks the RANSAC against a pair of images whose true homography is known.

The pair is drawn here rather than loaded: random shapes on a canvas, the left photo cut from it,
the right photo taken through a known homography and made 13% brighter. So the test needs no image
files, and it can say exactly how far the recovered homography is from the truth.

    python -m pytest test_ransac.py       or       python test_ransac.py
"""

import cv2
import numpy as np

import PA6StitchingMontgomeryDimitri as stitching


def make_pair(seed=7):
    """Two overlapping 640 x 500 views of one drawn scene, and the true homography left -> right."""
    rng = np.random.default_rng(seed)
    scene = np.full((500, 1100, 3), 200, np.uint8)
    for _ in range(900):
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, 1100)), int(rng.integers(0, 500))
        shape = rng.integers(0, 3)
        if shape == 0:
            cv2.circle(scene, (x, y), int(rng.integers(3, 25)), colour, -1)
        elif shape == 1:
            cv2.rectangle(scene, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 40))), colour, -1)
        else:
            cv2.line(scene, (x, y), (x + int(rng.integers(-60, 60)), y + int(rng.integers(-60, 60))), colour, 2)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)

    corners = np.float32([[0, 0], [640, 0], [640, 500], [0, 500]])
    in_scene = np.float32([[430, 12], [1090, -6], [1096, 492], [424, 486]])  # the right camera, turned
    right_to_scene = cv2.getPerspectiveTransform(corners, in_scene)
    truth = np.linalg.inv(right_to_scene)

    left = cv2.cvtColor(scene[:, :640], cv2.COLOR_BGR2GRAY)
    right = cv2.warpPerspective(scene, truth, (640, 500))
    right = cv2.cvtColor(np.clip(right * 1.13, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return left, right, truth


def pixels_off(H, truth):
    """Mean distance between where H and the truth send points across the overlap."""
    grid = np.float64([[x, y] for x in range(440, 641, 40) for y in range(20, 481, 92)])
    return np.linalg.norm(stitching.project(H, grid) - stitching.project(truth, grid), axis=1).mean()


def test_recovers_the_true_homography():
    left, right, truth = make_pair()
    _, _, good, pts1, pts2 = stitching.find_matches(left, right)
    H, inliers = stitching.ransac(pts1, pts2, iterations=2_000, seed=1)
    assert inliers.sum() > 0.5 * len(good), "most matches on a clean pair should agree"
    assert pixels_off(H, truth) < 0.5


def test_holds_with_many_bad_matches():
    """A third of the matches replaced with random pairings: the inlier count should not budge."""
    left, right, truth = make_pair()
    _, _, _, pts1, pts2 = stitching.find_matches(left, right)
    rng = np.random.default_rng(3)
    bad = rng.choice(len(pts2), len(pts2) // 3, replace=False)
    pts2 = pts2.copy()
    pts2[bad] = rng.uniform([0, 0], [640, 500], (len(bad), 2))
    H, _ = stitching.ransac(pts1, pts2, iterations=2_000, seed=1)
    assert pixels_off(H, truth) < 0.5


def test_four_points_give_the_exact_homography():
    truth = np.array([[1.05, 0.01, -450.0], [0.03, 1.11, -26.0], [1e-4, 0.0, 1.0]])
    src = np.float64([[10, 20], [600, 40], [580, 470], [30, 450]])
    H = stitching.homography(src, stitching.project(truth, src))
    assert np.allclose(H, truth, atol=1e-6)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
