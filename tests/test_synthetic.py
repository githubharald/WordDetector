import numpy as np

from word_detector import detect, prepare_img, sort_multiline, BBox


def test_synthetic():
    """A synthetic test that checks the core functionality of the package."""

    # ground truth word bounding boxes, and ground truth sorting into lines
    gts = [BBox(100, 100, 100, 25), BBox(300, 110, 50, 15), BBox(100, 300, 50, 20)]
    gt_lines = [[gts[0], gts[1]], [gts[2]]]

    # draw synthetic image
    img = np.ones([512, 512], np.uint8) * 255
    for gt in gts:
        img[gt.y:gt.y + gt.h, gt.x: gt.x + gt.w] = 0

    # detect words
    img = prepare_img(img, 512)
    detections = detect(img, kernel_size=25, sigma=25, theta=5, min_area=100)

    # check if all words detected
    assert len(detections) == len(gts)

    # sort detected words into lines
    lines = sort_multiline(detections, min_words_per_line=1)

    # check if number of lines correct
    assert len(lines) == len(gt_lines)

    # go over all lines
    for det_line, gt_line in zip(lines, gt_lines):
        # check that number of words per line matches
        assert len(det_line) == len(gt_line)

        # go over all words
        for det_word, gt_word in zip(det_line, gt_line):
            # check that bounding boxes are similar (within +-10px)
            thres = 10
            assert abs(det_word.bbox.x - gt_word.x) < thres
            assert abs(det_word.bbox.y - gt_word.y) < thres
            assert abs(det_word.bbox.w - gt_word.w) < thres
            assert abs(det_word.bbox.h - gt_word.h) < thres
