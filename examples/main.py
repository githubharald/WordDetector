import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from word_detector import detect, prepare_img, sort_multiline


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/line'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=50)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = prepare_img(cv2.imread(fn_img), parsed.img_height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

        plt.show()


if __name__ == '__main__':
    main()
