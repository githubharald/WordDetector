# Word Segmentation with Scale Space Technique

**Update 2021: installable Python package, added line clustering and word sorting**

Implementation of the scale space technique for word segmentation proposed by 
[R. Manmatha and N. Srimal](http://ciir.cs.umass.edu/pubfiles/mm-27.pdf). 
Even though the paper is from 1999, the method still achieves good results, is fast, and has a simple implementation. 
The algorithm takes an **image containing words as input** and **outputs the detected words**.
Optionally, the words are sorted according to reading order (top to bottom, left to right).

![example](./doc/example.png)

## Installation

* Go to the root level of the repository
* Execute `pip install .`
* Go to `tests/` and execute `pytest` to check if installation worked

## Usage

This example loads an image of a text line, prepares it for the detector (1), detects words (2), 
sorts them (3), and finally shows the cropped words (4).

````python
from word_detector import prepare_img, detect, sort_line
import matplotlib.pyplot as plt
import cv2

# (1) prepare image:
# (1a) convert to grayscale
# (1b) scale to specified height because algorithm is not scale-invariant
img = prepare_img(cv2.imread('data/line/0.png'), 50)

# (2) detect words in image
detections = detect(img,
                    kernel_size=25,
                    sigma=11,
                    theta=7,
                    min_area=100)

# (3) sort words in line
line = sort_line(detections)[0]

# (4) show word images
plt.subplot(len(line), 1, 1)
plt.imshow(img, cmap='gray')
for i, word in enumerate(line):
  print(word.bbox)
  plt.subplot(len(line), 1, i + 2)
  plt.imshow(word.img, cmap='gray')
plt.show()
````

The repository contains some examples showing how to use the package:
* Install requirements: `pip install -r requirements.txt`
* Go to `examples/`
* Run `python main.py` to detect words in line images (IAM dataset)
* Or, run `python main.py --data ../data/page --img_height 1000 --theta 5` to run the detector on an image of a page (also from IAM dataset)


The package contains the following functions:
* `prepare_img`: prepares input image for detector
* `detect`: detect words in image
* `sort_line`: sort words in a (single) line
* `sort_multiline`: cluster words into lines, then sort each line separately

For more details on the functions and their parameters use `help(function_name)`, e.g. `help(detect)`.


## Algorithm

The illustration below shows how the algorithm works:

* top left: input image
* top right: apply filter to the image
* bottom left: threshold filtered image
* bottom right: compute bounding boxes

![illustration](./doc/illustration.png)

The filter kernel with size=25, sigma=5 and theta=3 is shown below on the left. 
It models the typical shape of a word, with the width larger than the height (in this case by a factor of 3). 
On the right the frequency response is shown (DFT of size 100x100). 
The filter is in fact a low-pass, with different cut-off frequencies in x and y direction.
![kernel](./doc/kernel.png)


## How to select parameters

* The algorithm is **not scale-invariant**
    * The default parameters give good results for a text height of 25-50 pixels
    * If working with lines, resize the image to 50 pixels height
    * If working with pages, resize the image so that the words have a height of 25-50 pixels
* The sigma parameter controls the width of the Gaussian function (standard deviation) along the x-direction. Small
  values might lead to multiply detection per word (over-segmentation), while large values might lead to a detection
  containing multiple words (under-segmentation)
* The kernel size depends on the sigma parameter and should be chosen large enough to contain as much of the non-zero
  kernel values as possible
* The average aspect ratio (width/height) of the words to be detected is a good initial guess for the theta parameter

The best way to find the optimal parameters is to use a dataset (e.g. IAM) and optimize the parameters w.r.t. some
evaluation metric (e.g. intersection over union).

## Results

This algorithm gives good results on datasets with large inter-word-distances and small intra-word-distances like IAM.
However, for historical datasets like Bentham or Ratsprotokolle results are not very good and more complex approaches
should be preferred (e.g., a neural network based approach as implemented in
the [WordDetectorNN](https://github.com/githubharald/WordDetectorNN) repository).
