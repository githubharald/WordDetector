# Word segmentation
Implementation of scale space technique for word segmentation as proposed by [R. Manmatha and N. Srimal](http://ciir.cs.umass.edu/pubfiles/mm-27.pdf).
Even though the paper is from 1999, the method still achieves good results, is fast, and is easy to implement.
The algorithm takes an image of a line as input and outputs the segmented words.

![example](./doc/example.png)


## Run demo
Go to the `src/` directory and run the script `python main.py`.
The images from the `data/` directory (taken from IAM dataset) are segmented into words and the results are saved to the `out/` directory.

## Documentation
An anisotropic filter kernel is applied to the input image to create blobs corresponding to words.
After thresholding the blob-image, connected components are extracted which correspond to words.

### Parameters

Most of the parameters of the function `wordSegmentation` deal with the shape of the filter kernel:
* img: grayscale uint8 image of the text-line to be segmented.
* kernelSize: size of filter kernel, must be an odd integer.
* sigma: standard deviation of Gaussian function used for filter kernel.
* theta: approximated width/height ratio of words, filter function is distorted by this factor.
* minArea: ignore word candidates smaller than specified area.

The function `prepareImg` can be used to convert the input image to grayscale and to resize it to a fixed height:
* img: input image.
* height: image will be resized to fit specified height.


### Algorithm

The illustration below shows how the algorithm works: 

* top left: input image. 
* top right: filter kernel is applied.
* bottom left: blob image after thresholding.
* bottom right: bounding boxes around words in original image.

![illustration](./doc/illustration.png)

## Results
This algorithm gives good results on datasets with large inter-word-distances and small intra-word-distances like IAM.
However, for historical datasets like Bentham or Ratsprotokolle results are not very good and more complex approaches should be used instead.
