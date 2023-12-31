# Computer-Vision

The projects undertaken within the scope of the "Computer Vision" course are presented in this repository.

# 1st Project - Handling Salt and Pepper Noise:

There are given two images, one without noise and one with noise. After removing the noise from the second image, the following calculations are requested:
- The number of objects 
- The area of each object and the bounding box surrounding it (sum of pixels)
- The average grayscale intensity in a way that the execution speed is independent of the object size

According to the exercise, for the denoising of the original image, no function from the OpenCV library was used, but I developed my own that implements the median filter.

For calculating the average grayscale intensity, I implemented the integral algorithm myself.

# 2nd Project - Panorama Creation:

Four initial images are given, and the task is to generate their panorama. The algorithm I will develop will also be applied to the four images I capture myself. The results will be compared with those of a panorama creation tool.
For the development of the algorithm and feature extraction, the SIFT and SURF algorithms were used.

Note that for merging the images, in order to achieve a result close to that of the corresponding tool, I apply the homography matrix to a different image each time. During the merging of the first two images, the transformations are applied to the first image, while during the merging of the last two images, the transformations are applied to the second image.


# 3rd Project - Classification with Bag Of Words model and K-NN and SVM classifiers:

The program is implemented by performing the following steps:
1) Visual vocabulary production based on the Bag of Visual Words model (BOVW). The creation of the dictionary should be done using the K-Means algorithm using all images in the training set (imagedb_train).
2) Extract a descriptor on each training image (imagedb_train) based on BOVW model using the dictionary generated in previous step.
3) Based on these results, implement the classification function of an image using two classifiers:
         k-NN algorithm (without using the associated OpenCV function) 
         one-versus-all method where for each class an SVM classifier is trained.
5) System evaluation: Using the test set (imagedb_test) measure the accuracy of the system (in both cases of classifiers) expressed as the percentage of successful classifications, overall and by class. The results are saved in a separate .txt file. 

There is also an Excel file that presents the results in charts.


# 4th Project - Classification with Neural Networks:

A Neural Network was developed, and a pre-trained neural network was appropriately modified.
For both networks, data augmentation was implemented on the training set using suitable transformations, early stopping, and storing the weight values for the best result.
