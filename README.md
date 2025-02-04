# IFT6390B Kaggle 2

https://www.kaggle.com/competitions/ift3395-ift6390-identification-maladies-retine 

## Description

This competition aims to identify various retinal diseases present in optical coherence tomography (OCT) images. The data describes four diagnostic types (classes):

0: choroidal neovascularization
1: diabetic macular edema
2: drusen
3: healthy retina
This is a multi-class image classification problem (4 classes). We have a large training dataset (~97,000 images), each sized 28x28 pixels. For relatively simple methods, feel free to use a subset of the training data. Your task is to develop a classifier capable of identifying one of the four mentioned diagnoses for a given OCT image.

## Important Milestones

X November: Team Formation
Teams are formed to participate in the competition.

X November:
To earn full points, your model must perform at least as well as the three classifiers listed on the Leaderboard:

## Random Prediction

Logistic Regression
SVM with an RBF kernel
Additionally, you must submit your code on Gradescope.

December 3: End of Competition
The competition closes, and all submissions must be finalized.

December 3: Report Submission Deadline
Teams must submit their reports.

## Evaluation

To evaluate your model, you must create a CSV file containing predictions for the examples in test_data.pkl. The format is as follows:

The ID of the first prediction should be 1, not 0.
The metric used for evaluation is prediction accuracy (proportion of correct predictions).

## Final Work
### Numpy only implementation of CNN 
Milestone_1_CNN file milestone_1_NN_original_overtrain
Accuracy of 65%

### Package allowed implementation of CNN
milestone_2_NN_V5
Accuracy of 85%
