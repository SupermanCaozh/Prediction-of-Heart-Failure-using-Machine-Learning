# Prediction-of-Heart-Failure-using-Machine-Learning
Heart failure prediction task - comparison of four machine learning methods

## Brief Introduction
This project aims to train a machine learning model that produces heart failure predictions using lifestyle, nutrition
habits, blood test results and some other basic health indicators while taking advantages of features extracted from each patient's 28 by 28 pixel heart scans. A following figure below shows the mechanism.
![image](https://github.com/SupermanCaozh/Prediction-of-Heart-Failure-using-Machine-Learning/assets/96049887/8090326a-3310-4723-b171-f74884b4fdf9)

### Data Set
The folder *data* contains all the essential data files you need. The file Xtab1.csv contains medical data stored as a table. Each line/row corresponds to a patient and each column to a measured attribute/variable/feature. A description of each variable is given below:
|age |Age (let's suppose that the age can be well over 100 for this particular nation :))
|blood pressure |Systolic blood pressure (in mmHg)
|blood type |Blood type (antigens and rhesus)
|cholesterol |Level of LDL cholesterol (”bad cholesterol”) in blood (in mg/dL)
|hemoglobin |Level of hemoglobin in blood (in g/dL)
|physical activity |If the patient practices a physical activity on a regular basis (yes - no)
|sarsaparilla |Consumption of sarsaparilla leaves (very low - low - moderate - high - very high)
|liquor |Consumption of liquor (very low - low - moderate - high - very high)
|donuts |Consumption of donuts (very low - low - moderate - high - very high)
|temperature |Body temperature at the time of the visit by Doctor Smurf (in °C)
|testosterone |Level of testosterone in blood (in ng/dL)
|weight |Body mass (in grams)

The risk of developping a heart failure within the next ten years is the target variable; it is stored
in the Y1.csv file. The indices match those of Xtab1.csv.

The last element of each line in Xtab1.csv is the name of the image file that contains the heart scan. These images are stored in the folder Img1. For convenience, we take advantage of image embeddings (i.e., low-dimensional vector representations of the images). These are stored in the Ximg1.csv file; each element of an embedding vector can be considered as an additional numerical feature. The image embeddings were obtained by training a convolutional neural network and extracting the output values from its last (fully connected) hidden layer. If you want to perform this feature extraction step on your own (for example, in order to improve it), some code is provided in *feature extraction.ipynb* to modify for better results. 

## Results
The codes provided in *main_comparison.ipynb* implements four well-known machine learning techniques, including Linear Model, Multi Layer Perceptron, KNN and Support Vector Machine/Regression, with data engineering, feature selection and model selection done in prior. The MLP demonstrate best prediction results.
