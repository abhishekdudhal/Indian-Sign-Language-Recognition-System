# Indian-Sign-Language-Recognition-System
Indian sign language (ISL) is one of the most used sign languages in the Indian subcontinent. This research aims at developing a simple Indian sign language recognition system based on convolutional neural network (CNN). The proposed system needs webcam and laptop and hence can be used anywhere. CNN is used for image classification. Scale invariant feature transformation (SIFT) is hybridized with adaptive thresholding and Gaussian blur image smoothing for feature extraction. Due to unavailability of ISL dataset, a dataset of 5000 images, 100 images each for 50 gestures, has been created. The system is implemented and tested using python-based library Keras. The proposed CNN with hybrid SIFT implementation achieves 92.78% accuracy, whereas the accuracy of 91.84% was achieved for CNN with adaptive thresholding.
## Indian Sign Language Dataset 
Due to unavailability of Indian Sign language dataset (ISL), We have created dataset of 5000 images. Data set contains 100 images each for 50 Indian signs. Improvements, suggestions and accuracy statistics on your algorithms are welcomed. Dataset can be downloaded from https://drive.google.com/drive/folders/1mHmmmSaU5ZV8QKIUSCF0fabVv54HhxWq?usp=sharing
## Hardware Requirements
• 4 GB Ram<br/>
• 1 GB Free Space<br/>
• Web Cam (5 MP preferable)<br/>
## Software Installation Requirements
• Python 2.7.13<br/> 
• OPENCV 2.4.8<br/> 
• Keras 2.0.2<br/>
• Theano 0.9.0<br/>
## How to Use System?
### 1. Creation of the Data-set:
Step 1: Open the terminal and move to the project folder.<br/>
Step 2: Run the command KERAS_BACKEND=theano python cnnCreateDataSet.py.<br/>
Step 3: Press key n to capture image for the given sign.<br/>
Step 4: When system says “Change gesture” then change the gesture.<br/>
### 2. Training of Model
Step 1: Open the terminal and move to the project folder.<br/>
Step 2: Run the command KERAS_BACKEND=theano python cnnTrain.py.<br/> 
Step 3: As per menu shown on terminal select the data filter on which you want to train the CNN model.<br/>
Step 4: Once the training is completed, a message "Model trained successfully"  will be shown on the terminal.<br/>
### 3. Prediction of Sign
Step 1: Open the terminal and move to the project folder.<br/>
Step 2: Run the command KERAS_BACKEND=theano python main.py.<br/>
Step 3: Follow the menu shown on the terminal to adjust the region of gesture (Green box).<br/>
Step 4: Once the Region of Gesture is fixed press key 'P' to start prediction.<br/>
Step 5: System will show predicted sign on display. For audio format, enable sound drivers.<br/>    
 
DOI : https://doi.org/10.1007/978-3-030-00665-5_72
