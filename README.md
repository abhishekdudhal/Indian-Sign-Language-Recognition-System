# Indian-Sign-Language-Recognition-System
## Hardware Requirements
• 4 GB Ram
• 1 GB Free Space
• Web Cam (5 MP preferable)
## Software Installation Requirements
• Python 2.7.13
• OPENCV 2.4.8
• Keras 2.0.2
• Theano 0.9.0
## How to Use System?
### 1. Creation of the Data-set:
Step 1: Open the terminal and move to the project folder
Step 2: Run the command KERAS_BACKEND=theano python cnnCreateDataSet.py
Step 3: Press key n to capture image for the given sign
Step 4: When system says “Change gesture” then change the gesture
### 2. Training of Model
Step 1: Open the terminal and move to the project folder
Step 2: Run the command KERAS_BACKEND=theano python cnnTrain.py
Step 3: As per menu shown on terminal select the data filter on which you want to train the CNN model
Step 4: Once the training is completed, a message "Model trained successfully"  will be shown on the terminal
### 3. Prediction of Sign
Step 1: Open the terminal and move to the project folder
Step 2: Run the command KERAS_BACKEND=theano python main.py
Step 3: Follow the menu shown on the terminal to adjust the region of gesture (Green box)
Step 4: Once the Region of Gesture is fixed press key 'P' to start prediction 
Step 5: System will show predicted sign on display. For audio format, enable sound. 
 



