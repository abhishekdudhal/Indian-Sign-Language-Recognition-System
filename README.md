# Indian-Sign-Language-Recognition-System
## Hardware Requirements
• 4 GB Ram <br/>  
• 1 GB Free Space <br/>  
• Web Cam (5 MP preferable) <br/>
## Software Installation Requirements
• Python 2.7.13 <br/> 
• OPENCV 2.4.8 <br/> 
• Keras 2.0.2  <br/>
• Theano 0.9.0  <br/>
## How to Use System?
### 1. Creation of the Data-set:
Step 1: Open the terminal and move to the project folder. <br/> 
Step 2: Run the command KERAS_BACKEND=theano python cnnCreateDataSet.py.  <br/>  
Step 3: Press key n to capture image for the given sign.  <br/>
Step 4: When system says “Change gesture” then change the gesture.  <br/>
### 2. Training of Model
Step 1: Open the terminal and move to the project folder. <br/>   
Step 2: Run the command KERAS_BACKEND=theano python cnnTrain.py. <br/>    
Step 3: As per menu shown on terminal select the data filter on which you want to train the CNN model. <br/>    
Step 4: Once the training is completed, a message "Model trained successfully"  will be shown on the terminal. <br/>   
### 3. Prediction of Sign
Step 1: Open the terminal and move to the project folder. <br/>    
Step 2: Run the command KERAS_BACKEND=theano python main.py. <br/>   
Step 3: Follow the menu shown on the terminal to adjust the region of gesture (Green box). <br/>   
Step 4: Once the Region of Gesture is fixed press key 'P' to start prediction. <br/>    
Step 5: System will show predicted sign on display. For audio format, enable sound. <br/>    
 



