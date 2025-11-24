# CVProject

To run the program, there are two approaches:

  Pre-trained: The hand_model.h5 is pre-trained by us and used a manual dataset consisting of roughly 300 images per hand gesture. If you choose to attempt and run the classifier/predictor, namely the predict_live.py file, it will predict only hand gestures which we have trained it. We have 5 classificiations of hand gestures that the model can detect. This is a quick approach for you to test the functionality and accuracy of this model.

  To be trained: You are able to train the model with any hand gesture you would like. However, this is a more lengthy appraoch as each hand gesture requires roughly 300 images for sufficient level of accuracy. To do so, run the dataCollection.py script, which should detect and crop the full camera image into a smaller image which tracks only your hand. This is part of our pre-processing. The file path is set by default to folder 1, where we have up to 5 folders. The user can manually add more folders and classes if wanted, or less (not recommended). To collect data through the dataCollection.py script:
  1. Press S to take an image and save it into the path folder (default is folder 1)
  2. Keep taking images of your hand until you have 300 images, a counter will be printed for you.
  3. Ensure that the 300 images in one folder share the same gesture, however, at different distances from camera and varying backgrounds will have the best results.
  4. Press 1-5 to move to the specified folder, example folder 2 by pressing 2
  5. Take another set of 300 images by pressing S in another folder, this will be another gesture that the model will be able to recognize
  6. Ensure different gestures are in different folders, the same gesture should be in the same folder
  7. After all images of differnet gestures have been taken and stored in distinct folders (such as folders 1-5, therefore, 5 gestures), run train.py
  8. Finally, run predict_live.py which should allow you to test the prediction model 
