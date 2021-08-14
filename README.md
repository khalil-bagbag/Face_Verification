# Django- Face Recognition
## An AI service that analyzes faces in images
#### Apply facial recognition for a range of scenarios Detect, identify, and analyze faces in images. 


## Introduction
This project aims :
-To classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML).
-Check the likelihood that two faces belong to the same person and receive a confidence score.
-Check if a person is wearing a mask or not.
-Detecting attributes belonging to a face. Example attributes are the color of hair, hairstyle, age, gender, etc.

## Folders

| Folder | Content |
| ------ | ------ |
| Models | Contains the 4 models of our application  |
| Notebook |Contains data preparation and training of the emotion deception model |
| face_emotion | Django App of the solution |

## Models

| Models | Infos |
| ------ | ------ |
| Emotion Recognition | CNN.48-0.66.hdf5 with 0.66 accuracy  |
| Attributes |  ModelCNN.json with 0.9095 |
| Mask detection | mask_detector.model with accuracy |
| Face Verificatiom | keras facenet (pip install keras-facenet |


## Screen_Shots

![alt text](https://github.com/khalil-bagbag/Face_Verification/blob/main/Screen_Shots/Attributes.png)
![alt text](https://github.com/khalil-bagbag/Face_Verification_git/blob/main/Screen_Shots/emotion.png)
![alt text](https://github.com/khalil-bagbag/Face_Verification_git/blob/main/Screen_Shots/mask.png)
![alt text](https://github.com/khalil-bagbag/Face_Verification_git/blob/main/Screen_Shots/verification.png)
