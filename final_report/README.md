# Gaze Detection: Final Report

## Abstract

Gaze detection is the process of identifying the direction of someone's gaze or where they are looking. For tackling this, we proposed a simple CNN model based on AlexNet architecture which takes an eye image as an input and predicts the gaze angle theta and phi. The output of the model, theta, and phi will be used to calculate the gaze direction and perform gaze detection. For the preprocessing of the data into the model, we performed face detection and landmark detection to crop the eye image using dlib library’s face detector and landmarks detection along with OpenCV. We trained three models with different hyperparameters. The best model gave satisfactory results with the validation MSE loss  for theta being  2.0583  and phi being 2.8460. The calculated gaze was then used to draw an arrow on the image to denote the direction the user is looking at. 

*Keywords: Gaze detection, AlexNet, dlib, OpenCV*

## Introduction

Gaze detection, a rapidly progressing area of computer vision and human-computer interaction, seeks to precisely track and decipher the direction of a person's gaze. The widespread use of devices like smartphones, virtual reality headsets, and smart glasses has further emphasized the significance of gaze detection for various applications, such as eye-tracking interfaces, attention-aware systems, and facial expression recognition. Eye tracking is still a novel technology that requires adequate computing resources. However, with recent advances in deep learning, high accuracy can be achieved without the need for complex hardware. 

Gaze detection offers a wide range of applications, including improving accessibility for people with disabilities and enhancing immersive experiences in gaming and virtual reality. Furthermore, it has promising implications for fields such as marketing research, psychology, and neuroscience, where it can offer valuable insights into human behavior and cognition. 

### Problem Statement

Gaze detection or estimation requires specialized software and hardware to work accurately, this can be highly expensive and bulky. Using simpler approach to perform gaze detection and estimation may cause low accuracy. 

The problem we aim to solve is to perform gaze detection without specialized hardware and software, using model based approach without compromising the accuracy obtained in the results. 

### Objectives

The objectives for the gaze detection system will be:

- To predict the direction the user is looking at.
- To learn the basics of CNN and Computer Vision Concepts.

## Literature Review

In the paper [1], the authors discuss the results of various models for the eye gaze estimation and predicting gaze detection using eye landmarks in unconstrained settings. The authors found out that in real-world settings model-based and feature-based approaches were out performed by appearance-based methods because of factors like illumination changes and other visual artifacts.

In the paper [2], the authors use VGG16 architecture of CNN to predict the pitch and yaw of the gaze vector with respect to camera from the normalized pre-processed images. The calculated gaze angles of the gaze vector is then passed through screen calibration techniques and the gaze is mapped into the 2D screen point, using geometry modeling and regression. The authors used MPII Face Gaze dataset and EYEDIAP dataset. 

In the paper [3], the authors propose GazeNet a capsule network based architecture capable of decoding, representing, and estimating gaze information from ocular region images. The authors used MPII Gaze dataset and Columbia Gaze where the authors divided the gaze into 21 gaze directions observed at 5 different camera angles. The model proposed obtained result of 2.84 for the combined error of the gaze angles for MPII Gaze dataset. And 10.04 for Columbia Gaze dataset, which was reduced to 5.9 after using transfer learning.

In the paper [4], the authors proposed a gaze estimation method for smart phones. 

In the paper [5], the authors propose a dataset for gaze detection and estimation. The dataset proposed is SynthesEyes, a synthetic dataset. It consists of close up eye images for wide range of head poses, gaze direction and illumination. Each of the image synthesized will be perfectly labelled and photorealistic, which will save significant amount of time compared to data collection and manual annotation.  

In the paper [6], the authors propose a dataset for gaze estimation. The dataset presented is MPII gaze dataset, which contains 213,659 images collected from 15 participants. The authors also propose an algorithm for gaze estimation. The authors used the LeNet architecture CNN. .In the LeNet model , the authors send the head pose angle and the extracted eye image from the Convolutional Layers. 

## Project Methodology

*Some Terminologies*

![gaze.PNG](Gaze%20Detection%20Final%20Report%202f917c433ed44904bcf0aa89694ca57a/gaze.png)

Fig: Gaze direction(g) with respect to the primary line of sight(z) is described by two angular coordinates $\theta \space and \space \phi$ [7].

Here are some terms to be familiar with: 

- Theta ($\theta$)
    
    Provided the gaze direction ‘**g’** and the projection of ‘**g**’ onto planes X–Z (nasal-temporal) is $g_x$. Then, $\theta$ is the angle between direction Z and vector $g_x$. Therefore $\theta$ quantifies the angular
    deviation of the gaze direction, in the horizontal direction, from a primary line of sight along Z. [7]
    
- Phi ($\phi$)
    
    Provided the gaze direction ‘**g’** and the projection of ‘**g**’ onto planes Y–Z (vertical) is $g_y$. Then, $\phi$ is the angle between direction Z and vector $g_y$. Therefore $\phi$ quantifies the angular
    deviation of the gaze direction, in the vertical direction, from a primary line of sight along Z.[7]
    

*Data Collection*

The dataset we will be using for our project is MPIIGaze which is a dataset for appearance-based gaze estimation in the wild. It contains 213,659 images collected from 15 participants during natural everyday laptop use over more than three months[6].

*Model Process*

The project pipeline starts from an input image or a frame from an input video. This frame goes through a face and landmark detection block by using the pre-trained model Dlib. 

Dlib uses HOG (Histogram of Oriented Gradients) and a linear SVM (Support Vector Machine) classifier to identify faces. Dlib also provides a pre-trained model for facial landmark detection. The facial landmark detector in Dlib is based on the shape model of facial landmarks and uses a combination of regression trees and a linear SVM classifier to locate the key points on a face. It detects the face and the 68 facial landmarks which can be used for the detection of eyes. 

Both of the eyes are detected and cropped. Eye images flipping horizontally to handle both eyes by a single regression function [6]. The eye is then fed into a CNN model and we will perform the gaze classification. 

For the CNN model, we have currently used AlexNet architecture. AlexNet is used to perform regression and predict gaze angle theta and phi. The angle then will be used to calculate the gaze direction.

![Block diagram gaze detection.png](Gaze%20Detection%20Final%20Report%202f917c433ed44904bcf0aa89694ca57a/Block_diagram_gaze_detection.png)

## Model

*Model Architecture*

The model architecture we chose was the AlexNet model. AlexNet consists of eight layers, including five convolutional layers and three fully connected layers. The first convolutional layer has 96 filters and the last convolutional layers have 256 filters. The first two fully connected layers have 4096 neurons each. Each layer also contains ReLu activation.

The first two convolutional layers and the fifth convolutional layer also contain a response normalization and the max pooling layer. 

The first two fully connected layers contain a dropout layer, which uses the dropout value of 0.5.  

![Multi Head AlexNet Model.png](Gaze%20Detection%20Final%20Report%202f917c433ed44904bcf0aa89694ca57a/Multi_Head_AlexNet_Model.png)

The Alexnet model originally performed a classification task, but we used it for regression. We made some changes in the model, so the task at hand could be accomplished. 

For it, we modified the final fully connected layers, which originally contained 1000 neurons in output for classifying images of 1000 classes. 

We modified the Alexnet architecture to have two sideheads. Each side head has a similar structure to that of the original fully connected layers. The final layer of each side head returns a single neuron in the output instead of the original 1000 neurons. Each side head predicts the gaze angle theta($\theta$) and phi($\phi$) respectively. The predicted gaze angles are then used to predict the direction of the gaze. The figure above gives the block diagram for the model we used. 

![alexnetarc.PNG](Gaze%20Detection%20Final%20Report%202f917c433ed44904bcf0aa89694ca57a/alexnetarc.png)

*Model Performance*

For training the model, we used the Adam optimizer and MSE Loss function. We trained three model with different learning rate and batch size. The results we obtained is described in the table below:

| Model (AlexNet) | theta MSE | phi MSE | theta MAE | phi MAE |
| --- | --- | --- | --- | --- |
| Adam Optimizer (learning rate = 0.001) and batch size = 256 | training = 1.020 validation = 2.7436 | training = 1.2207 validation = 3.6295 | training = 0.7885 validation = 1.1504 | training = 0.8590 validation = 1.2576 |
| Adam Optimizer (learning rate = 0.0001) and batch size = 256 (best) | training = 0.6315 validation = 2.0583 | training = 0.7976 validation = 2.8460 | training = 0.6241 validation = 0.9806 | training = 0.6954 validation = 1.0648 |
| Adam Optimizer (learning rate = 0.001, weight decay = 1e-5) and batch size = 128 | training = 1.4701 validation = 3.0785 | training = 1.6563 validation = 4.1316 | training = 0.9451 validation = 1.2670 | training = 1.0029 validation = 1.3760 |

As we can see from the table above that the best performing model was the model with Adam optimizer with learning rate of 0.0001 and batch size 256. The graph below shows its performance graph. We trained each model for 50 epochs. We can see that the best performance was obtained in the 49th epoch.  

![plot best.png](Gaze%20Detection%20Final%20Report%202f917c433ed44904bcf0aa89694ca57a/plot_best.png)

## Deliverables

The main deliverables for the gaze detection system will be as follows:

- Detect a user's face and eye from a given video feed.
- Determine the gaze direction and gaze detection.

## Limitations

There are some limitations in the model that we developed.

1. False detection of eye pupils may occur and cause incorrect or delayed gaze prediction.
2. Slightly unstable prediction.
3. Predicted gaze may be incorrect sometimes.

## Conclusion

We were successfully able to perform gaze detection. The system we built was an model-based system, where we fed the picture of the eye into the model and predict the gaze angle theta and phi and use those angles to predict gaze direction. 

## References

1. A. Kottwani and A. Kumar, ”*Eye Gaze Estimation Model Analysis.”* 28 July 2022. [Online]. Available: [https://arxiv.org/pdf/2207.14373.pdf](https://arxiv.org/pdf/2207.14373.pdf)
2. A. Gudi, X. Li and J. van Gemert  “*Efficiency in Real-time Webcam Gaze Tracking*” 2 September 2020  [Online]. Available: [https://arxiv.org/pdf/2009.01270.pdf](https://arxiv.org/pdf/2009.01270.pdf)
3. B. Mahanama, Y. Jayawardana, and S. Jayarathna, “*Gaze-Net: Appearance-Based Gaze Estimation using Capsule Networks.*” 20 May 2020. [Online]. Available: [https://www.cs.odu.edu/~sampath/publications/conferences/2020/AH_2020_Bhanuka.pdf](https://www.cs.odu.edu/~sampath/publications/conferences/2020/AH_2020_Bhanuka.pdf)
4.  T. Guo, Y. Liu, H. Zhang, X. Liu, Y. Kwak, B. I. Yoo, J. Han, and C. Choi, “*A Generalized and Robust Method Towards Practical Gaze Estimation on Smart Phone.”* 16 October 2019*.* [Online]. Available: [https://arxiv.org/pdf/1910.07331.pdf](https://arxiv.org/pdf/1910.07331.pdf)
5. E. Wood, T. Baltrusaitis, X. Zhang, Yusuke Sugano, P. Robinson, and A. Bulling, “*Rendering of Eyes for Eye-Shape Registration and Gaze Estimation.*” 21 May 2015. [Online]. Available: [https://arxiv.org/pdf/1505.05916.pdf](https://arxiv.org/pdf/1505.05916.pdf) 
6. X. Zhang, Y. Sugano, M. Fritz, and  A. Bulling, “*Appearance-Based Gaze Estimation in the Wild.*”
**** 11 Apr 2015. [Online]. Available: [https://arxiv.org/pdf/1504.02863.pdf](https://arxiv.org/pdf/1504.02863.pdf)
7. S. Barbero and J. Portilla,  “*Simulating real-world scenes viewed through ophthalmic lenses.*”  5 July 2017. [Online]. Available: h[ttps://www.researchgate.net/publication/318134973_Simulating_real-world_scenes_viewed_through_ophthalmic_lenses](https://www.researchgate.net/publication/318134973_Simulating_real-world_scenes_viewed_through_ophthalmic_lenses)