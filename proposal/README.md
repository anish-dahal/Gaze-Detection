# Gaze Detection: Proposal Report

## Abstract

Gaze detection is to locate the position on a monitor screen where a user is looking. The direction of a person's gaze can provide valuable information about their focus of attention, and this can be utilized to enhance the interaction between humans and computers. Gaze detection has a multitude of applications, including detecting fatigue, authenticating bio-metrics, diagnosing diseases, recognizing activities, estimating alertness levels, utilizing gaze-contingent displays, improving human-computer interaction, and more. In this document, we have presented a proposal for a gaze detection system that can detect the gaze of a user’s eye when they are using a computer using a webcam. We have proposed this project be completed in two phases the first one includes localization of eye position and the second includes tracking the motion of the pupil to determine gaze direction. 

*Keywords: Eye gaze detection, Eye tracking*

## Introduction

Gaze detection, which is a rapidly progressing area of computer vision and human-computer interaction, seeks to precisely track and decipher the direction of a person's gaze. The widespread use of devices like smartphones, virtual reality headsets, and smart glasses has further emphasized the significance of gaze detection for a range of applications, such as eye-tracking interfaces, attention-aware systems, and facial expression recognition. 

Eye tracking is still a novel technology that requires adequate computing resources. However, with recent advances in deep learning, high accuracy can be achieved without the need for complex hardware. 

Gaze detection offers a wide range of applications, including improving accessibility for people with disabilities and enhancing immersive experiences in gaming and virtual reality. Furthermore, it has promising implications for fields such as marketing research, psychology, and neuroscience, where it can offer valuable insights into human behavior and cognition.

In our work, we implement gaze detection with a computer vision system setting a web camera above the monitor screen, and the user moves his/her face/eyes to gaze at different positions on the monitor. 

### Problem Statement

The gaze detection help us determine the direction an user is focusing on. The inability to determine the user’s gaze direction may hinder us in various fields like human-computer interaction system, user experience in using a software, gaming, etc. The problem with gaze detection is it may require special and complex hardware. The complex hardware may be hard to set up and use. This causes the loss of practicality. Using deep learning methods and using software only application for gaze detection could improve usability, practicality of the gaze detection system.

### Objectives

The objectives for the gaze detection system will be:

- Detect face and eyes of the user.
- Detect the pupils of the eye, along with other eye landmarks
- Perform gaze detection and predict the direction the user is looking at.

## Literature Review

Gaze estimation methods can classified as either feature-based, model-based or appearance-based. Appearance based methods directly use image patches of the eye or face for estimation through using deep convolutional neural network (CNN)[7], where CNN-based estimation model significantly outperforms state-of-the-art methods in the most challenging
person- and pose-independent training scenario . Appearance-based gaze estimation methods
directly use eye images as input and can therefore potentially work with low-resolution eye images.

Various datasets have contributed to the progress of gaze estimation methods and the reporting of their performance[5]. Wood et al. [8] presented a method to synthesize perfectly labelled realistic close-up images of the human eye. Zhang et. al [7] built a novel in-the-wild gaze dataset through a long-term data collection using laptops, which shows significantly larger variations in eye appearance than existing datasets.

As presented by Park et. al [5], there has been significant improvement in the performance using EVE dataset proposed in the paper itself using GazeNet architecture. In discussed by Mahanama et. al[3], Gaze-Net combines components trained via both data-driven and interaction-driven approaches, which enables to realize the benefits of both methodologies.

Further various architectures have been proposed over years serving as CNN based architecture for gaze estimation including Alexnet, Lenet, VGG and many more.

## Project Pipeline

*Data Collection*

The dataset that we will be using for our project is MPIIGaze which is a dataset for appearance-based gaze estimation in the wild. It contains 213,659 images collected from 15 participants during natural everyday laptop use over more than three months.

*Model Process*

The project pipeline starts from an input image or an input frame from a video. In this frame we perform face and landmark detection using pre-trained models like: Haar Cascade and Dlib. Using these models, we perform face detection. After the face detection is complete, we crop the face segment of the image. On the cropped image we perform eye detection and crop the eye. The eye is then fed into a CNN model and we will perform the gaze classification. For the CNN model, we have currently considered AlexNet, VGG and LeNet. 

![Block diagram.drawio.png](Gaze%20Detection%20Proposal%20Report%20edc6b65fd4cd4767981efd6396ab40d2/Block_diagram.drawio.png)

## Deliverables

The main deliverables for the gaze detection system will be as follows:

- To detect the face, eye of a user from given video feed.
- To determine the gaze direction and gaze estimation.

## Task Division

As the main goal for the project is to learn, we will be helping each other through each task in the project without strict division of the said tasks. However we have prepared a Gantt chart to show the flow of the tasks. 

![Gantt Chart Project Timeline Graph.png](Gaze%20Detection%20Proposal%20Report%20edc6b65fd4cd4767981efd6396ab40d2/Gantt_Chart_Project_Timeline_Graph.png)

## References

1. A. Kottwani and A. Kumar, ”*Eye Gaze Estimation Model Analysis.”* 28 July 2022. [Online]. Available: [https://arxiv.org/pdf/2207.14373.pdf](https://arxiv.org/pdf/2207.14373.pdf)
2. A. George, “*Image based eye gaze tracking and its applications,*” Ph.D. dissertation, Dept. Elect. Eng., IIT Kharagpur, India. 9 July 2019. [Online]. Available: [https://arxiv.org/pdf/1907.04325.pdf](https://arxiv.org/pdf/1907.04325.pdf)
3. B. Mahanama, Y. Jayawardana, and S. Jayarathna,“*Gaze-Net: Appearance-Based Gaze Estimation using Capsule Networks.*” 20 May 2020. [Online]. Available: [https://www.cs.odu.edu/~sampath/publications/conferences/2020/AH_2020_Bhanuka.pdf](https://www.cs.odu.edu/~sampath/publications/conferences/2020/AH_2020_Bhanuka.pdf)
4.  T. Guo, Y. Liu, H. Zhang, X. Liu, Y. Kwak, B. I. Yoo, J. Han, and C. Choi, “*A Generalized and Robust Method Towards Practical Gaze Estimation on Smart Phone.”* 16 October 2019*.* [Online]. Available: [https://arxiv.org/pdf/1910.07331.pdf](https://arxiv.org/pdf/1910.07331.pdf)
5. S. Park, E. Aksan, X. Zhang, and O. Hilliges, “*Towards End-to-end Video-based Eye-Tracking.*” 26 Jul 2020. [Online]. Available: [https://arxiv.org/pdf/2007.13120.pdf](https://arxiv.org/pdf/2007.13120.pdf)
6. E. Wood, T. Baltrusaitis, X. Zhang, Yusuke Sugano, P. Robinson, and A. Bulling, “*Rendering of Eyes for Eye-Shape Registration and Gaze Estimation.*” 21 May 2015. [Online]. Available: [https://arxiv.org/pdf/1505.05916.pdf](https://arxiv.org/pdf/1505.05916.pdf) 
7. X. Zhang, Y. Sugano, M. Fritz, and  A. Bulling, “*Appearance-Based Gaze Estimation in the Wild.*”
**** 11 Apr 2015. [Online]. Available: [https://arxiv.org/pdf/1504.02863.pdf](https://arxiv.org/pdf/1504.02863.pdf)
8. E. Wood, T. Baltrusaitis , X. Zhang, Y. Sugano, P. Robinson, and A. Bulling, “*Rendering of Eyes for Eye-Shape Registration and Gaze Estimation.”* 21 May 2015. [Online]. Available: [https://arxiv.org/pdf/1505.05916.pdf](https://arxiv.org/pdf/1505.05916.pdf)