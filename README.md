# CV - Gaze Detection


## First Approach
For solving the gaze detection, we firstly decided that we will try to implement the mathematical solution for gaze detection.
For this the GI4E dataset provided the landmarks for the corners and the pupil for each eye. Using these landmarks, we decided to calculate the general direction of where the pupil is looking at.  The problem with this approach was the threshold consideration, as there might be general differences in human eyes individually causing problem to consider threshold value where we determine where the user is looking at. Another problem with this approach was we cannot further process output, so it is difficult to track where the user might be looking at on the screen or other similar applications.

Thus we decided to use the model based approach for the gaze detection problem.

## Model based Approach
We have decided that we will be training a CNN model for calculating the gaze angles theta and phi. For this we are currently researhing and trying to implement the Alexnet model architecture and VGG-16 model architecture. For this we are currentl using MPII Gaze dataset. For the face and eye detection we used mtcnn from 'facenet.mtcnn' library and haar cascade for eye detection. But due to negative readings of haar cascade for eye detection we decided not to use the haar cascade.

For model based approach, the result for the model trained in partial data was satisfactory, so we decided for further training with more data with more epochs. 

After training further for model gave satisfactory results, we also implemented simple direction prediction.