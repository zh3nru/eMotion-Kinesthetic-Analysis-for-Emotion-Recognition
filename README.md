# eMotion: Kinesthetic Analysis for Emotion Recognition

eMotion is a system that analyzes kinesthetic features ***(body movements, postures, gestures)*** to detect and recognize human emotions ***(Happiness, Sadness, Surprise, Disgust, Anger, Fear, Neutral)*** in real time. Unlike traditional emotion detection tools that rely on facial expressions, eMotion focuses exclusively on kinesthetic features, offering a unique approach that can possibly contribute to the advancement of existing emotion recognition systems.

This repository mainly focuses on the machine learning side of the system - dataset creation and modification, model training, and the actual model that can be used for inference purposes.

## Dataset
Our model utilizes the dataset from the **BoLD (BOdy Language Dataset) Challenge**: [here](https://cydar.ist.psu.edu/emotionchallenge/dataset.php)

## Model Training
In order for the BoLD dataset to accomodate the needs and objectives of our system, we decided to modify the contents of the dataset. Then, fed the modified dataset to our very own CNN architecture to train the model. The code used can be located "here"

![Image](https://github.com/user-attachments/assets/edcdb371-7fda-42d7-896e-1cf99a6effbf)

## Result
Our model achieved a benchmark accuracy of approximately **75%** during its training with a total iteration of 20 epochs.

![Image](https://github.com/user-attachments/assets/9daec850-4e3d-43bb-987d-c8466d83229f)

### Citation:
This project utilizes the dataset from the BoLD (BOdy Language Dataset) Challenge:

[1] Yu Luo, Jianbo Ye, Reginald B. Adams, Jr., Jia Li, Michelle G. Newman, and James Z. Wang, ARBEE: Towards Automated Recognition of Bodily Expression of Emotion In the Wild,'' International Journal of Computer Vision, vol. 128, no. 1, pp. 1-25, 2020.
