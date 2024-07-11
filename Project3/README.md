# Image Classification using AWS Sagemaker

This README file provides an overview and instructions for the project using Sagemaker.

## Project Overview


This project focuses on classifying different categories of clothing using AWS SageMaker. It demonstrates the process of building, training, and deploying an image classification model. The project involves uploading training data to S3, training a pre-trained model with hyperparameter tuning, and deploying the model to a SageMaker endpoint for inference.

## Project Setup Instructions

1. **Clone the Repository**:
    Use the AWS SageMaker "Clone a Git Repository" option to clone the repository directly into your SageMaker environment.

2. **Install Required Packages**:
    Open a terminal in your SageMaker environment and run:
    ```bash
    pip install -r requirements.txt
    ```
3. **Download the Data**:
    Download the dataset from the following link: [Polyvore Dataset](https://github.com/xthan/polyvore-dataset)

4. **Upload Data to S3**:
    Upload the data to S3.

5. **Open Jupyter Notebook**:
    In the SageMaker Jupyter environment, open the provided Jupyter notebook for the project and follow the steps.


## Files in the Project

1. fetch_and_prepare_training_data.ipynb:
2. MobileNetTraining.py:
3. img/:
4. 

## Model Selection: MobileNet

We are using MobileNet as our pre-trained model for this project. MobileNet is a lightweight, efficient model that is well-suited for mobile and embedded vision applications. Its architecture is designed to reduce the number of parameters and computational cost, making it ideal for scenarios with limited resources.

## Hyperparameter Optimization

In this section, we fine-tune the MobileNet model and optimize its hyperparameters to improve performance.

1. **Define Hyperparameter Ranges**:
    
2. **Initialize the Hyperparameter Tuner**:
    
3. **Start Hyperparameter Tuning**:

### Visualize Tuning Progress

Monitor the progress of the hyperparameter tuning jobs through the SageMaker console.

![Hyperparameter Tuning Results](img/HT.png)

![Best Hyperparameters](img/HT2.png)

![Logging](img/HT3.png)

## Training Model with Best Hyperparameters and Debugging

In this section, we describe how to train the MobileNet model with the best hyperparameters found during hyperparameter tuning. Additionally, we provide guidelines for monitoring and debugging the training process.

### Training with Best Hyperparameters

After completing the hyperparameter tuning job, use the best hyperparameters to train the model. This ensures the model is trained with the most optimal settings for performance.

### Monitoring and Debugging

It's important to monitor the training process and be prepared to debug any issues that arise. Below is an example plot showing the CrossEntropyLoss over training steps for both training and validation datasets, which helps in identifying potential issues.

![Debugging Plot](img/debugging_plot.png)

### Interpreting the Debugging Plot

The plot above shows the training and validation loss throughout the training process. 

1. **Normal Behavior**:
    - The training loss (blue line) should generally decrease over time.
    - The validation loss (orange line) should also decrease, ideally following a similar trend to the training loss.

2. **Anomalous Behavior**:
    - **Spikes or Plateaus**: Sudden spikes or plateaus in the loss curves can indicate issues.
    - **Divergence**: A large divergence between training and validation loss might suggest overfitting or data issues.

### Debugging Steps

If you observe any anomalous behavior in the plot, follow these steps to debug the model:

1. **Identify the Anomaly**:
    - Determine the point at which the anomaly occurs in the training process.

2. **Check Data Pipeline**:
    - Verify that the data is being loaded and preprocessed correctly. Ensure data augmentation and normalization steps are applied consistently.

3. **Review Hyperparameters**:
    - Examine the chosen hyperparameters to ensure they are within reasonable ranges.

4. **Analyze Model Architecture**:
    - Confirm that the MobileNet architecture is correctly implemented without any issues.

5. **Monitor Training Logs**:
    - Use AWS CloudWatch logs to review detailed logs of the training process. Look for warnings or errors.

6. **Conduct Experiments**:
    - Try different combinations of hyperparameters to see if the issue persists. 



