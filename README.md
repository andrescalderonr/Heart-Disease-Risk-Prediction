# Heart-Disease-Risk-Prediction

## Andrés Felipe Calderón Ramírez

## AWS SageMaker Execution Evidence 

## Summary:

This repository implements a complete logistic regression pipeline for heart disease prediction using the Kaggle Heart Disease dataset. 
The project covers the full machine learning workflow, including data exploration, preprocessing, model training from
scratch, regularization analysis, visualization, and deployment preparation.

## Dataset:

The dataset used in this project is the Heart Disease Dataset obtained from Kaggle. It contains 230 patient records, where each
instance represents clinical and demographic information collected for heart disease diagnosis.

Each sample includes features such as age, blood pressure, cholesterol level, maximum heart rate, ST depression, and the number
of major vessels observed via fluoroscopy. The target variable indicates the presence (1) or absence (0) of heart disease, making
this a binary classification problem.

The dataset is moderately balanced, with both classes (disease presence and absence) reasonably
represented. Its size and feature set make it suitable for exploring logistic regression, feature selection, regularization, and model
interpretability.

## Process in AWS:

1. I have started the lab in the AWS academy. 
2. I enter to sagemaker with the user we created in class.
3. In sagemaker I created a new code editor instance.

![](img/newinstance.png)

4. In the original git hub I exported the weight and bias form the model with the best score in F1-score.
5. In the editor I create a folder and drag the weight and bias.
6. I created a inference.py archive were I load the data.

![](img/exportandmodel.png)

7. In a test.py archive I test the model with a patient.

![](img/test.png)