# Mushroom Prediction Program

This program is the third in a series of machine learning applications designed to help users understand the mechanics of machine learning. It utilizes a Decision Tree Classifier to predict characteristics of mushrooms based on historical data.

## Overview
- **Objective**: Predict mushroom characteristics based on specific features using a decision tree model. *Note: This tool is strictly for educational purposes and should not be used to determine the edibility of mushrooms in real life.*
- **Dataset**: The program uses a historical mushroom dataset in CSV format.

## Features
- **Preprocessing**: Label encoding is applied to convert categorical data into a numerical format suitable for the Decision Tree Classifier.
- **Text-Based and GUI Versions**: 
  - **Text-Based Version**: Prints the data, confusion matrix, and decision tree to the console.
  - **Graphic-Based Version**: Uses Tkinter for a GUI, allowing users to view past data and predictions interactively.
- **Model Evaluation**: The program displays a confusion matrix and visualizes the decision tree for easy understanding of model performance.

## Program Structure
1. **Data Loading and Preprocessing**:
   - Uses `pandas` to load the mushroom dataset.
   - Applies `LabelEncoder` to convert categorical features into numerical values.
   - Splits data into training and testing sets.

2. **Machine Learning Model**:
   - **Decision Tree Classifier**: A supervised learning algorithm trained on labeled mushroom data to classify new data points based on input features.
   - **Scikit-Learn**: The program uses the Scikit-Learn library for machine learning functions, including the decision tree and evaluation metrics.

3. **Evaluation and Visualization**:
   - **Confusion Matrix**: Provides insights into the model's accuracy by comparing predicted vs. actual values.
   - **Decision Tree Visualization**: Displays the structure of the decision tree to illustrate how the model makes decisions.

## Caution
This program is for educational purposes only. Always exercise caution with mushroom identification, as no machine learning model can ensure 100% accuracy in real-world applications.

## Comments and Explanation
Extensive comments are provided throughout the code to explain each step of the process. However, please feel free to reach out with any questions.

## Requirements
- **pandas**: For data loading and manipulation.
- **LabelEncoder**: For encoding categorical data into numerical values.
- **Scikit-Learn**: Provides the `DecisionTreeClassifier`, evaluation metrics, and visualization tools.
- **Tkinter (GUI Version)**: Enables a graphical interface for interactive exploration of the data.

## Summary
The Mushroom Prediction Program provides a hands-on introduction to machine learning for classification. It offers both text-based and GUI versions, allowing users to gain insight into the data and the decision-making process of a machine learning model.

Feel free to ask any questions or seek further clarification!
