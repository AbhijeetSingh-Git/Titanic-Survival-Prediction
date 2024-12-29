# Titanic Survival Prediction

## Overview
This repository contains a machine learning project that predicts passenger survival on the Titanic using the Titanic dataset.

## Approach
1. **Data Preprocessing**:
    - Handled missing values.
    - Encoded categorical variables.
    - Standardized numerical features.
    - Selected key features for the model.
2. **Model Training**:
    - Used a Random Forest Classifier.
3. **Evaluation**:
    - Achieved an accuracy of 1.00 on the validation dataset.

## Files
- `train.csv`: Original training dataset.
- `test.csv`: Original testing dataset.
- `processed_train_data.csv`: Preprocessed training dataset.
- `titanic_survival_model.pkl`: Trained model file.

## Usage
1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script:
    ```bash
    python titanic_survival_prediction.py
    ```

## Challenges
- Handling missing data in the `Age` and `Embarked` columns.
- Balancing the dataset for better model performance.

## Results
- Accuracy: 1.00

