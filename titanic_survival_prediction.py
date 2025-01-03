                                               # Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                                            #Load the dataset
import kagglehub
path = kagglehub.dataset_download("brendan45774/test-file")

train_data = pd.read_csv(f"train.csv")
test_data = pd.read_csv(f"test.csv")

                                            # Data Exploration
print("Training data overview:")
print(train_data.head())
print(train_data.info())
print(train_data.describe())

                                            # Data Preprocessing
                                                # Fill values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

                                            # Drop irrelevant features
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

                                        # Encode categorical variables
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

                                        # Separate features and target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

                                        # Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                                            # Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

                                                # Model Training

model = RandomForestClassifier(random_state=42)

                                                # Train the model
model.fit(X_train, y_train)

                                            # Predict on validation set
y_pred = model.predict(X_val)

                                            # Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save preprocessed data
train_data.to_csv("processed_train_data.csv", index=False)

# Save the model
import joblib
joblib.dump(model, "titanic_survival_model.pkl")

