from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import os
import pandas as pd

# Assuming you have your dataset and labels loaded into X and y
file_path = os.path.abspath("../Datasets/AdultDatasetGroupEncoded.csv")
df = pd.read_csv(file_path)

y_label = df['income']
df = df.drop('income', axis=1)

S = df['gender']
df = df.drop('gender', axis=1)
profiles = df['groups']
df = df.drop('groups', axis=1)

X = df
y = y_label

# Number of train/test splits
num_splits = 30

# List to store F1 scores for each split
f1_scores = []

for _ in range(num_splits):
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # You can set a specific random state if needed

    # Create and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters as needed
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the F1 score and store it
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Print the F1 scores for each split
for i, f1 in enumerate(f1_scores):
    print(f'Split {i+1} - F1 Score: {f1:.2f}')

# Calculate the average F1 score across all splits
average_f1 = np.mean(f1_scores)

print(f'Average F1 Score over {num_splits} splits: {average_f1:.3f}')
