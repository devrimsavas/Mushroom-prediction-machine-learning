#prediction text mode

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Open CSV data file
df = pd.read_csv("mushroom.csv")

#columns data correction
df=df.applymap(lambda x: x.strip() if isinstance(x,str) else x)

# Data Inspection
print("First few rows of the dataset:")
print(df.head())

# Create a label encoder for columns 
color_encoder = LabelEncoder()
df['Color'] = color_encoder.fit_transform(df['Color'])

appearance_encoder = LabelEncoder()
df['Appearance'] = appearance_encoder.fit_transform(df['Appearance'])

smell_encoder = LabelEncoder()
df['Smell'] = smell_encoder.fit_transform(df['Smell'])

poisoned_encoder = LabelEncoder()
df['Poisoned'] = poisoned_encoder.fit_transform(df['Poisoned'])

# Test for label encoding 
print("\nData after encoding:")
print(df.head())

# Set data as features and output X, y
features = ['Color', 'Appearance', 'Smell']
X = df[features]
y = df['Poisoned']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree with Train Data
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

# Predict using the test set
y_pred = dtree.predict(X_test)

# Evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=poisoned_encoder.classes_)
cm_display.plot()
plt.show()

# Plot the decision tree
plot_tree(dtree, feature_names=features, filled=True)
plt.show()

# Further sample prediction
sample_data = [[1, 0, 2]]  # Replace this with your sample data
predicted_class = dtree.predict(sample_data)
print(f"\nPredicted Class for sample data: {poisoned_encoder.inverse_transform(predicted_class)[0]}")
