import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Prepare the data
input_dir = r"D:\d\data\clf-data"
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    print(f"Checking directory: {category_path}")
    if not os.path.exists(category_path):
        print(f"Directory does not exist: {category_path}")
        continue

    files = os.listdir(category_path)
    print(f"Number of files in {category_path}: {len(files)}")
    if len(files) == 0:
        print(f"No files in {category_path}")
        continue

    for file in files:
        img_path = os.path.join(category_path, file)
        if not os.path.isfile(img_path):
            print(f"Skipping non-file item: {img_path}")
            continue
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)
print(f"Total samples: {len(data)}")
print(f"Labels: {np.unique(labels, return_counts=True)}")

# Check if there's more than one class
if len(np.unique(labels)) < 2:
    raise ValueError("The dataset must contain more than one class for classification.")

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters, cv=5)
grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(x_test)

# Print performance metrics
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))


# Save the model to a file
model_path = r"D:\d\Imageclassifier\svm_model.pkl"
with open(model_path, 'wb') as model_file:
    pickle.dump(best_estimator, model_file)

print(f"Model saved to '{model_path}'")
