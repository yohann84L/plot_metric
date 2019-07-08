# Package
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from pandas import DataFrame
from plot_metric.functions import MultiClassClassification
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
X, y = load_digits(n_class=4, return_X_y=True)

# Add noisy features to make the problem harder
random_state = np.random.RandomState(23)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 1000 * n_features)]

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(DataFrame(X), y, test_size=0.2, random_state=42)

# Building Classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(X_train, y_train)

# Predict test set
y_pred = gnb.predict_proba(X_test)

# Visualisation with plot_metric :
mc = MultiClassClassification(y_test, y_pred, labels=[0, 1, 2, 3])


plt.figure(figsize=(13,4))
plt.subplot(131)
mc.plot_roc()
plt.subplot(132)
mc.plot_confusion_matrix()
plt.subplot(133)
mc.plot_confusion_matrix(normalize=True)

plt.savefig('images/example_multi_classification.png')
plt.show()

mc.print_report()

