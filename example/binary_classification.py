# Package
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from pandas import DataFrame
from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(DataFrame(X), y, test_size=0.2, random_state=42)

# Building Classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(X_train, y_train)

# Predict test set
y_pred = gnb.predict_proba(X_test)[:,1]

# Visualisation with plot_metric :
bc = BinaryClassification(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(8,8))
plt.subplot(221)
bc.plot_roc()
plt.subplot(222)
bc.plot_class_distribution()
plt.subplot(223)
bc.plot_confusion_matrix()
plt.subplot(224)
bc.plot_confusion_matrix(normalize=True)

plt.savefig('images/example_binary_classification.png')
plt.show()

bc.print_report()