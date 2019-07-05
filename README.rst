plot_metric
===========

|PyPI-Versions|

Librairie to simplify plotting of metric like ROC curve, confusion matrix etc..

Installation
------------
Using pip :

.. code:: sh

    pip install plot-metric


Example
-------

Let's load a simple dataset and make a train & test set :

.. code:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), y, test_size=0.2, random_state=42)


Train our classifier and predict our test set :

.. code:: python

    from sklearn.naive_bayes import GaussianNB
    
    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)
    # Use predict_proba to predict probability of the class
    y_pred = gnb.predict_proba(X_test)[:,1]


We can now use ``plot_metric`` to plot ROC Curve, distribution class and classification matrix :

.. code:: python

    from plot_metric.functions import  BinaryClassification
    import matplotlib.pyplot as plt
    bc = BinaryClassification(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(10,9))
    plt.subplot(221)
    bc.plot_roc()
    plt.subplot(222)
    bc.plot_class_distribution()
    plt.subplot(223)
    bc.plot_confusion_matrix()
    plt.subplot(224)
    bc.plot_confusion_matrix(normalize=True)
    plt.show()
    bc.print_report()

    >>>                    ________________________
    >>>                   |  Classification Report |
    >>>                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    >>>               precision    recall  f1-score   support
    >>>            0       1.00      0.93      0.96        43
    >>>            1       0.96      1.00      0.98        71
    >>>    micro avg       0.97      0.97      0.97       114
    >>>    macro avg       0.98      0.97      0.97       114
    >>> weighted avg       0.97      0.97      0.97       114


.. image:: example/images/example_binary_classification.png
    :width: 150px

.. |PyPI-Versions| image:: https://img.shields.io/badge/plot__metric-v0.0.2-blue.svg
    :target: https://pypi.org/project/plot-metric/
    