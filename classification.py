import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree

import itertools
from sklearn.metrics import classification_report, confusion_matrix

#######################################################################################################################
# KNN
#######################################################################################################################
def knn_findk(Ks, X_train, y_train, X_test, y_test):
    """
    """
    # will test several k's
    mean_acc = np.zeros((Ks - 1))
    std_acc = np.zeros((Ks - 1))

    best_k = None
    best_acc = None
    best_model = None
    for k in range(1, Ks):
        kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        yhat = kNN_model.predict(X_test)
        mean_acc[k - 1] = np.mean(yhat == y_test)  # mean accuracy
        std_acc[k - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])  # std
        if best_acc is None or best_acc < mean_acc[k - 1]:
            best_k = k
            best_acc = mean_acc[k - 1]
            best_model = kNN_model

    plt.plot(range(1, Ks),
             mean_acc,
             'g')
    plt.fill_between(range(1, Ks),
                     mean_acc - 1 * std_acc,
                     mean_acc + 1 * std_acc,
                     alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3std'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()

    # best k where accuracy the highest
    print(f"Best k out of {Ks} is {best_k} with accuracy {np.round(max(mean_acc), 4)}")

    return best_model


#######################################################################################################################
# DECISION TREES
#######################################################################################################################

# DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# DT_model.fit(X_train,y_train)
def plot_decisionTrees(DT_model, X_features, y_features):

    # DOT data
    dot_data = tree.export_graphviz(DT_model, out_file=None,
                                    feature_names=X_features,
                                    class_names=y_features,
                                    filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png")
    return graph


#######################################################################################################################
# LOGISTIC REGRESSION: CONFUSION MATRIX
#######################################################################################################################

def plot_confusion_matrix(y_test,
                          yhat,
                          labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=labels)
    np.set_printoptions(precision=2)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#######################################################################################################################
# SVM
#######################################################################################################################

"""
The SVM algorithm offers a choice of kernel functions for performing its processing. 
Basically, mapping data into a higher dimensional space is called kernelling. 
The mathematical function used for the transformation is known as the kernel function, and can be of different types, 
such as:
    1.Linear
    2.Polynomial
    3.Radial basis function (RBF)
    4.Sigmoid
Each of these functions has its characteristics, its pros and cons, and its equation, 
but as there's no easy way of knowing which function performs best with any given dataset,
 we usually choose different functions in turn and compare the results. 
 Let's just use the default, RBF (Radial Basis Function) for this lab.
"""