import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




def create_random_forest(X_train, y_train, X_val, y_val, forest_size, max_depth):
    """
    The function creates a random forest and return the forest and its accuracy.
    :param X_train: X train
    :param y_train: y train
    :param X_val: X test
    :param y_val: y test
    :param forest_size: the number of trees in the forest
    :param max_depth: the max depth of each tree
    :return: the forest and its accuracy
    """
    forest = RandomForestClassifier(n_estimators=forest_size, criterion="entropy", max_depth=max_depth, max_samples=0.7)
    forest.fit(X_train, y_train)
    test_prediction =  forest.predict(X_val)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, test_prediction)
    return forest, auc(false_positive_rate, true_positive_rate)



def return_decision_tree(data_frame, iterations=1, validation_size=0.3, forest_size = None, depth_size = None):
    """
    The function will return a decision tree classifier for the given data_frame.
    in each iteration the function will create decision tree base on a different split to train and test sets.
    :param data_frame: the data_frame to work on.
    :param iterations: number of iteration we want the function to do.
    :param validation_size: the percentage of the test set from the entire data.
    :param forest_size: the number of trees in the forest. By default tries different sizes and picks the best.
    :param depth_size: the max depth of each tree in the forest. By default tries different sizes and picks the best.
    :return: best decision tree found.
    """
    X = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    data_size = len(data_frame)
    size_sqrt = int(data_size ** 0.5)
    forest_size = [4, 16, 64, 256, 1024, 4096] if not forest_size else [forest_size]
    depth_size = [i*size_sqrt for i in range(1, size_sqrt+1)] if not depth_size else [depth_size]
    max_forest = None
    max_auc = 0
    max_depth = 0
    max_trees = 0
    auc_by_forest_size = np.array([0] * len(forest_size)).astype(float)
    auc_by_depth_size = np.array([0] * size_sqrt).astype(float)
    for i in range(iterations):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size)
        for j, size in enumerate(forest_size):
            for z, depth in enumerate(depth_size):
                forest, auc = create_random_forest(X_train, y_train, X_val, y_val, size, depth)
                auc_by_forest_size[j] += auc
                auc_by_depth_size[z] += auc
                if auc > max_auc:
                    max_forest = forest
                    max_auc = auc
                    max_trees = size
                    max_depth = depth
    auc_by_forest_size = auc_by_forest_size / (iterations*len(depth_size))
    auc_by_depth_size = auc_by_depth_size / (iterations*len(forest_size))
    plt.plot(forest_size, auc_by_forest_size, 'b')
    plt.ylabel("accuracy")
    plt.xlabel("forest size")
    plt.show()
    plt.figure()
    plt.plot(depth_size, auc_by_depth_size, 'r')
    plt.ylabel("accuracy")
    plt.xlabel("max depth")
    plt.show()
    print("Best accuracy: " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth))
    return max_forest


#
#
# a = pandas.read_csv(r"d:\Documents\Toar\IML\Hackathon\flights_demo_test_file.csv")
# d_tree = return_decision_tree(a.iloc[:,8:])
# b = 5