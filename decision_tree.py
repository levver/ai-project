import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from decimal import Decimal


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

    max_auc_iter = [0] * iterations
    max_depth_iter = [0] * iterations
    max_trees_iter = [0] * iterations
    for i in range(iterations):
        auc_by_forest_size = np.array([0] * len(forest_size)).astype(float)
        auc_by_depth_size = np.array([0] * size_sqrt).astype(float)
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
                if auc > max_auc_iter[i]:
                    max_auc_iter[i] = auc
                    max_trees_iter[i] = size
                    max_depth_iter[i] = depth
        plt.figure()
        auc_by_forest_size = auc_by_forest_size / (iterations*len(depth_size))
        auc_by_depth_size = auc_by_depth_size / (iterations*len(forest_size))
        plt.plot(forest_size, auc_by_forest_size, 'b')
        plt.ylabel("accuracy")
        plt.xlabel("forest size")
        plt.savefig('d:\Documents\Toar\AI\project\graphs\forest_size_avg' + str(i) +'.png')
        plt.figure()
        plt.plot(depth_size, auc_by_depth_size, 'r')
        plt.ylabel("accuracy")
        plt.xlabel("max depth")
        plt.savefig('d:\Documents\Toar\AI\project\graphs\max_depth_avg' + str(i) + '.png')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(max_trees_iter, max_depth_iter, max_auc_iter, c='r', marker='o')
    ax.set_xlabel('best forest size')
    ax.set_ylabel('best max depth')
    ax.set_zlabel('accuracy')
    plt.savefig('d:\Documents\Toar\AI\project\graphs\iterations_sum.png')
    print("Best accuracy: " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth))
    return max_forest


def process_data(data):
    data[data.columns[0]], uniques = pandas.factorize(data[data.columns[0]])
    drops = []
    for i in range(len(data)):
        for j in range(1, len(data.columns)-1):
            val = np.array(data.iloc[i,j])
            if data.columns[j] not in  ["RSI", "volume"]:
                if val[0] != 0:
                    val = val / val[0]
                if len(val[val >= 10]) != 0:
                    drops.append(i)
                    break
            elif data.columns[j] == "RSI":
                if val[0] != 0:
                    val = val / 100
                if len(val[val >= 1]) != 0:
                    drops.append(i)
                    break
            else:
                if val[0] != 0:
                    val = val / max(val)
            try:
                data.iloc[i, j] = Decimal("".join(map("{:3.2f}".format, val)).replace(".", ""))
            except:
                drops.append(i)
                break
    data = data.drop(drops)
    data.to_pickle(r'd:\Documents\Toar\AI\project\normalized_data.pkl')
    data.to_csv(r'd:\Documents\Toar\AI\project\normalized_data.csv')

data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\normalized_data.pkl')
#data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\data_set.pkl')
#process_data(data)
d_tree = return_decision_tree(data, 5)
#data[data.columns[0]] = uniques
