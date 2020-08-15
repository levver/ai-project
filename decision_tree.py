import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from decimal import Decimal
from math import log


def create_random_forest(X_train, y_train, X_val, y_val, forest_size, max_depth, file):
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
    val_prediction =  forest.predict(X_val)
    true_rate = 1 - (np.count_nonzero(y_val-val_prediction.T)/len(y_val))
    print("validation prediction: " + str(true_rate) + " matches")
    file.write("validation prediction: " + str(true_rate) + " matches\n")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, forest.predict_proba(X_val)[:,1])
    return forest, auc(false_positive_rate, true_positive_rate), true_rate, val_prediction


def return_decision_tree(data_frame, iterations, path, validation_size=0.3, forest_size = None, depth_size = None):
    """
    The function will return a decision tree classifier for the given data_frame.
    in each iteration the function will create decision tree base on a different split to train and test sets.
    :param data_frame: the data_frame to work on.
    :param iterations: number of iteration we want the function to do.
    :param path: path to write results to
    :param validation_size: the percentage of the test set from the entire data.
    :param forest_size: the number of trees in the forest. By default tries different sizes and picks the best.
    :param depth_size: the max depth of each tree in the forest. By default tries different sizes and picks the best.
    :return: best decision tree found.
    """
    file = open(path + "\\results.txt", "w")
    factorize_data(data)
    X = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    data_size = len(data_frame)
    size_sqrt = int(data_size ** 0.5)
    forest_size = [1024, 256, 64, 16] if not forest_size else [forest_size]
    depth_size = list(map(lambda x: size_sqrt*2**x, range(0, int(log(size_sqrt, 2)+1))))[::-1] if not depth_size else [depth_size]
    max_forest = None
    #values of best tree (tree with max auc)
    max_auc = 0
    max_depth = 0
    max_true = 0
    max_trees = 0
    max_prediction = []
    max_y = []
    max_x = []

    all_trues = []
    all_depths = []
    all_aucs = []
    all_tree_sizes = []
    for i in range(iterations):
        auc_by_forest_size = np.array([0] * len(forest_size)).astype(float) #saves avg auc for each forest size
        auc_by_depth_size = np.array([0] * len(depth_size)).astype(float) #saves avg auc for each max depth
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size)
        print("percent of ones in validation: " + str(len(y_val[y_val==1])/len(y_val)))
        file.write("percent of ones in validation: " + str(len(y_val[y_val==1])/len(y_val))+"\n")
        for j, size in enumerate(forest_size):
            for z, depth in enumerate(depth_size):
                print("start iteration " + str(i) + " with " + str(size) + " trees and depth " + str(depth))
                file.write("\nstart iteration " + str(i) + " with " + str(size) + " trees and depth " + str(depth)+"\n")
                forest, auc, true_rate, y_prediction = create_random_forest(X_train, y_train, X_val, y_val, size, depth, file)
                print("tree auc value: " + str(auc))
                print("")
                file.write("tree auc value: " + str(auc) +"\n")
                file.write("\n")
                if auc_by_forest_size[j] < auc:
                    auc_by_forest_size[j] = auc
                if auc_by_depth_size[z] < auc:
                    auc_by_depth_size[z] = auc
                if auc > max_auc: #find max auc
                    max_forest = forest
                    max_auc = auc
                    max_trees = size
                    max_true = true_rate
                    max_depth = depth
                    max_prediction = y_prediction
                    max_y = y_val
                    max_x = X_val
                all_aucs.append(auc)
                all_trues.append(true_rate)
                all_tree_sizes.append(size)
                all_depths.append(depth)

        if forest_size is None:
            #first plot: avg auc value by forest size
            plt.figure()
            # auc_by_forest_size = auc_by_forest_size / len(depth_size)
            # auc_by_depth_size = auc_by_depth_size / len(forest_size)
            plt.plot(forest_size[::-1], auc_by_forest_size[::-1], 'b')
            plt.title("Iteration "+ str(i+1) +" max auc for each forest size")
            plt.ylabel("max auc")
            plt.xlabel("forest size")
            plt.savefig(path + '\\forest_size_max' + str(i) + '.png')

        if depth_size is None:
            # second plot: avg auc value by depth size
            plt.figure()
            plt.plot(depth_size[::-1], auc_by_depth_size[::-1], 'r')
            plt.title("Iteration "+ str(i+1) +" max auc for each max depth")
            plt.ylabel("max auc")
            plt.xlabel("max depth")
            plt.savefig(path + '\max_depth_max' + str(i) + '.png')


    #third plot: true predictions as a function of forest size and max depth
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_tree_sizes, all_depths, all_trues, c='r', marker='o')
    plt.title("true predictions as a function of forest size and max depth")
    ax.set_xlabel('forest size')
    ax.set_ylabel('max depth')
    ax.set_zlabel('true predictions rate')
    plt.savefig(path + '\\true_predictions_as_function.png')

    # forth plot: auc as a function of forest size and max depth
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_tree_sizes, all_depths, all_aucs, c='b', marker='o')
    plt.title("auc (rpr-fpr relation) as a function of forest size and max depth")
    ax.set_xlabel('forest size')
    ax.set_ylabel('max depth')
    ax.set_zlabel('auc value')
    plt.savefig(path + '\\auc_as_function.png')
    print("Best score: " + str(max_true) + " correct predictions with auc (tpr-fpr relation): " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth) + "\n")
    file.write("Best score: " + str(max_true) + " correct predictions with auc (tpr-fpr relation): " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth) + "\n")

    #fifth plot: prediction pie
    tn, fp, fn, tp = (confusion_matrix(max_y, max_prediction.T) / len(max_y)).ravel()
    print("validation: fp: " + str(fp) + " tp: " + str(tp) + " fn: " + str(fn) + " tn: " + str(tn))
    file.write("validation: fp: " + str(fp) + " tp: " + str(tp) + " fn: " + str(fn) + " tn: " + str(tn) + "\n")
    fig, ax1 = plt.subplots()
    ax1.pie([tp, fp, fn, tn], labels=["true-positive", "false-positive", "false-negative", "true-negative"],
            colors=['g', 'orange', 'r', 'b'], autopct='%1.1f%%', shadow=True)
    plt.title("prediction results on validation\nauc value: " + str(max_auc))
    plt.savefig(path + '\\validation_pie.png')

    #sixth plot: ROC curve
    plt.subplots(1, figsize=(10, 10))
    plt.title('Validation ROC curve')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(max_y, max_forest.predict_proba(max_x)[:, 1])
    plt.plot(false_positive_rate, true_positive_rate, linewidth=1.5)
    plt.plot([0, 1], ls="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path + '\\validation_ROC_curve.png')
    file.close()
    return max_forest


def factorize_data(data):
    """
    this function codes the stock name column according to the ascii values of the name letters.
    :param data: the data to work on
    """
    for i in range(len(data)):
        code = ""
        for z in data.iloc[i, 0]:
            code += str(ord(z))
        data.iloc[i, 0] = int(code)
        continue

def process_data(data, path):
    """
    this function normalize the data and concatenate the normalized values.
    It creates new pkl and csv files with the normalized data.
    :param data: the data to work on.
    :param path: path to save new files in
    """
    drops = [] #bad rows to drop
    for i in range(len(data)):
        for j in range(1, len(data.columns)-1):
            val = np.array(data.iloc[i,j])
            if data.columns[j] not in  ["RSI", "volume"]:
                if val[0] != 0:
                    val = val / val[0] #normalize by first value
                if len(val[val >= 10]) != 0: #erase row if stock value changed dramatically (probably wrong or exceptional case)
                    drops.append(i)
                    break
            elif data.columns[j] == "RSI":
                val = val / 100 #normalize by 100
                if len(val[val >= 1]) != 0: #erase row if found bad value that grater then 100
                    drops.append(i)
                    break
            else:
                if max(val) != 0:
                    val = val / max(val) #normalize by max value
            try:
                #concatenate all normalized values in list when their format is: d.dd where d is a digit
                data.iloc[i, j] = Decimal("".join(map("{:3.2f}".format, val)).replace(".", ""))
            except:
                drops.append(i)
                break
    data = data.drop(drops)
    data.to_pickle(path + '\\normalized_data.pkl')
    data.to_csv(path + '\\normalized_data.csv')

def run_tree_on_test(tree, test_data, path):
    """
    this function run the best tree found on the test set.
    :param tree: the best tree we found
    :param X_test_set: the test_set X
    :param y_test_set: the test set y
    :param path: path to write results to
    :return: prediction
    """
    file = open(path + "\\results.txt", "w")
    factorize_data(test_data)
    X_test_set = test_data.iloc[:, :-1]
    y_test_set =  test_data.iloc[:, -1]
    y_prediction = tree.predict(X_test_set)
    true_rate = 1 - (np.count_nonzero(y_test_set - y_prediction.T) / len(y_test_set))
    print("Test true prediction: " +  str(true_rate))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_set, tree.predict_proba(X_test_set)[:,1])

    #plot roc curve
    plt.subplots(1, figsize=(10, 10))
    plt.title('Test ROC curve')
    plt.plot(false_positive_rate, true_positive_rate, linewidth=1.5)
    plt.plot([0, 1], ls="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path + '\\test_ROC_curve.png')

    #creates pie chart of the prediction
    plt.figure()
    tn, fp, fn, tp = (confusion_matrix(y_test_set, y_prediction.T) / len(y_test_set)).ravel()
    print("test: fp: " + str(fp) + " tp: " + str(tp) + " fn: " + str(fn) + " tn: " + str(tn))
    file.write("test: fp: " + str(fp) + " tp: " + str(tp) + " fn: " + str(fn) + " tn: " + str(tn) + "\n")
    fig, ax1 = plt.subplots()
    ax1.pie([tp, fp, fn, tn], labels=["true-positive", "false-positive", "false-negative", "true-negative"],
            colors=['g', 'orange', 'r', 'b'], autopct='%1.1f%%', shadow=True)
    plt.title("prediction results on test\nauc value: " + str(auc(false_positive_rate, true_positive_rate)))
    plt.savefig(path + '\\test_pie.png')

    print("auc value: " + str(auc(false_positive_rate, true_positive_rate)))
    file.write("auc value: " + str(auc(false_positive_rate, true_positive_rate))+"\n")
    file.write("Test true prediction: " +  str(true_rate))
    file.close()
    return y_prediction

