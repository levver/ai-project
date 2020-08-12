import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from decimal import Decimal
from math import log


def create_random_forest(X_train, y_train, X_val, y_val, forest_size, max_depth, f):
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
    f.write("validation prediction: " + str(true_rate) + " matches\n")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, val_prediction)
    return forest, auc(false_positive_rate, true_positive_rate), true_rate


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
    f = open("d:\Documents\Toar\AI\project\graphs\\results.txt", "w")
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

    all_trues = []
    all_depths = []
    all_aucs = []
    all_tree_sizes = []
    for i in range(iterations):
        auc_by_forest_size = np.array([0] * len(forest_size)).astype(float) #saves avg auc for each forest size
        auc_by_depth_size = np.array([0] * len(depth_size)).astype(float) #saves avg auc for each max depth
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size)
        print("percent of ones in validation: " + str(len(y_val[y_val==1])/len(y_val)))
        f.write("percent of ones in validation: " + str(len(y_val[y_val==1])/len(y_val))+"\n")
        for j, size in enumerate(forest_size):
            for z, depth in enumerate(depth_size):
                print("start iteration " + str(i) + " with " + str(size) + " trees and depth " + str(depth))
                f.write("\nstart iteration " + str(i) + " with " + str(size) + " trees and depth " + str(depth)+"\n")
                forest, auc, true_rate = create_random_forest(X_train, y_train, X_val, y_val, size, depth, f)
                print("tree auc value: " + str(auc))
                print("")
                f.write("tree auc value: " + str(auc) +"\n")
                f.write("\n")
                auc_by_forest_size[j] += auc
                auc_by_depth_size[z] += auc
                if auc > max_auc: #find max auc
                    max_forest = forest
                    max_auc = auc
                    max_trees = size
                    max_true = true_rate
                    max_depth = depth
                all_aucs.append(auc)
                all_trues.append(true_rate)
                all_tree_sizes.append(size)
                all_depths.append(depth)

        #first plot: avg auc value by forest size
        plt.figure()
        auc_by_forest_size = auc_by_forest_size / len(depth_size)
        auc_by_depth_size = auc_by_depth_size / len(forest_size)
        plt.plot(forest_size[::-1], auc_by_forest_size[::-1], 'b')
        plt.title("Iteration "+ str(i+1) +" average auc for each forest size")
        plt.ylabel("auc")
        plt.xlabel("forest size")
        plt.savefig('d:\Documents\Toar\AI\project\graphs\\forest_size_avg' + str(i) +'.png')

        # second plot: avg auc value by depth size
        plt.figure()
        plt.plot(depth_size[::-1], auc_by_depth_size[::-1], 'r')
        plt.title("Iteration "+ str(i+1) +" average auc for each max depth")
        plt.ylabel("auc")
        plt.xlabel("max depth")
        plt.savefig('d:\Documents\Toar\AI\project\graphs\max_depth_avg' + str(i) + '.png')

    #third plot: true predictions as a function of forest size and max depth
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_tree_sizes, all_depths, all_trues, c='r', marker='o')
    plt.title("true predictions as a function of forest size and max depth")
    ax.set_xlabel('forest size')
    ax.set_ylabel('max depth')
    ax.set_zlabel('true predictions rate')
    plt.savefig('d:\Documents\Toar\AI\project\graphs\\true_predictions_as_function.png')

    # forth plot: auc as a function of forest size and max depth
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_tree_sizes, all_depths, all_aucs, c='b', marker='o')
    plt.title("auc (rpr-fpr relation) as a function of forest size and max depth")
    ax.set_xlabel('forest size')
    ax.set_ylabel('max depth')
    ax.set_zlabel('auc value')
    plt.savefig('d:\Documents\Toar\AI\project\graphs\\auc_as_function.png')
    print("Best score: " + str(max_true) + " correct predictions with auc (tpr-fpr relation): " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth) + "\n")
    f.write("Best score: " + str(max_true) + " correct predictions with auc (tpr-fpr relation): " + str(max_auc) + " using " + str(max_trees) + " trees with depth " + str(max_depth) + "\n")
    f.close()
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

def process_data(data):
    """
    this function normalize the data and concatenate the normalized values.
    It creates new pkl and csv files with the normalized data.
    :param data: the data to work on.
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
    data.to_pickle(r'd:\Documents\Toar\AI\project\normalized_data.pkl')
    data.to_csv(r'd:\Documents\Toar\AI\project\normalized_data.csv')

def run_tree_on_test(tree, X_test_set, y_test_set):
    """
    this function run the best tree found on the test set.
    :param tree: the best tree we found
    :param X_test_set: the test_set X
    :param y_test_set: the test set y
    :return: prediction
    """
    y_prediction = tree.predict(X_test_set)
    true_rate = 1 - (np.count_nonzero(y_test_set - y_prediction.T) / len(y_test_set))
    print("Test true prediction: " +  str(true_rate))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_set, y_prediction)
    f = open("d:\Documents\Toar\AI\project\graphs\\results.txt", "w")
    print("auc value: " + str(auc(false_positive_rate, true_positive_rate)))
    f.write("auc value: " + str(auc(false_positive_rate, true_positive_rate))+"\n")
    f.write("Test true prediction: " +  str(true_rate))
    f.close()
    return y_prediction

#data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\normalized_data2.pkl')
data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\new_normalized_data.pkl')
data = data.iloc[:,list(range(231))+[-1]]#.sample(n=10000)
#data = pandas.read_csv(r'd:\Documents\Toar\AI\project\normalized_data2.csv')
#data.to_pickle(r'd:\Documents\Toar\AI\project\normalized_data2.pkl')
#data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\data_set.pkl')
#process_data(data)
factorize_data(data)
d_tree = return_decision_tree(data, 2)
test_data = pandas.read_pickle(r'd:\Documents\Toar\AI\project\test_normalized.pkl')
factorize_data(test_data)
run_tree_on_test(d_tree, test_data.iloc[:, :-1], test_data.iloc[:, -1])

