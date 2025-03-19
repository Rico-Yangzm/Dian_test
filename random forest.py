import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

MAX_DEPTH = 3
MIN_OBJECTS = 2

from ucimlrepo import fetch_ucirepo

weight = lambda a, b, c, d: np.sum(a) * b + np.sum(c) * d
div = lambda a, b: a / b
def gini_impurity(labels):
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    class_probabilities = class_counts / n_samples
    gini = 1 - np.sum(class_probabilities ** 2)
    return gini



class TreeNode:
    def __init__(self, left = None, right= None, value= None, feature= None, threshold= None):
        self.left = left
        self.right = right
        self.value = value
        self.feature = feature
        self.threshold = threshold

def build_tree(x, y, depth, max_depth = MAX_DEPTH, min_objects = MIN_OBJECTS, max_features = None):
    samples, features = x.shape
    part = np.unique(y)

    if len(part) == 1 or depth >= MAX_DEPTH or samples < min_objects:
        return TreeNode(value = np.argmax(np.bincount(y)))

    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    if max_features is None:
        max_features = features
    selected_features = np.random.choice(features, max_features, replace=False)

    for feature in selected_features:
        thresholds = np.unique(x[:, feature])
        for threshold in thresholds:
            left_son = x[:, feature] <= threshold
            right_son = ~left_son
            if np.sum(left_son) == 0 or np.sum(right_son) == 0:
                continue
            gini_left = gini_impurity(y[left_son])
            gini_right = gini_impurity(y[right_son])
            weighted = div (weight (left_son, gini_left, right_son, gini_right), samples)

            if weighted < best_gini:
                best_gini = weighted
                best_feature = feature
                best_threshold = threshold

    left_idx = x[:, best_feature] <= best_threshold
    left_son = x[left_idx]
    right_son = x[~left_idx]
    left = build_tree(left_son, y[left_idx], depth + 1, max_depth, min_objects)
    right = build_tree(right_son, y[~left_idx], depth + 1, max_depth, min_objects)
    return TreeNode(left, right, None , best_feature, best_threshold)

def predict_tree(tree, sample):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature] <= tree.threshold:
        return predict_tree(tree.left, sample)
    else:
        return predict_tree(tree.right, sample)

class Forest:
    def __init__(self, tree_num=100 , max_depth = 3, min_objects = 2, max_feature = None):
        self.tree_num = tree_num
        self.max_depth = max_depth
        self.min_objects = min_objects
        self.max_feature = max_feature
        self.trees = []
        self.feature_importance = {}

    def train(self, x, y):
        feature = x.shape[1]
        self.feature_importance = {i: 0 for i in range(feature)}
        if self.max_feature is None:
            self.max_feature = int(np.sqrt(feature))

        for _ in range(self.tree_num):
            sample = x.shape[0]
            idx = np.random.choice(sample, sample, replace = True)
            oob_idx = list(set(range(sample)) - set(idx))
            init_x = x[idx]
            init_y = y[idx]
            tree_node = build_tree(init_x, init_y, 0, self.max_depth, self.min_objects, self.max_feature)
            self.trees.append((tree_node, oob_idx))
            self.update_feature_importance(tree_node)

    def prediction(self, x):
        predicts = []
        for tree in self.trees:
            predict = [predict_tree(tree, sample) for sample in x]
            predicts.append(predict)
        predicts = np.array(predicts)
        final = []
        for i in range(predicts.shape[1]):
            count = predicts[:, i]
            majority = Counter(count).most_common(1)[0][0]
            final.append(majority)
        print(final)
        return final

    def evaluate_oob(self, x, y):
        oob_predicts = [[] for _ in range(x.shape[0])]
        for tree, oob_idx in self.trees:
            for i in oob_idx:
                predict = predict_tree(tree, x[i])
                oob_predicts[i].append(predict)

        correct = 0
        total = 0
        for i, prediction in enumerate(oob_predicts):
            if len(prediction) > 0:
                predicted = Counter(prediction).most_common(1)[0][0]
                if predicted == y[i]:
                    correct += 1
                total += 1
                #print(predicted)
        accuracy = correct / total if total > 0 else 0

        return accuracy

    def update_feature_importance(self, node):
        if node.feature is not None:
            self.feature_importance[node.feature] += 1  # 统计特征使用次数
            self.update_feature_importance(node.left)
            self.update_feature_importance(node.right)

    def plot_feature_importance(self, feature_names):
        index = np.argsort(list(self.feature_importance.values()))[::-1]
        sorted_importance = np.array([self.feature_importance[i] for i in index])
        sorted_features = [feature_names[i] for i in index]

        if np.sum(sorted_importance) > 0:
            sorted_importance = sorted_importance / np.sum(sorted_importance)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
        plt.xticks(range(len(sorted_importance)), sorted_features)
        plt.ylabel('Feature Importance (Split Count)')
        plt.title('Random Forest Feature Importance')
        plt.show()


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

print("X 数据类型:")
print(X.dtypes)
print("y 数据类型:")
print(y.dtypes)

X = X.apply(pd.to_numeric, errors='coerce')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = X.to_numpy()
y = np.array(y)

print(iris.metadata)
print(iris.variables)

test = Forest()
test.train(X, y)
accuracy = test.evaluate_oob(X, y)
print(f"Out-of-Bag Accuracy: {accuracy}")
# 获取特征名称（假设使用鸢尾花数据集）
feature_names = iris.variables[iris.variables['role'] == 'Feature']['name'].tolist()

# 可视化特征重要性
test.plot_feature_importance(feature_names)
