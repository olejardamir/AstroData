import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(n_samples_per_class)

        if depth == self.max_depth or np.all(y == y[0]):
            return {'predicted_class': predicted_class}

        best_split = self._best_split(X, y)
        if best_split is None:
            return {'predicted_class': predicted_class}

        left_indices, right_indices, split_feature, split_value = best_split
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'left': left_tree,
                'right': right_tree,
                'split_feature': split_feature,
                'split_value': split_value}

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None

        parent_entropy = self._entropy(y)
        best_info_gain = -1
        best_split = None

        for feature_idx in range(self.n_features_):
            feature_values = np.unique(X[:, feature_idx])
            for value in feature_values:
                left_indices = np.where(X[:, feature_idx] <= value)[0]
                right_indices = np.where(X[:, feature_idx] > value)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_entropy = self._entropy(y[left_indices])
                right_entropy = self._entropy(y[right_indices])
                weighted_entropy = (len(left_indices) * left_entropy + len(right_indices) * right_entropy) / n_samples

                info_gain = parent_entropy - weighted_entropy

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = (left_indices, right_indices, feature_idx, value)

        return best_split

    def _entropy(self, data):
        N = len(data)
        if N == 0:
            return 0

        fft_data = np.fft.fft(data)
        power_spectrum = np.abs(fft_data) ** 2

        if not np.isfinite(power_spectrum).all():
            return 1

        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 1

        high_freq_power = np.sum(power_spectrum[N // 4:]) / total_power
        randomness_score = 1 - high_freq_power

        return randomness_score

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree_) for x in X])

    def _predict_tree(self, x, tree):
        if 'predicted_class' in tree:
            return tree['predicted_class']

        feature_value = x[tree['split_feature']]
        if feature_value <= tree['split_value']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)

            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_sampled = X_sampled[:, feature_indices]

            tree.fit(X_sampled, y_sampled)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_sampled = X[:, feature_indices]
            tree_predictions = tree.predict(X_sampled)
            predictions[:, i] = tree_predictions

        y_pred = np.zeros(X.shape[0], dtype=int)
        for j in range(X.shape[0]):
            y_pred[j] = np.argmax(np.bincount(predictions[j]))

        return y_pred

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


# Example usage
X = np.random.randint(0, 16, size=(1000, 50))
y = np.random.randint(0, 64, size=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

rf = RandomForest(n_trees=100, max_depth=10, max_features=10)
rf.fit(X_train, y_train)

# Save the model
rf.save_model('random_forest_model.pkl')

# Load the model
loaded_rf = RandomForest.load_model('random_forest_model.pkl')

# Initial accuracy calculation
y_pred = loaded_rf.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)
print(f'Initial Accuracy: {initial_accuracy}')
