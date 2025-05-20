from correlation_based_feature_selection import CorrelationBasedFeatureSelector
import sklearn.svm
import sklearn.model_selection
import numpy as np

# Load the dataset
from sklearn.datasets import fetch_openml

df = fetch_openml(name="madelon", version=1, as_frame=True, parser="pandas").frame
label = "Class"
features = [name for name in df.columns if name != label]
df[label] = df[label].astype("int") - 1
df[features] = df[features].astype("float")

# Get baseline performance using all features
X = df[features]
y = df[label]
svc = sklearn.svm.SVC(kernel="rbf", C=100, gamma=0.01, random_state=42)
scores = sklearn.model_selection.cross_val_score(svc, X, y, cv=10)
print(np.mean(scores))  # 0.5

# Get improvement by using only selected features
# the API uses a sklearn.feature_selection.SelectorMixin
# so it should work well in pipelines etc
X = CorrelationBasedFeatureSelector().fit_transform(df[features], df[label])
y = df[label]
svc = sklearn.svm.SVC(kernel="rbf", C=100, gamma=0.01, random_state=42)
scores = sklearn.model_selection.cross_val_score(svc, X, y, cv=10)
print(np.mean(scores))  # 0.6684... > 0.5 Improvement!
