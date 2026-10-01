import streamlit as st
st.title("My Machine Learning Classifier App")

st.write("""
## Explore different classifier
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
      ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest"))

#get the dataset
def get_dataset(name):
    from sklearn import datasets
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(set(y)))

def add_parameter_ui(clf_name):
    #add one parameter for each classifier
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        leaf_size = st.sidebar.slider("leaf_size", 1, 100, 30)
        params["K"] = K
        params["leaf_size"] = leaf_size
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        degree = st.sidebar.slider("degree", 1, 10, 3)
        params["C"] = C
        params["degree"] = degree
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split
    return params
params = add_parameter_ui(classifier_name)

st.write("Classifier parameters:", params)

def get_classifier(clf_name, params):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"], leaf_size=params["leaf_size"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], degree=params["degree"], kernel='poly')
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], random_state=1234)
    return clf
clf = get_classifier(classifier_name, params)

#splitting our dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

#training the classifier
clf.fit(X_train, y_train)

#evaluating the classifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.write("Precision:", precision)
st.write("Confusion Matrix:", cm)

#Plot
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
x1 = X_pca[:, 0]
x2 = X_pca[:, 1]

st.write("PCA Plot:")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(x1, x2, c=y, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
st.pyplot(fig)