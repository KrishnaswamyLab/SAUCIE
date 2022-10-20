from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target
