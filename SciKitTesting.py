import pandas as pd
import numpy as np
from os import path
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import decomposition
import matplotlib.pylab as plt

# ****saving iris dataset as features and labels to disk*******************************
def saveDisk():
    with open("iris.data") as f:
        lines = [i[:-1] for i in f.readlines()]
    n = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
    x = [n.index(i.split(",")[-1]) for i in lines if i !=""]
    x = np.array(x, dtype ="uint8")

    y = [[float(j) for j in i.split(",")[:-1]] for i in lines if i !=""]
    y = np.array(y)

    i = np.argsort(np.random.random(x.shape[0]))
    x = x[i]
    y = y[i]

    np.save("iris_features.npy",y)
    np.save("iris_labels.npy",x)

# principle component analysis enables viewing the direction of variance in the dataset
# Uncomment to graph PCA for class1 and class2

# x = np.load("iris_features.npy")[:,:2]
# y = np.load("iris_labels.npy")
# idx = np.where(y != 0)
# x = x[idx]
# x[:,0]-=x[:,0].mean()
# x[:,1]-=x[:,1].mean()

# pca = decomposition.PCA(n_components=2)
# pca.fit(x)
# v = pca.explained_variance_ratio_

# plt.scatter(x[:,0],x[:,1],marker='o',color='b')
# ax = plt.axes()
# x0 = v[0]*pca.components_[0,0]
# y0 = v[0]*pca.components_[0,1]
# ax.arrow(0,0,x0,y0,head_width=0.05,head_length=0.1,fc='r',ec='r')
# x1 = v[1]*pca.components_[1,0]
# y1 = v[1]*pca.components_[1,1]
# ax.arrow(0,0,x1,y1,head_width=0.05,head_length=0.1,fc='r',ec='r')
# plt.xlabel("$x_0$",fontsize=16)
# plt.ylabel("$x_1$",fontsize=16)
# plt.show()

def generateData(pca,x,start):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)
    for i in range(start,ncomp):
        pca.components_[i,:]+=np.random.normal(scale=0.1,size=ncomp)
    b = pca.inverse_transform(a)
    pca.components_ = original.copy()
    return b

# loads all features and data and principal components: [0.92461621 0.05301557 0.01718514 0.00518309]
# First two p comps describe 97% of the iris variance
# used second two indices to generate new samples

# ****Augmenting the dataset using PCA******************************************************
def Augment():
    x = np.load("iris_features.npy")
    y = np.load("iris_labels.npy")

    N = 120

    x_train = x[:N]
    y_train = y[:N]
    x_test = x[N:]
    y_test = y[N:]

    pca = decomposition.PCA(n_components=4)
    pca.fit(x)
    print(pca.explained_variance_ratio_)
    start = 2
    nsets = 10
    nsamp = x_train.shape[0]
    newx = np.zeros((nsets*nsamp,x_train.shape[1]))
    newy = np.zeros(nsets*nsamp,dtype="uint8")

    for i in range(nsets):
        if(i==0):
            newx[0:nsamp,:]=x_train
            newy[0:nsamp]=y_train
        else:
            newx[(i*nsamp):(i*nsamp+nsamp),:]=generateData(pca,x_train,start)
            newy[(i*nsamp):(i*nsamp+nsamp)]=y_train

    idx = np.argsort(np.random.random(nsets*nsamp))
    newx=newx[idx]
    newy=newy[idx]
    np.save("iris_train_features_augmented.npy",newx)
    np.save("iris_train_labels_augmented.npy",newy)
    np.save("iris_test_features_augmented.npy",x_test)
    np.save("iris_test_labels_augmented.npy",y_test)

# ****esting Scikit Models********************************************************************

def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    print(" predicitions :",clf.predict(x_test))
    print(" actual labels:", y_test)
    print(" score = %0.4f"%clf.score(x_test,y_test))
    print()

def TestModels():
    x = np.load("iris_features.npy")
    y = np.load("iris_labels.npy")

    N = 120

    x_train = x[:N]; x_test = x[N:]
    y_train = y[:N]; y_test = y[N:]

    xa_train = np.load("iris_train_features_augmented.npy")
    ya_train = np.load("iris_train_labels_augmented.npy")
    xa_test = np.load("iris_test_features_augmented.npy")
    ya_test = np.load("iris_test_labels_augmented.npy")

    print("nearest centeroid:")
    run(x_train,y_train,x_test,y_test,NearestCentroid())
    print("k-NN classifier (k=3):")
    run(x_train,y_train,x_test,y_test,KNeighborsClassifier(n_neighbors=3))
    print("Naive Bayes classifier(Gaussian):")
    run(x_train,y_train,x_test,y_test,GaussianNB())
    print("Naive Bayes classifier(Multinominal):")
    run(x_train,y_train,x_test,y_test,MultinomialNB())
    print("decision tree classifier:")
    run(x_train,y_train,x_test,y_test,DecisionTreeClassifier())
    print("random forest classifer:(estimators=5):")
    run(xa_train,ya_train,xa_test,ya_test,RandomForestClassifier(n_estimators=5))

    print("SVM(linear, C=1.0):")
    run(xa_train,ya_train,xa_test,ya_test,SVC(kernel="linear",C=1.0))
    print("SVM(RBF, C=1.0, gamma=0.25):")
    run(xa_train,ya_train,xa_test,ya_test,SVC(kernel="rbf",C=1.0,gamma=0.25))
    print("SVM(RBF, C=1.0, gamma=0.001, augmented):")
    run(xa_train,ya_train,xa_test,ya_test,SVC(kernel="rbf",C=1.0,gamma=0.001))
    print("SVM(RBF, C=1.0, gamma=0.001, original):")
    run(x_train,y_train,x_test,y_test,SVC(kernel="rbf",C=1.0,gamma=0.001))

def main():
    print("Welcome to SciKit Model testing program! Wait a moment while all necessary datasets are populated.")
    if not path.exists("iris_features.npy"):
        try:
            saveDisk()
        except:
            pass
    if not path.exists("iris_train_features_augmented.npy"):
        try:
            Augment()
        except:
            pass
    TestModels()


if __name__ == "__main__":
    main()