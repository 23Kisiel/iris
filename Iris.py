from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import pandas as pd  # Interaktywna konsola wspomagająca obliczenia
import numpy as np  # Biblioteka do obliczeń na n-wymiarowych macierzach
import matplotlib.pyplot as plt  # Biblioteka wykresów i grafów

plt.style.use('ggplot')

iris = datasets.load_iris()  # załadowanie danych do zmiennej iris
'''
# DESCRIPTION OF A DATA
print(type(iris))  #<class 'sklearn.utils.Bunch'> Bunch - similiar to a dictionary
print(iris.keys())  #dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(iris['DESCR'])
print(type(iris.data)) #<class 'numpy.ndarray'>
print(iris.data.shape) #(150, 4) #amount of rows and colums
'''

### EXPLORATORY DATA ANALYSIS(EDA)

features = iris.data  # X
type = iris.target    # y
df = pd.DataFrame(features, columns=iris.feature_names)

# print(df.head())  # nagłówki + pierwsze 5 wierszy
# _ = pd.plotting.scatter_matrix(df, c= type,  figsize = [8,10], s = 150 , marker = 'D')
# plt.show()

'''
BINARY VALUES EXAMPLE OF PLOT
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
'''

### Classification witk KNN
'''
# VISULSATION OF EXAMPLE
pl = df['petal length (cm)']
pw = df['petal width (cm)']

plt.scatter(pl, pw, label = 'pedal with vs length', c = type, marker = 'D')
plt.show()
'''

print(train_test_split)

knn = KNeighborsClassifier(n_neighbors=6) # metoda klasyfikacji (6 sąsiadów)
knn.fit(iris['data'], iris['target'])  # training data

X_new = np.array([[5.3, 3.6, 1.5, 0.2]]) #my example of train data
prediction = knn.predict(X_new)
#print('Prediction: {}'.format(prediction))

### TRAIN/TEST SPLI
X_train, X_test, y_train, y_test = train_test_split(features,type, test_size=0.3, random_state=21, stratify = type)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set Prediction: {}".format(y_pred))
print(knn.score(X_test, y_test))