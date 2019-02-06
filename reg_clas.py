from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial import distance
print('\nChoose data below:\n1. Pima Indian Diabetes\n2. Housing')
data=int(input('Choose option number: '))
k=int(input("\nEnter k: "))
print('\nChoose distance method:\n1. Manhattan\n2. Euclidean\n3. Cosine\n4. Minkowski')
dMethod=int(input('Choose option number: '))
if(data==1):
	y = np.genfromtxt('pima.csv', usecols=(-1), delimiter=',')
	X = np.genfromtxt('pima.csv', usecols=range(8), delimiter=',')
else:
	y = np.genfromtxt('housing.csv', usecols=(-1))
	X = np.genfromtxt('housing.csv', usecols=range(13))
if(dMethod==1):
	dist='manhattan'
elif(dMethod==2):
	dist='euclidean'
elif(dMethod==3):
	dist='cosine'
elif(dMethod==4):
	dist='minkowski'
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
if(data==1):
	skf = StratifiedKFold(n_splits=10).split(X,y)
	if(dMethod==4):
		knn = KNeighborsClassifier(n_neighbors = k, p=5, metric=dist)
	else:
		knn = KNeighborsClassifier(n_neighbors = k, metric=dist)
	scores = model_selection.cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
	scr=0
	for i in range(len(scores)):
		scr+=scores[i]
	print('\nAccuracy:',100*scr/10,'%') 
else:
	kf = KFold(n_splits=10).split(X,y)
	result=[]
	for X_train, X_test in kf:
		for x in X_test:
			arr=[]
			for w in X_train:
				a=X[x]
				b=X[w]
				if(dMethod==2):
					dst=distance.euclidean(a,b)
				elif(dMethod==1):
					dst=distance.cityblock(a,b)
				elif(dMethod==3):
					dst=distance.cosine(a,b)
				elif(dMethod==4):
					dst=distance.minkowski(a,b,p=5)
				arr.insert(w,dst)
			for w in X_test:
				arr.insert(w,1000)
			knn=np.array(arr)
			knn=np.argsort(knn)[:k]
			res=0
			for i in range(k):
				res+=y[knn[i]]
			res=res/k
			res=abs(y[x]-res)/y[x]
			result.insert(x, res)
	final=0
	for m in range(len(X)):
		final+=result[m]
	final=final/len(X)
	print('\nMAPE:',100*final,'%')
