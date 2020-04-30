from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import matplotlib.pyplot as plt
import pylab as pl

# charger le dataset, le data et le target
digits = datasets.load_digits()
features = digits.data
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

# modification de l'image
img = misc.imread("chiffre.png")
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img, high=16, low=0)


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)

# afficher le chiffre de l'image dans la console
var=clf.predict([x_test])
print("le chiffre est : ", var[0])
print("la matrice est : \n", digits.images[var[0]])

# afficher la figure
pl.gray()
pl.matshow(digits.images[var[0]])
pl.show()