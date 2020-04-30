import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_digits

# charger les digits
digits = load_digits()

# afficher les matrices de tous les chiffres
i=0
for i in range(10):
    print("chiffre %i :\n" %i,digits.images[i])

# afficher les chiffres
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
plt.show()
