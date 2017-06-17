# apples and oranges
# #
# Josh Gordon {ML} Recipes
#   -> https://youtu.be/cKxRvEZd3Mw?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
#
# Developed by Nathan Shepherd

from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()

# train classifier
clf = clf.fit(features, labels)

#make prediction
print(clf.predict([[160, 0]]))
