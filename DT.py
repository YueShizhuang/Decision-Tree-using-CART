#读取并打印数据集：
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
df = pandas.read_csv('E:/pyproject/show.csv')
#数据预处理
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
print(df)
#分离特征与决策
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
print(X,y)
#创建一个决策树，将其另存为图像，然后显示该图像
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
'''
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('showdecisiontree.png')
img=pltimg.imread('showdecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
'''
#预测
print(dtree.predict([[40, 10, 7, 1]]))
