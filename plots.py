import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
data = [10,25,50,100,150,200]
scoreA = [0.45,  0.6699999999999999, 0.74, 0.8300000000000001, 0.8733333333333333, 0.9075]
scoreB = [0.425, 0.51,               0.6,  0.705,              0.7866666666666667, 0.8934210526315789]
ax.plot(data, scoreA, color='firebrick', label= "Object A")
ax.plot(data, scoreB, color='royalblue', label= "Object B")
ax.set_ylabel('Best Score')
ax.set_xlabel('data entries per tap area')
ax.set_title('Precision compared to training data size')
plt.legend(loc='best')
plt.show()
labels = ['KNN', 'DecisionTree', 'SVC', 'RandomForest', 'MLP']
objectA = [0.81375, 0.51125, 0.8950000000000001, 0.6300000000000001, 0.75125]
objectB = [0.7387500000000001, 0.54, 0.8934210526315789, 0.6862499999999999, 0.73]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objectA, width, color='firebrick', label='Object A')
rects2 = ax.bar(x + width/2, objectB, width, color='royalblue', label='Obejct B')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Best Scores')
ax.set_title('Grid search cross validation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()