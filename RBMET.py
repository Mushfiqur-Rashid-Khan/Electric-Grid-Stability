#Python libraries
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#Dataset Upload
import pandas as pd
df=pd.read_csv("Dataset (Grid_Stability).csv")

from sklearn.model_selection import train_test_split
# Split data into features and labels
X = df.iloc[:,:13]
y = df["Pred"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Data Preprocessing
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

rbm = BernoulliRBM(n_components=10, n_iter=10, learning_rate=0.01, verbose=True)

# Created a pipeline combining the RBM and a logistic regression classifier (for supervised fine-tuning). The RBM will be used for unsupervised feature learning.

rbm_pipeline = Pipeline(steps=[('rbm', rbm)])

# Fitted the RBM to the normalized data (unsupervised learning)
rbm_pipeline.fit(X_normalized)

#Used the trained RBM for feature transformation
X_transformed = rbm_pipeline.transform(X_normalized)


#Extra Trees Classifier
et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=None)

# Fitted the Extra Trees classifier to the data
et_classifier.fit(X,y)

# The anomaly score represents the likelihood of each data point being an outlier.
anomaly_scores = et_classifier.predict_proba(X)[:, 1]  # Probability of being an anomaly (1)

# A threshold is set for stability detection
threshold = 0.5

# Generated stability labels based on the threshold
anomalies = anomaly_scores >= threshold

# Made predictions on the test set
y_pred = et_classifier.predict(X_test)

#Generation a Confusion Matrix
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Stable','Voltage Instability','Frequency Instability','Power Imbalance'])

#Visualization of the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

#Calculating Accuracy
Accuracy = metrics.accuracy_score(y_test, y_pred)
acc=Accuracy*100
print(acc)

#Calculating Precision
precision = metrics.precision_score(y_test, y_pred, average='micro')
prec=precision*100
print(prec)

#Calculating Recall
recall = metrics.recall_score(y_test, y_pred, average='micro')
rec=recall*100
print(rec)

#Calculating Training Time
import time
tic = time.time()
toc = time.time()
t=(toc - tic)
print(t)






