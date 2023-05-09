#Imports
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, make_scorer, cohen_kappa_score
import pandas as pd
from keras.utils import to_categorical 
from statistics import mean

#Neural networks defining
def FCNN(unit, unit2):
    FCNN = tf.keras.models.Sequential()
    FCNN.add(tf.keras.layers.Dense(units = unit, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = unit2, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    FCNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return FCNN

def CNN(unit,unit2):
    CNN = tf.keras.models.Sequential()
    CNN.add(tf.keras.layers.Conv1D(unit, kernel_size= 5 ,activation='relu',))
    CNN.add(tf.keras.layers.Conv1D(unit2, kernel_size= 5, activation='relu'))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    CNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return CNN

def FCNN_train():
    FCNN = tf.keras.models.Sequential()
    FCNN.add(tf.keras.layers.Dense(units = 64, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = 32, activation='relu'))
    FCNN.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    FCNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return FCNN

def CNN_train():
    CNN = tf.keras.models.Sequential()
    CNN.add(tf.keras.layers.Conv1D(32, kernel_size= 5 ,activation='relu',))
    CNN.add(tf.keras.layers.Conv1D(32, kernel_size= 5, activation='relu'))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    CNN.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return CNN

#Otsu thresholding
def bgremove(img):
    ## convert to hsv
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # store the a-channel
    a_channel = lab[:,:,1]
    # Automate threshold using Otsu method
    th = cv.threshold(a_channel,127,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    # Mask the result with the original image\n",
    masked = cv.bitwise_and(img, img, mask = th)
    return masked

image = cv.imread("Training/Early_Blight/Early_Blight_1.jpg")#Test Display
image2 = cv.imread("Training/Healthy/Healthy_1.jpg")#Test Display
image3 = cv.imread("Training/Late_Blight/Late_Blight_1.jpg")#Test Display
#Plot Base image
ax1 = plt.subplot(1,3,1)
ax1.set_title("Early Blight")
plt.imshow(image)
#Plot darkened image
ax2 = plt.subplot(1,3,2)
ax2.set_title("Healthy")
plt.imshow(image2)
#plot removed background
ax3 = plt.subplot(1,3,3)
ax3.set_title("Late Blight")
plt.imshow(image3)

plt.show()
#Create lists to store features and labels
HOG = []
labels = []
new_size = (64,64)

#Loops Through every image in given dictionary
for root, subFolders, files in os.walk("Training/"):
    for file in files:
        file_name = root + "/" + file 
        image = cv.imread(file_name)#Reads image
        
        resize_image = cv.resize(image, new_size)
        #Enhance - Runs functions
        noBG = bgremove(resize_image)
        
        #HOG - Settings
        winSize = (32,32)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        h = hog.compute(noBG)#Caclualtes the HOG
        HOG.append(h)#Adds to array

        # Extract label from filename
        label = file_name.split("/")[-2] 
        labels.append(label)

#Create numpy arrays
X = np.array(HOG)
y = np.array(labels)
print(X.shape, y.shape)

#Reshape for neural network
NNX = X.reshape(3251, 8100, 1)

#Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

NNX_train, NNX_test, NNy_train, NNy_test = train_test_split(NNX, y, test_size=0.3)
print(NNX_train.shape, NNy_train.shape)

new_y_train = (np.unique(y_train, return_inverse=True)[1])
new_y_test = (np.unique(y_test, return_inverse=True)[1])
#-------------------------------------Hyper Parameter tuning----------------------------------------------------------
#Declaring the parameters to check for each classifier
param_SVC= {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['linear','rbf']}
param_KNN = {'n_neighbors':[1,2,3,4], 'leaf_size':[100, 200, 300, 400]}
param_RF = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8]}
param_FCNN = {'batch_size':[50,100,150],'epochs':[9, 12, 15], 'unit':[32, 64], 'unit2':[32, 16]}
FCNN_Model = KerasClassifier(FCNN)
param_CNN = {'batch_size':[50,100],'epochs':[3, 6], 'unit':[64, 32], 'unit2':[32, 16]}
CNN_Model = KerasClassifier(CNN)

#Scores to evaulte models
f1 = make_scorer(f1_score, average='macro')
kappa_scorer = make_scorer(cohen_kappa_score)
scores = {
    'accuracy': 'accuracy',
    'f1': f1,
    'cohen':kappa_scorer
    }

#Run hyper parater tests
#Note tests commented out to save computational power and time - see results in report


TUNE_SVC = GridSearchCV(estimator=SVC(), param_grid=param_SVC, cv=10, scoring=scores, refit='accuracy')#Define model to be tested
TUNE_SVC.fit(X_train,y_train)#Fit data to use to test parameters
print("HyperParameter results")
print(TUNE_SVC.cv_results_)#Results
print("")
print("Best HyperParameters:")#Best parameters
print(TUNE_SVC.best_estimator_)


print("Summary of results")
df = pd.DataFrame({'parammeters': TUNE_SVC.cv_results_["params"], 'Mean_Accuracy': TUNE_SVC.cv_results_["mean_test_accuracy"], 'Mean_F1': TUNE_SVC.cv_results_["mean_test_f1"], 'Mean_Cohen_K': TUNE_SVC.cv_results_["mean_test_cohen"]})
print(df)#Create a dataframe for ease of viewing

TUNE_KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_KNN, cv=10, scoring=scores, refit='accuracy')
TUNE_KNN.fit(X_train,y_train)
print("HyperParameter results")
print(TUNE_KNN.cv_results_)
print("")
print("Best HyperParameters:")
print(TUNE_KNN.best_estimator_)


print("Summary of results")
df = pd.DataFrame({'parammeters': TUNE_KNN.cv_results_["params"], 'Mean_Accuracy': TUNE_KNN.cv_results_["mean_test_accuracy"], 'Mean_F1': TUNE_KNN.cv_results_["mean_test_f1"], 'Mean_Cohen_K': TUNE_KNN.cv_results_["mean_test_cohen"]})
print(df)#Create a dataframe for ease of viewing


TUNE_FCNN = GridSearchCV(estimator=FCNN_Model, param_grid=param_FCNN, cv=10, scoring=scores, refit='accuracy')
gs = TUNE_FCNN.fit(X_train,y_train)
print("HyperParameter results")
print(gs.cv_results_)
print("")
print("Best HyperParameters:")
t = gs.best_estimator_
print(t)


print("Summary of results")
df = pd.DataFrame({'parammeters': gs.cv_results_["params"], 'Mean_Accuracy': gs.cv_results_["mean_test_accuracy"], 'Mean_F1': gs.cv_results_["mean_test_f1"], 'Mean_Cohen_K': gs.cv_results_["mean_test_cohen"]})
print(df)#Create a dataframe for ease of viewing


TUNE_CNN = GridSearchCV(estimator=CNN_Model, param_grid=param_CNN, cv=10, scoring=scores, refit='accuracy')
gs = TUNE_CNN.fit(NNX_train,NNy_train)
print("HyperParameter results")
print(gs.cv_results_)
print("")
print("Best HyperParameters:")
t = gs.best_estimator_
print(t)


print("Summary of results")
df = pd.DataFrame({'parammeters': gs.cv_results_["params"], 'Mean_Accuracy': gs.cv_results_["mean_test_accuracy"], 'Mean_F1': gs.cv_results_["mean_test_f1"], 'Mean_Cohen_K': gs.cv_results_["mean_test_cohen"]})
print(df)#Create a dataframe for ease of viewing


TUNE_RF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_RF, cv=10, scoring=scores, refit='accuracy')
TUNE_RF.fit(X_train,y_train)
print("HyperParameter results")
print(TUNE_RF.cv_results_)
print("")
print("Best HyperParameters:")
print(TUNE_RF.best_estimator_)

print("Summary of results")
df = pd.DataFrame({'parammeters': TUNE_RF.cv_results_["params"], 'Mean_Accuracy': TUNE_RF.cv_results_["mean_test_accuracy"], 'Mean_F1': TUNE_RF.cv_results_["mean_test_f1"], 'Mean_Cohen_K': TUNE_RF.cv_results_["mean_test_cohen"]})
print(df)#Create a dataframe for ease of viewing
#---------------------------HyperParameter End------------------------------
#---------------------------Models Creation------------------------------

clf_svc = SVC(C=10, gamma=0.001, kernel='rbf')#Build models
clf_svc.fit(X_train,y_train)#train models
SCVaccuracy = cross_val_score(clf_svc, X_train, y_train, scoring='accuracy', cv = 10)
SCVF1Acc = cross_val_score(clf_svc, X_train, y_train, scoring=f1, cv = 10)
SCVcohen = cross_val_score(clf_svc, X_train, y_train, scoring=kappa_scorer, cv = 10)

print("Model_accuracy: ", mean(SCVaccuracy))
print("Model_f1: ", mean(SCVF1Acc))
print("Model_'cohen': ", mean(SCVcohen))

pred = clf_svc.predict(X_test)#predict for models

matrix = confusion_matrix(y_test, pred)#confsuin matrix for percsion
cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix)

print("Prediction Accuracy = ", accuracy_score(y_test, pred))#check accurcy
print("Prediction f1 = ", f1_score(y_test, pred, average='macro'))
print("Prediction choen = ", cohen_kappa_score(y_test, pred))

cm_display.plot()
plt.show()

estimator = KerasClassifier(build_fn=FCNN_train, epochs=3, batch_size=100)
FCNNMdoel = estimator.fit(X_train, new_y_train)
FCNN_accuracy = cross_val_score(estimator, X_train, new_y_train, cv = 10)
FCNN_F1_score = cross_val_score(estimator, X_train, new_y_train, scoring=f1, cv = 10)
FCNN_Choen = cross_val_score(estimator, X_train, new_y_train, scoring=kappa_scorer, cv = 10)

print("Model_accuracy: ", mean(FCNN_accuracy))
print("Model_f1: ", mean(FCNN_F1_score))
print("Model_'cohen': ", mean(FCNN_Choen))

FCNNpredictions = estimator.predict(X_test)

matrix_FCNN = confusion_matrix(new_y_test, FCNNpredictions)#confsuin matrix for percsion
cm_display_FCNN = ConfusionMatrixDisplay(confusion_matrix = matrix_FCNN)

print("Prediction Accuracy = ", accuracy_score(new_y_test, FCNNpredictions))#check accurcy
print("Prediction f1 = ", f1_score(new_y_test, FCNNpredictions, average='macro'))
print("Prediction choen = ", cohen_kappa_score(new_y_test, FCNNpredictions))
cm_display_FCNN.plot()
plt.show()


estimatorCNN = KerasClassifier(build_fn=CNN_train, epochs=3, batch_size=100)
FCNNMdoel = estimatorCNN.fit(NNX_train, new_y_train)
CNN_accuracy = cross_val_score(estimatorCNN, NNX_train, new_y_train, cv = 10)
CNN_F1_score = cross_val_score(estimatorCNN, NNX_train, new_y_train, scoring=f1, cv = 10)
CNN_Choen = cross_val_score(estimatorCNN, NNX_train, new_y_train, scoring=kappa_scorer, cv = 10)

print("Model_accuracy: ", mean(CNN_accuracy))
print("Model_f1: ", mean(CNN_F1_score))
print("Model_'cohen': ", mean(CNN_Choen))

CNNpredictions = estimatorCNN.predict(NNX_test)

matrix_CNN = confusion_matrix(new_y_test, CNNpredictions)#confsuin matrix for percsion
cm_display_CNN = ConfusionMatrixDisplay(confusion_matrix = matrix_CNN)

print("Prediction Accuracy = ", accuracy_score(new_y_test, CNNpredictions))#check accurcy
print("Prediction f1 = ", f1_score(new_y_test, CNNpredictions, average='macro'))
print("Prediction choen = ", cohen_kappa_score(new_y_test, CNNpredictions))
cm_display_CNN.plot()
plt.show()


clf_KNN = KNeighborsClassifier(leaf_size=100, n_neighbors=1)#Build models
clf_KNN.fit(X_train,y_train)#train models
KNNaccuracy = cross_val_score(clf_KNN, X_train, y_train, scoring='accuracy', cv = 10)
KNNF1_acc = cross_val_score(clf_KNN, X_train, y_train, scoring=f1, cv = 10)
KNNcohen = cross_val_score(clf_KNN, X_train, y_train, scoring=kappa_scorer, cv = 10)

print("Model_accuracy: ", mean(KNNaccuracy))
print("Model_f1: ", mean(KNNF1_acc))
print("Model_'cohen': ", mean(KNNcohen))


predKNN = clf_KNN.predict(X_test)#predict for models

matrix_KNN = confusion_matrix(y_test, predKNN)#confsuin matrix for percsion
cm_display_KNN = ConfusionMatrixDisplay(confusion_matrix = matrix_KNN)

print("Prediction Accuracy = ", accuracy_score(y_test, predKNN))#check accurcy
print("Prediction f1 = ", f1_score(y_test, predKNN, average='macro'))
print("Prediction choen = ", cohen_kappa_score(y_test, predKNN))
cm_display_KNN.plot()
plt.show()

clf_RF = RandomForestClassifier(max_features='auto', max_depth=8, n_estimators=500)#Build models
clf_RF.fit(X_train,y_train)#train models
RFaccuracy = cross_val_score(clf_RF, X_train, y_train, scoring='accuracy', cv = 10)
RFF1_acc = cross_val_score(clf_RF, X_train, y_train, scoring=f1, cv = 10)
RFcohen = cross_val_score(clf_RF, X_train, y_train, scoring=kappa_scorer, cv = 10)

print("Model_accuracy: ", mean(RFaccuracy))
print("Model_f1: ", mean(RFF1_acc))
print("Model_'cohen': ", mean(RFcohen))

predRF = clf_RF.predict(X_test)#predict for models

matrix_RF = confusion_matrix(y_test, predRF)#confsuin matrix for percsion
cm_display_RF = ConfusionMatrixDisplay(confusion_matrix = matrix_RF)

print("Prediction Accuracy = ", accuracy_score(y_test, predRF))#check accurcy
print("Prediction f1 = ", f1_score(y_test, predRF, average='macro'))
print("Prediction choen = ", cohen_kappa_score(y_test, predRF))
cm_display_RF.plot()
plt.show()