import scipy.io as sio
import numpy as np
from sklearn import preprocessing
#from convolutional_sparseFiltering import cross_validate_linearSVM, train_linearSVM
#
##poolFile = "../ufldl/sparseFiltering/features/SF_maxiter100_L1_md_20x20_skew4_SignPreserveNorm_with_confirmed1_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat"
##features = sio.loadmat(poolFile)
#
##pooledFeaturesTrain = features["pooledFeaturesTrain"]
##X = np.transpose(pooledFeaturesTrain, (0,2,3,1))
##numTrainImages = np.shape(X)[3]
##X = np.reshape(X, (int((pooledFeaturesTrain.size)/float(numTrainImages)), \
##                               numTrainImages), order="F")
##pooledFeaturesTrain = None
#data = sio.loadmat("../data/md/md_20x20_skew4_SignPreserveNorm_with_confirmed1.mat")
#y = data["y"]
#X = data["X"]
##pooledFeaturesTest = features["pooledFeaturesTest"]
##testX = np.transpose(pooledFeaturesTest, (0,2,3,1))
##numTestImages = np.shape(testX)[3]
##testX = np.reshape(testX, (int((pooledFeaturesTest.size)/float(numTestImages)), \
##                               numTestImages), order="F")
#testy = data["testy"]
#testX = data["testX"]
##features = None
##pooledFeaturesTest = None
#
##poolFile = "../ufldl/sparseFiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stl-10_unlabeled_meansub_20150409_psdb_6x6_pooled5.mat"
##features = sio.loadmat(poolFile)
#
#data1 = sio.loadmat("../data/3pi_20x20_skew2_signPreserveNorm.mat")
#y = np.concatenate((y,data1["y"]))
#testy = np.concatenate((testy, data1["testy"]))
##pooledFeaturesTrain = features["pooledFeaturesTrain"]
##X_1 = np.transpose(pooledFeaturesTrain, (0,2,3,1))
##numTrainImages = np.shape(X_1)[3]
##X_1 = np.reshape(X_1, (int((pooledFeaturesTrain.size)/float(numTrainImages)), \
##                   numTrainImages), order="F")
##pooledFeaturesTrain = None
##print np.shape(X),np.shape(X_1)
#X_1 = data1["X"]
#X = np.concatenate((X,X_1), axis=0)
#X_1 = None
#
##pooledFeaturesTest = features["pooledFeaturesTest"]
##testX_1 = np.transpose(pooledFeaturesTest, (0,2,3,1))
##numTestImages = np.shape(testX_1)[3]
##testX_1 = np.reshape(testX_1, (int((pooledFeaturesTest.size)/float(numTestImages)), \
##                           numTestImages), order="F")
#testX_1 = data1["testX"]
#testX = np.concatenate((testX, testX_1),axis=0)
#testX_1= None
##features = None
##pooledFeaturesTest = None
#X = np.concatenate((X,testX),axis=0)
#y = np.concatenate((y,testy))
#print np.shape(y)
#m = np.shape(y)[0]
#np.random.seed(0)
#order = np.random.permutation(m)
#X = X[order]
#y = np.squeeze(y[order])
#
#sio.savemat("md_3pi_20150729.mat",{"X":X[:.75*m],"y":y[:.75*m],"testX":testX[.75*m:],"testy":y[.75*m:]})

data = sio.loadmat("md_3pi_covolved_features_20150729.mat")
pooledFile = "md_3pi_covolved_features_20150729.mat"
dataFile = "md_3pi_features_20150729.mat"
X = data["X"]
y = data["y"]
print np.shape(X)
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)  # Don't cheat - fit only on training data
X = scaler.transform(X)
cross_validate_linearSVM(dataFile, X, np.squeeze(y), pooledFile, 20, False)


