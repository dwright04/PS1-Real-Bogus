import sys, multiprocessing, pickle
import scipy.io as sio
sys.path.append("/Users/dew/development/PS1-Real-Bogus/tools/")
import multiprocessingUtils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class trainRF(multiprocessingUtils.Task):
    def __init__(self, X, y, dataFile, fold, rf):
        self.X = X
        self.y = np.squeeze(y)
        self.dataFile = dataFile
        self.fold = fold
        self.rf = rf
        self.n_estimators= rf.n_estimators
        self.max_features = rf.max_features
        self.min_samples_leaf = rf.min_samples_leaf

    def __call__(self):
        self.rf.fit(self.X, self.y)
        outputFile = open("cv/RF_n_estimators"+str(self.n_estimators)+\
                          "_max_features"+str(self.max_features)+"_min_samples_leaf"+str(self.min_samples_leaf)+\
                          "_"+self.dataFile.split("/")[-1].split(".")[0]+"_fold_"+str(self.fold)+".pkl", "wb")
        pickle.dump(self.rf, outputFile)
        outputFile.close()
        return 0
    
    def __str__(self):
        return "### Training Random Forest with n_estimator = %f, max_features = %f, min_samples_leaf = %d ###" \
               % (self.n_estimators, self.max_features, self.min_samples_leaf)

def generateFolds(dataFile):
    data = sio.loadmat(dataFile)
    scaler = preprocessing.StandardScaler().fit(data["X"])
    train_x = scaler.transform(np.concatenate((data["X"], data["validX"])))
    train_y = np.concatenate((data["y"], data["validy"]))
    #dataConfirmed = sio.loadmat("confirmedListVectors_minus13avb.mat")
    #train_x = np.concatenate((train_x, dataConfirmed["X"]))
    #train_y = np.concatenate((train_y, np.ones((np.shape(dataConfirmed["X"])[0],1))))
    #train_x = np.concatenate((train_x, data["testX"]))
    #train_y = np.concatenate((train_y, data["testy"]))
    m = np.shape(train_x)[0]
    np.random.seed(22)
    order = np.random.permutation(m)
    train_x = train_x[order]
    train_y = train_y[order]
    nFolds = 5
    foldSize = (m / nFolds)
    folds_X = []
    folds_y = []
    for i in range(nFolds):
        start = i*foldSize
        stop = (i+1)*foldSize
        if i == (nFolds-1):
            stop = None
        folds_X.append(train_x[start:stop])
        folds_y.append(train_y[start:stop])

    sets_X = [[folds_X[i-j] for i in range(nFolds)] for j in range(nFolds)] # cyclic permutation
    sets_y = [[folds_y[i-j] for i in range(nFolds)] for j in range(nFolds)]
    
    trainSets_X_tmp = [sets_X[i][:4] for i in range(nFolds)] # select first 4 sets from each permutation
    trainSets_y_tmp = [sets_y[i][:4] for i in range(nFolds)]
    testSets_X = [sets_X[i][-1] for i in range(nFolds)] # select last set from each permutation
    testSets_y = [sets_y[i][-1] for i in range(nFolds)]
    
    trainSets_X = []
    trainSets_y = []
    for i in range(nFolds):
        #print np.shape(testSets_X[i])
        trainSets_X.append([item for sublist in trainSets_X_tmp[i] for item in sublist])
        #print np.shape(trainSets_X[i])
        trainSets_y.append([item for sublist in trainSets_y_tmp[i] for item in sublist])

    return trainSets_X, trainSets_y, testSets_X, testSets_y

def main(argv = None):
    
    if argv is None:
        argv = sys.argv
    
    if len(argv) != 2:
        sys.exit("Usage: multicore_rf_gridSearch.py <.mat file>")

    dataFile = argv[1]
   
    trainSets_X, trainSets_y, testSets_X, testSets_y = generateFolds(dataFile)
    n_estimatorsGrid = [1000] # ntree in Brink et al. 2013
    #max_featuresGrid = [2,4,10,25,50,100] # mtry in Brink et al. 2013
    #min_samples_leafGrid = [1,2,4]
    #max_featuresGrid = [2,4,25,50] # mtry in Brink et al. 2013
    max_featuresGrid = [20,25,50]
    min_samples_leafGrid = [1,2,4]

    taskList = []

    nFolds = 5
    for i in range(nFolds):
        X = trainSets_X[i]
        y = trainSets_y[i]
        for n_estimators in n_estimatorsGrid:
            for max_features in max_featuresGrid:
                for min_samples_leaf in min_samples_leafGrid:
                    taskList.append(trainRF(X, y, dataFile, i+1, \
                                            RandomForestClassifier(n_estimators=n_estimators, \
                                                                   max_features=max_features, \
                                                                   min_samples_leaf=min_samples_leaf)))
    cpu_count = multiprocessing.cpu_count() - 2
    
    print "%d available CPUs.\n" % (cpu_count)

    multiprocessingUtils.multiprocessTaskList(taskList, cpu_count)
    """
    data = sio.loadmat(dataFile)
    train_x = np.concatenate((data["X"], data["validX"]))
    train_y = np.concatenate((data["y"], data["validy"]))
    #dataFile2 = "/Users/dew/development/PS1-Real-Bogus/data/3pi/3pi_20x20_realOnly_signPreserveNorm.mat"
    #data = sio.loadmat(dataFile2)
    #m = np.shape(data["X"])[0]
    #train_x = np.concatenate((train_x, data["X"]))
    #train_y = np.concatenate((np.squeeze(train_y), np.ones((m,))))
    #dataConfirmed = sio.loadmat("confirmedListVectors_minus13avb.mat")
    #train_x = np.concatenate((train_x, dataConfirmed["X"]))
    #train_y = np.concatenate((train_y, np.ones((np.shape(dataConfirmed["X"])[0],1))))

    #dataFile = dataFile.split("/")[-1].split(".")[0] + dataFile2.split("/")[-1].split(".")[0]
    m = np.shape(train_x)[0]

    n_estimatorsGrid = [1000] # ntree in Brink et al. 2013
    max_featuresGrid = [20,25,50,100]
    min_samples_leafGrid = [1]

    order = np.random.permutation(m)
    train_x = train_x[order]
    train_y = np.squeeze(train_y[order])
    print np.shape(train_x)
    for n_estimators in n_estimatorsGrid:
        for max_features in max_featuresGrid:
            for min_samples_leaf in min_samples_leafGrid:
                rf = RandomForestClassifier(n_estimators=n_estimators, \
                                            max_features=max_features, \
                                            min_samples_leaf=min_samples_leaf)
                rf.fit(train_x, train_y)
                outputFile = open("RF_n_estimators"+str(n_estimators)+\
                  "_max_features"+str(max_features)+"_min_samples_leaf"+str(min_samples_leaf)+\
                  "_"+dataFile.split("/")[-1].split(".")[0]+".pkl", "wb")
                pickle.dump(rf, outputFile)
    """
if __name__ == "__main__":
    main()
