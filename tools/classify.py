import pickle, sys, optparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, f1_score
from sklearn import preprocessing
sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/demos/")
import mlutils

def predict(clfFile, X):

    clf = pickle.load(open(clfFile, "rb"))
    pred = clf.predict_proba(X)[:,1]
    return pred
	
def hypothesisDist(y, pred, threshold=0.5):

    # the raw predictions for actual garbage
    garbageHypothesis = pred[np.where(y == 0)[0]]
    realHypothesis = pred[np.where(y == 1)[0]]
                                                                          
    font = {"size"   : 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)
    
    #plt.yticks([x for x in np.arange(500,3500,500)])                                                
    bins = [x for x in np.arange(0,1.04,0.04)]
    
    real_counts, bins, patches = plt.hist(realHypothesis, bins=bins, alpha=1, \
                                          label="real", color="#FF0066", edgecolor="none")
    print real_counts
    garbage_counts, bins, patches = plt.hist(garbageHypothesis, bins=bins, alpha=1, \
                                             label="bogus", color="#66FF33", edgecolor="none")
    print garbage_counts
    # calculate where the real counts are less than the garbage counts.
    # these are to be overplotted for clarity
    try:
        real_overlap = list(np.where(np.array(real_counts) <= np.array(garbage_counts))[0])
        for i in range(len(real_overlap)):
            to_plot = [bins[real_overlap[i]], bins[real_overlap[i]+1]]
            plt.hist(realHypothesis, bins=to_plot, alpha=1, color="#FF0066", edgecolor="none")
    except IndexError:
        pass
    
    max = int(np.max(np.array([np.max(real_counts), np.max(garbage_counts)])))
    print max
    decisionBoundary = np.array([x for x in range(0,max,100)])
    
    if garbage_counts[0] != 0:
        plt.text(0.01, 1000, str(garbage_counts[0]), rotation="vertical", size=22)
    
    plt.plot(threshold*np.ones(np.shape(decisionBoundary)), decisionBoundary, \
             "k--", label="decision boundary=%.3f"%(threshold), linewidth=2.0)
             
    y_min = -0.02*int(plt.axis()[-1])
    y_max = plt.axis()[-1]
    plt.xlim(-0.015,1.015)
    plt.ylim(y_min,y_max)
    #plt.title(dataFile.split("/")[-1])
    plt.xlabel("Hypothesis")
    plt.ylabel("Frequency")
    #plt.legend(loc="upper center")
    plt.show()

def plot_ROC(Ys, preds, indices, color="#FF0066", Labels=None):

    fig = plt.figure()
    font = {"size": 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)

    plt.xlabel("Missed Detection Rate (MDR)")
    plt.ylabel("False Positive Rate (FPR)")
    plt.yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    plt.ylim((0,1.05))
    default_ticks = [0, 0.05, 0.10, 0.25]
    ticks = []
    
    colours = ["#FF0066", "#66FF33", "#3366FF"]
    #Labels = ["Wright+15", "no normalisation"]

    for j,pred in enumerate(preds):
        y = Ys[indices[preds.index(pred)]] 
        fpr, tpr, thresholds = roc_curve(y, pred)
    
        FoMs = []
        decisionBoundaries = []
        if len(preds) == 1:
            FPRs = [0.01, 0.05, 0.1]
        else:
            FPRs = [0.01]
            color = colours[j]
            #label=Labels[j]

        for FPR in FPRs:
            FoMs.append(1-tpr[np.where(fpr<=FPR)[0][-1]])
            decisionBoundaries.append(thresholds[np.where(fpr<=FPR)[0][-1]])
        
        plt.plot(1-tpr, fpr, "k-", lw=5)
        #color = "#FF0066" # pink
        #color = "#66FF33" # green
        #color = "#3366FF" #blue
        #print label
        #plt.plot(1-tpr, fpr, color=color, lw=4, label=label)
        plt.plot(1-tpr, fpr, color=color, lw=4)
        for i,FoM in enumerate(FoMs):
            print "[+] FoM at %.3f FPR : %.3f | decision boundary : %.3f " % (FPRs[i], FoM, decisionBoundaries[i])
            plt.plot([x for x in np.arange(0,FoM+1e-3,1e-3)], \
                     FPRs[i]*np.ones(np.shape(np.array([x for x in np.arange(0,FoM+1e-3,1e-3)]))), \
                     "k--", lw=3, zorder=100)
            
            plt.plot(FoM*np.ones(np.shape([x for x in np.arange(0,FPRs[i]+1e-3, 1e-3)])), \
                     [x for x in np.arange(0,FPRs[i]+1e-3, 1e-3)], "k--", lw=3, zorder=100)
            #print round(FoM, 2)
            if round(FoM,2) in default_ticks:
                default_ticks.remove(round(FoM,2))
                ticks.append(FoM)
            else:
                ticks.append(FoM)
        plt.xticks(default_ticks+ticks, rotation=70)
                 
        locs, labels = plt.xticks()
        plt.xticks(locs, map(lambda x: "%.3f" % x, locs))
    if Labels:
        plt.legend()
    plt.show()

def test_FDR_procedure(y, pred):
    # get all the test cases where the null hypothesis is true
    null_true_pred = pred[y==0]
    # get bins for predicted values
    bins = [x for x in np.arange(0,1.001,0.001)]
    # determine the counts of true null hypothesis cases in
    # each bin
    counts, bins = np.histogram(null_true_pred, bins=bins)
    # calculate the normalised cumulative sum of the counts in
    # each bin.
    cumsum_norm = np.cumsum(counts) / float(len(null_true_pred))
    cumsum_norm = cumsum_norm.tolist()
    cumsum_norm.insert(0,1)
    cumsum_norm = np.array(cumsum_norm)
    #plt.plot(bins[:-1], cumsum_norm)
    #plt.xlim(xmin=-0.01, xmax=1.01)
    #plt.ylim(ymin=-0.01, ymax=1.01)
    #plt.show()
    # p-values are probability that a test element would
    # be observed in a given bin assuming the null hypothesis
    # is true for that element.
    p_values_per_bin = (1 - cumsum_norm)
    #print p_values_per_bin
    # set alpha the FDR to 0.01
    alpha = 0.01
    # divide the test set into 2 different sized chunks to test the
    # adaptive thresholding.
    pred_chunk_1 = pred[:-200]
    #print pred_chunk_1
    pred_chunk_2 = pred[-200:]
    # get number of tests
    num_tests_chunk_1 = float(len(pred_chunk_1))
    #print num_tests_chunk_1
    # calculate p-values for each example in chunk 1
    indices = np.digitize(pred_chunk_1, bins[:-1])
    # get and sort p-values
    p_vals_chunk_1 = [p_values_per_bin[i-1] for i in indices]
    p_vals_chunk_1.sort()
    j_alpha_chunk_1 = [j*alpha/num_tests_chunk_1 for j in range(1, int(num_tests_chunk_1)+1)]
    #for j in range(int(num_tests_chunk_1)):
    #    print j*alpha/num_tests_chunk_1
    diff_chunk_1 = np.array(p_vals_chunk_1) - np.array(j_alpha_chunk_1)
    print np.where(diff_chunk_1<0)[0][-1]
    #print p_vals_chunk_1
    print p_vals_chunk_1[np.where(diff_chunk_1<0)[0][-1]]
    print 


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def FoM(y, pred, threshold=0.5):
    # the raw predictions for actual garbage
    garbageHypothesis = pred[np.where(y == 0)[0]]
    realHypothesis = pred[np.where(y == 1)[0]]
    
    fpr, tpr, thresholds = roc_curve(y, pred)

    threshold = find_nearest(thresholds, threshold)
    print find_nearest(thresholds, threshold)
    print 1-tpr[np.squeeze(np.where(thresholds==threshold))]
    print "[+] FPR: %.3f" % fpr[np.squeeze(np.where(thresholds==threshold))]
    print "[+] FoM: %.3f" % (1-tpr[np.squeeze(np.where(thresholds==threshold))])

def getUniqueIds(files):
    
    ids = []
    for file in files:
        ids.append(file.rstrip().split("_")[0])
    return set(ids)
    
def predict_byName(pred, files, outputFile):
    
    files = files.tolist()
    ids = getUniqueIds(files)
    predictions_byName = {}
    for id in ids:
        predictions_byName[id] = []
        for file in files:
            if file.rstrip().split("_")[0] == id:
                predictions_byName[id].append(pred[files.index(file)])
    output = open(outputFile, "w")
    pred = np.zeros((len(predictions_byName.keys(),)))
    for i,key in enumerate(predictions_byName.keys()):
        #print key, np.median(np.array(predictions_byName[key])), len(predictions_byName[key])
        pred[i] += np.median(np.array(predictions_byName[key]))
        output.write(str(key) +"," + str(np.median(np.array(predictions_byName[key]))) + "\n")
    output.close()
    return pred

def labels_byName(files, y):
    files = files.tolist()
    ids = getUniqueIds(files)
    labels_byName = {}
    for id in ids:
        labels_byName[id] = []
        for file in files:
            if file.rstrip().split("_")[0] == id:
                labels_byName[id].append(y[files.index(file)])
    labels = np.zeros(np.shape(labels_byName.keys()));
    for i,key in enumerate(labels_byName.keys()):
        labels[i] += np.argmax(np.bincount(labels_byName[key]))
    return labels

def main():
    

    parser = optparse.OptionParser("[!] usage: python classify.py\n"+\
                                   " -F <data files [comma-separated]>\n"+\
                                   " -c <classifier files [comma-separated]>\n"+\
                                   " -t <threshold [default=0.5]>\n"+\
                                   " -s <data set>\n"+\
                                   " -o <output file>\n"
                                   " -p <plot hypothesis distribution [optional]>\n"+\
                                   " -r <plot ROC curve [optional]>\n"+\
                                   " -f <output Figure of Merit [optional]>\n"+\
                                   " -n <classify by name [optional]>\n"
                                   " -P <pooled features file [optional]>")

    parser.add_option("-F", dest="dataFiles", type="string", \
                      help="specify data file[s] to analyse")
    parser.add_option("-c", dest="classifierFiles", type="string", \
                      help="specify classifier[s] to use")
    parser.add_option("-t", dest="threshold", type="float", \
                      help="specify decision boundary threshold [default=0.5]")
    parser.add_option("-o", dest="outputFile", type="string", \
                      help="specify output file")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse [default=test]")
    parser.add_option("-p", action="store_true", dest="plot", \
                      help="specify whether to plot the hypothesis distribution [optional]")
    parser.add_option("-r", action="store_true", dest="roc", \
                      help="specify whether to plot the ROC curve [optional]")
    parser.add_option("-f", action="store_true", dest="fom", \
                      help="specify whether to output Figure of Merit to stdout [optional]")
    parser.add_option("-n", action="store_true", dest="byName", \
                      help="specify whether to classify objects by name [optional]")
    parser.add_option("-P", dest="poolFile", type="string", \
                      help="specify pooled features file [optional]")
                          
    (options, args) = parser.parse_args()
    
    try:
        dataFiles = options.dataFiles.split(",")
        classifierFiles = options.classifierFiles.split(",")
        threshold = options.threshold
        outputFile = options.outputFile
        dataSet = options.dataSet
        plot = options.plot
        roc = options.roc
        fom = options.fom
        byName = options.byName
        poolFile = options.poolFile
    except AttributeError:
        print parser.usage
        exit(0)

    if dataFiles == None or classifierFiles == None:
        print parser.usage
        exit(0)
    
    if threshold == None:
        threshold = 0.5
    
    if dataSet == None:
        dataSet = "test"
        
    Xs = []
    Ys = []
    Files = []
    for dataFile in dataFiles:
        data = sio.loadmat(dataFile)
        print "[+] %s" % dataFile
        X = data["X"]
        scaler = preprocessing.StandardScaler(with_std=False).fit(X)
        if dataSet == "test":
            try:
                #Xs.append(data["testX"])
                Xs.append(scaler.transform(data["testX"]))
                Ys.append(np.squeeze(data["testy"]))
                Files.append(data["test_files"])
            except KeyError:
                if plot:
                    y = np.zeros((np.shape(X)[0],))
                else:
                    print "[!] Exiting: Could not load test set from %s" % dataFile
                    exit(0)
        elif dataSet == "training":
            try:
                Xs.append(data["X"])
                try:
                    Ys.append(np.squeeze(data["y"]))
                except KeyError:
                    if fom:
                        print "[!] Exiting: Could not load labels from %s" % dataFile
                        print "[*] FoM calculation is not possible without labels."
                        exit(0)
                    else:
                        Ys.append(np.zeros((np.shape(X)[0],)))
                Files.append(data["images"])
            except KeyError:
                try:
                    Files.append(data["train_files"])
                except KeyError, e:
                    print e
                    try:
                        Files.append(data["files"])
                    except KeyError, e:
                        print e
                        print "[!] Exiting: Could not load training set from %s" % dataFile
                        exit(0)
        else:
            print "[!] Exiting: %s is not a valid choice, choose one of \"training\" or \"test\"" % dataSet
            exit(0)
    
    if poolFile != None:
        Xs = []
        try:
            features = sio.loadmat(poolFile)
            pooledFeaturesTrain = features["pooledFeaturesTrain"]
            X = np.transpose(pooledFeaturesTrain, (0,2,3,1))
            numTrainImages = np.shape(X)[3]
            X = np.reshape(X, (int((pooledFeaturesTrain.size)/float(numTrainImages)), \
                           numTrainImages), order="F")
            scaler = preprocessing.MinMaxScaler()
            # load pooled feature scaler
            #scaler = mlutils.getMinMaxScaler("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_naturalImages_6x6_signPreserveNorm_pooled5.mat")
            #scaler = mlutils.getMinMaxScaler("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_stlTrainSubset_whitened_6x6_signPreserveNorm_pooled5.mat")
            scaler.fit(X.T)  # Don't cheat - fit only on training data
            X = scaler.transform(X.T)
            #X = X.T
            #Xs.append(X)
            if dataSet == "training":
                pass
            elif dataSet == "test":
                pooledFeaturesTest = features["pooledFeaturesTest"]
                X = np.transpose(pooledFeaturesTest, (0,2,3,1))
                numTestImages = np.shape(X)[3]
                X = np.reshape(X, (int((pooledFeaturesTest.size)/float(numTestImages)), \
                               numTestImages), order="F")
                X = scaler.transform(X.T)
            Xs.append(X)
        except IOError:
            print "[!] Exiting: %s Not Found" % (poolFile)
            exit(0)
        finally:
            features = None
            pooledFeaturesTrain = None
            pooledFeaturesTest = None

    preds = []
    indices = []
    for classifierFile in classifierFiles:
        for dataFile in dataFiles:
            if dataFile.rstrip().split("/")[-1].split(".")[0] in classifierFile:
                dataIndex = dataFiles.index(dataFile)
                indices.append(dataIndex)
        X = Xs[dataIndex]
        pred = predict(classifierFile, X)
        preds.append(pred)
    
    if byName:
        files = Files[0]
        preds = [predict_byName(pred, files, outputFile)]
        try:
            Ys = [labels_byName(files, Ys[0])]
        except NameError, e:
            print e

    for pred in preds:
        dataIndex = indices[preds.index(pred)]
        if fom:
            FoM(Ys[dataIndex], pred, threshold)
        
        if plot:
            #y = np.zeros(np.shape(pred))
            try:
                hypothesisDist(Ys[dataIndex], pred, threshold)
            except NameError, e:
                print "[!] NameError : %s", e
    if roc:
        #plot_ROC(Ys, preds, indices)
        test_FDR_procedure(Ys[0], preds[0])
    
if __name__ == "__main__":
    main()
