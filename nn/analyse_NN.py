import optparse
import pickle, sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from NeuralNet import NeuralNet
#from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
import matplotlib as mpl

def measure_FoM(X, y, classifier, plot=True):
    pred = classifier.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, pred)

    FoM = 1-tpr[np.where(fpr<=0.01)[0][-1]]
    print "[+] FoM: %.4f" % (FoM)
    threshold = thresholds[np.where(fpr<=0.01)[0][-1]]
    print "[+] threshold: %.4f" % (threshold)
    print

    if plot:
        font = {"size": 18}
        plt.rc("font", **font)
        plt.rc("legend", fontsize=14)
    
        plt.xlabel("Missed Detection Rate (MDR)")
        plt.ylabel("False Positive Rate (FPR)")
        plt.yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
        plt.ylim((0,1.05))

        plt.plot(1-tpr, fpr, "k-", lw=5)
        plt.plot(1-tpr, fpr, color="#FF0066", lw=4)

        plt.plot([x for x in np.arange(0,FoM+1e-3,1e-3)], \
                  0.01*np.ones(np.shape(np.array([x for x in np.arange(0,FoM+1e-3,1e-3)]))), \
                 "k--", lw=3)

        plt.plot(FoM*np.ones((11,)), [x for x in np.arange(0,0.01+1e-3, 1e-3)], "k--", lw=3)

        plt.xticks([0, 0.05, 0.10, 0.25, FoM], rotation=70)

        locs, labels = plt.xticks()
        plt.xticks(locs, map(lambda x: "%.3f" % x, locs))
        plt.show()
    return FoM, threshold

def main():

    parser = optparse.OptionParser("[!] usage: python analyse_RF.py -F <data file>"+\
                                   " -c <classifier file> -s <data set>")

    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")
    parser.add_option("-c", dest="classifierFile", type="string", \
                      help="specify classifier to use")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse ([training] or [test] set)")

    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    classifierFile = options.classifierFile
    dataSet = options.dataSet

    print

    if dataFile == None or classifierFile == None or dataSet == None:
        print parser.usage
        exit(0)

    if dataSet != "training" and dataSet != "test":
        print "[!] Exiting: data set must be 1 of 'training' or 'test'"
        exit(0)

    try:
        data = sio.loadmat(dataFile)
    except IOError:
        print "[!] Exiting: %s Not Found" % (dataFile)
        exit(0)

    #scaler = preprocessing.StandardScaler().fit(data["X"])
    #d = sio.loadmat("../contextual_classification_dataset_20150706_shuffled.mat")
    d = sio.loadmat("../contextual_classification_dataset_pca2_20150707_shuffled.mat")
    scaler = preprocessing.StandardScaler().fit(d["X"])
    if dataSet == "training":
        print np.shape(data["X"])
        print data["y"]
        print data["ids"]
        X = scaler.transform(data["X"]).T
        y = np.squeeze(data["y"])
    elif dataSet == "test":
        X = scaler.transform(data["testX"]).T
        y = np.squeeze(data["testy"])
        print

    try:
        classifier = pickle.load(open(classifierFile, "rb"))
    except IOError:
        print "[!] Exiting: %s Not Found" % (classifierFile)
        exit(0)
    #measure_FoM(X, y, classifier)
    print np.array(classifier.predict_proba(X) <= .5)[:,0]
    print "f1 score : %.3f" % f1_score(np.squeeze(y), np.array(classifier.predict_proba(X) <= .5)[:,0])
    print "accuracy score : %.3f" % accuracy_score(np.squeeze(y), np.array(classifier.predict_proba(X) <= .5)[:,0])



    h = .02
    x_min, x_max = scaler.transform(data["X"]).T[:, 0].min() - 2, scaler.transform(data["X"]).T[:, 0].max() + 4
    y_min, y_max = scaler.transform(data["X"]).T[:, 1].min() - 2, scaler.transform(data["X"]).T[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    print np.shape(X)
    print np.shape(np.c_[xx.ravel(), yy.ravel()].T)
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()].T)[:,1]
    Z = Z.reshape(xx.shape)

    font = {"size"   : 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    pred = np.array(classifier.predict_proba(X) <= .5, dtype="int64")[:,0]
    print "f1 score : %.3f" % f1_score(np.squeeze(y), np.array(classifier.predict_proba(X) <= .5)[:,0])

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    
    ax2.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10),
                 cmap=plt.cm.cool_r, alpha=.7)
        
    ax2.contour(xx, yy, Z, [0.5], colors='w',lw="10")
                 
    ax2.scatter(X.T[y==1,0],\
                X.T[y==1,1], color="#008BF8",s=75, label="SNe",edgecolor="k")
                 
    ax2.scatter(X.T[y!=1,0],\
                X.T[y!=1,1], color="#DC0073",s=75,label="not SNe",edgecolor="k")
        
                 
    ax2.set_xlabel("first principal component")
    ax2.set_ylabel("second principal component")

    ax2.set_xlim(x_min,x_max)
    ax2.set_ylim(y_min,y_max)
    print accuracy_score(y, pred)
    plt.legend(loc="upper left", numpoints=1)
    plt.show()

    #print "f1 score : %.3f" % f1_score(np.squeeze(y), np.array(classifier.predict(X))[:,0])

#    pred = np.array(classifier.predict_proba(X) <= .5, dtype="int64")[:,0]
#    print pred
#    output = open("nn_preds.txt","w")
#    print classifier.predict_proba(X)[:,1]
#    i = 0
#    for p in classifier.predict_proba(X)[:,1]:
#        output.write("%.3f,%.3f\n" % (p,X[7,i]))
#        i+=1
#    output.close()
    """
    X = data["X"]
    print np.shape(X)
    h = .02

    x_min, x_max = X[:, 7].min()-.1, X[:, 7].max()+.2
    y_min, y_max = X[:, 4].min()-.1, X[:, 4].max()+.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    print np.shape(xx)
    print np.shape(yy.ravel())
    step = 14840
    p = np.ones((step,1))
    aa = np.arange(X[:,0].min(),X[:,0].max(), (X[:,0].max()-X[:,0].min()) / float(step))
    bb = np.arange(X[:,1].min(),X[:,1].max(), (X[:,1].max()-X[:,1].min()) / float(step))
    cc = np.arange(X[:,2].min(),X[:,2].max(), (X[:,2].max()-X[:,2].min()) / float(step))
    dd = np.arange(X[:,3].min(),X[:,3].max(), (X[:,3].max()-X[:,3].min()) / float(step))
    ee = np.arange(X[:,5].min(),X[:,5].max(), (X[:,5].max()-X[:,5].min()) / float(step))
    ff = np.arange(X[:,6].min(),X[:,6].max(), (X[:,6].max()-X[:,6].min()) / float(step))
    gg = np.arange(X[:,8].min(),X[:,8].max(), (X[:,8].max()-X[:,8].min()) / float(step))
    print np.shape(dd)

    #print np.shape(np.c_[p.ravel(), p.ravel(), p.ravel(), p.ravel(), yy.ravel(), p.ravel(), p.ravel(), xx.ravel(), p.ravel()])
    Z = classifier.predict_proba(np.c_[aa.ravel(), bb.ravel(), cc.ravel(), dd.ravel(), yy.ravel(), ee.ravel(), ff.ravel(), xx.ravel(), gg.ravel()].T)[:,1]
    Z = np.reshape(Z,(56, 265))

    print type(Z)
    print np.shape(Z)

    print np.shape(xx[0][:])
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                 cmap=plt.cm.Purples)

    """
#    X = data["X"]
#    print np.shape(X)
#    h = 1
#
#    x_min, x_max = X[:, 7].min()-.1, X[:, 7].max()+.2
#    y_min, y_max = X[:, 4].min()-.1, X[:, 4].max()+.2
#
#    a_min, a_max = X[:, 0].min()-.1, X[:, 0].max()+.2
#    b_min, b_max = X[:, 1].min()-.1, X[:, 1].max()+.2
#    c_min, c_max = X[:, 2].min()-.1, X[:, 2].max()+.2
#    d_min, d_max = X[:, 3].min()-.1, X[:, 3].max()+.2
#    e_min, e_max = X[:, 5].min()-.1, X[:, 5].max()+.2
#    f_min, f_max = X[:, 6].min()-.1, X[:, 6].max()+.2
#    g_min, g_max = X[:, 8].min()-.1, X[:, 8].max()+.2
#    aa, bb, cc, dd, yy, ee, ff, xx, gg = \
#                         np.meshgrid(np.arange(a_min, a_max, h),
#                         np.arange(b_min, b_max, h),
#                         np.arange(c_min, c_max, h),
#                         np.arange(d_min, d_max, h),
#                         np.arange(y_min, y_max, h),
#                         np.arange(e_min, e_max, h),
#                         np.arange(f_min, f_max, h),
#                         np.arange(x_min, x_max, h),
#                         np.arange(g_min, g_max, h))
#    print np.shape(xx)
#    print np.shape(yy.ravel())
#    print np.shape(dd)
#    Z = classifier.predict_proba(np.c_[aa.ravel(), bb.ravel(), cc.ravel(), dd.ravel(), yy.ravel(), ee.ravel(), ff.ravel(), xx.ravel(), gg.ravel()].T)[:,1]
#    print np.shape(Z)
#    Z = np.reshape(Z,(6, 10, 5, 8, 2, 2, 2, 6, 13))
##    Z = np.sum(Z,axis=0)
##    Z = np.sum(Z,axis=0)
##    Z = np.sum(Z,axis=0)
##    Z = np.sum(Z,axis=0)
##    Z = np.sum(Z,axis=1)
##    Z = np.sum(Z,axis=1)
##    Z = np.sum(Z,axis=2)
##    Z = np.mean(Z,axis=0)
##    Z = np.mean(Z,axis=0)
##    Z = np.mean(Z,axis=0)
##    Z = np.mean(Z,axis=0)
##    Z = np.mean(Z,axis=1)
##    Z = np.mean(Z,axis=1)
##    Z = np.mean(Z,axis=2)
#    print np.shape(Z)
#    h = 1
#    
#    x_min, x_max = X[:, 7].min()-.1, X[:, 7].max()+.2
#    y_min, y_max = X[:, 4].min()-.1, X[:, 4].max()+.2
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#    Z = Z[(0,0,0,0)]
#    print np.shape(Z)
#    Z = Z
#    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
#                 cmap=plt.cm.Purples)

#    X = data["testX"]
#    #X = data["X"]
#    #y = np.squeeze(data["y"])
#    #print pred
#    #print X[np.where(np.squeeze(y) != pred)[0],7]
#
#    print np.where(y != pred)[0]
#
#    print y[np.where(y != pred)]
#    
#    pos= pred[y==1]
#    neg = pred[y==0]
#    
#    
#    tps = np.where(pos==1)[0]
#    fns = np.where(pos!=1)[0]
#
#    tns = np.where(neg==0)[0]
#    fps = np.where(neg!=0)[0]
#    #print len(fps)
#
#    font = {"size"   : 26}
#    plt.rc("font", **font)
#    plt.rc("legend", fontsize=22)
#    #plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    
#    plt.scatter(X[y==1,:][tps,7],\
#                X[y==1,:][tps,4],\
#                color="#3A86FF", edgecolor="none",alpha=.5, s=100, label="true positive")
#        
#    plt.scatter(X[y==0,:][tns,7],\
#                X[y==0,:][tns,4],\
#                color="#FF006E", edgecolor="none",alpha=.5, s=100, label="true negative")
#    plt.scatter(X[y==0,:][fps,7],\
#                X[y==0,:][fps,4],color="#FFBE0B", s=100, edgecolor="none",label="false positive")
#
#    print X[y==1,:][fns,7]
#    plt.scatter(X[y==1,:][fns,7],\
#                X[y==1,:][fns,4],color="#89FC00", s=100, edgecolor="none",label="false negative")
#                
#    plt.plot([0.5,0.5],[-1,10],"k--")
#                
#                
#    plt.tick_params(axis='both', which='major', labelsize=18)
#    mpl.rcParams['legend.scatterpoints'] = 1
#    plt.legend(loc="lower center", numpoints=1)
#    #plt.axes().set_aspect('equal', 'datalim')
#                
#    # Hide the right and top spines
#    plt.axes().spines['right'].set_visible(False)
#    plt.axes().spines['top'].set_visible(False)
#    
#    # Only show ticks on the left and bottom spines
#    plt.axes().yaxis.set_ticks_position('left')
#    plt.axes().xaxis.set_ticks_position('bottom')
#    plt.tick_params(axis='both', which='major', labelsize=22)
#    plt.xlabel("offset [arcsec]")
#    plt.ylabel("photo. redshift")
#    plt.xticks([0,0.5,1,2,3,4,5])
#    plt.xlim(xmin=-.1)
#    plt.ylim(ymin=-.1,ymax=0.61)
#    plt.legend(loc="upper right")
#
#    plt.show()

if __name__ == "__main__":
    main()
