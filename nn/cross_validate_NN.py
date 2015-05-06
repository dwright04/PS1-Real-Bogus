import optparse, pickle
import numpy as np
import scipy.io as sio
from train_nn import train_nn
from analyse_NN import measure_FoM
from NeuralNet import NeuralNet
from sklearn.cross_validation import KFold

def main():

    parser = optparse.OptionParser("[!] usage: python cross_validate_RF.py -F <data file>")

    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")

    (options, args) = parser.parse_args()
    dataFile = options.dataFile

    if dataFile == None:
        print parser.usage
        exit(0)

    data = sio.loadmat(dataFile)

    X = data["X"]
    m,n = np.shape(X)
    y = np.squeeze(data["y"])

    arch_grid = [200]
    lambda_grid = [1.0,5.0]


    kf = KFold(m, n_folds=5, indices=False)
    fold = 1
    for arch in arch_grid:
        for LAMBDA in lambda_grid:
            fold=1
            FoMs = []
            for train, test in kf:
                print "[*]", fold, arch, LAMBDA
                file = "cv/NeuralNet_"+dataFile.split("/")[-1].split(".")[0]+\
                       "_arch"+str(arch)+"_lambda%.6f"% (LAMBDA) +"_fold"+str(fold)+".mat"
                try:
                    #nn = pickle.load(open(file,"rb"))
                    nn = NeuralNet(X[train].T, y[train], saveFile=file)
                    print nn._architecture
                    print nn._trainedParams
                except IOError:
                    train_x, train_y = X[train], y[train]
                    nn = train_NN(train_x.T, train_y, arch, LAMBDA)
                    outputFile = open(file, "wb")
                    pickle.dump(nn, outputFile)
                FoM, threshold = measure_FoM(X[test].T, y[test], nn, False)
                fold+=1
                FoMs.append(FoM)
            print "[+] mean FoM: %.3lf" % (np.mean(np.array(FoMs)))
            print

if __name__ == "__main__":
    main()
