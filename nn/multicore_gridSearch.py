#!/usr/bin/python
import sys, multiprocessing
import scipy.io as sio
import multiprocessingUtils
#from DeepANNMultiLayer import *
from NeuralNet import *
#from DropoutNeuralNet import *
from sklearn.cross_validation import KFold

class trainNetwork(multiprocessingUtils.Task):
    def __init__(self, net, outputFile):
        # net is a NeuralNet object to be trained.
        self.net = net
        self.outputFile = outputFile

    def __call__(self):
        self.net.train()
        self.net.saveNetwork(self.outputFile)
        return 0
    
    def __str__(self):
        return "### Training %s with arch = %d, LAMBDA = %f ###" % (self.net.search(), self.net._architecture[1], self.net._LAMBDA)

def main(argv = None):
    
    if argv is None:
        argv = sys.argv
    
    if len(argv) != 2:
        sys.exit("[-] Usage: multicore_gridSearch.py <.mat file>")

    dataFile = argv[1]
    data = sio.loadmat(dataFile)
    train_x = data["X"]
    #train_x = np.concatenate((data["X"], data["validX"]))
    #mu = np.mean(train_x, axis=0)
    #sigma = np.std(train_x, axis=0)
    #print np.shape(mu)
    #train_x = train_x - mu
    #train_x = train_x / sigma
    #train_y = np.squeeze(np.concatenate((data["y"], data["validy"])))
    train_y = np.squeeze(data["y"])
    #print np.shape(train_y)
    #archGrid = [200]
    hiddenSize = 200
    #lambdaGrid = [1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e+1, 3e+1, 1e+2]
    #lambdaGrid = [100]
    lambdaGrid = [1]
    taskList = []

    kf = KFold(len(train_y), n_folds=5, indices=False)
    fold = 1
    for train, test in kf:
        X, y = train_x[train], np.squeeze(train_y[train])
        print np.shape(X.T)
        print np.shape(y[np.newaxis].T)
        #for hiddenSize in archGrid:
        for LAMBDA in lambdaGrid:
                arch = {1:int(hiddenSize)}

                #taskList.append(trainNetwork(NeuralNet(X.T, y, architecture=arch, LAMBDA=LAMBDA, maxiter=10000), \
                #                     "cv/trainedNet_NerualNet_%s_arch%d_lambda%f_fold%d.mat" % \
                #                     (argv[1].split("/")[-1].split(".")[0], arch[1], LAMBDA, fold)))

                outputFile = "cv/trainedNet_NerualNet_%s_arch%d_lambda%f_fold%d.mat" % \
                              (argv[1].split("/")[-1].split(".")[0], arch[1], LAMBDA, fold)
                #outputFile = "cv/trainedNet_DropoutNerualNet_%s_arch%d_fold%d.mat" % \
                #              (argv[1].split("/")[-1].split(".")[0], arch[1], fold)
                nn = NeuralNet(X.T, y[np.newaxis], architecture=arch, LAMBDA=LAMBDA, maxiter=10000)
                print "### Training %s with arch = %d, LAMBDA = %f ###" % \
                      (nn.search(), nn._architecture[1], nn._LAMBDA)
                print nn._architecture
                #nn = DropoutNeuralNet(X.T, y[np.newaxis], architecture=arch, maxiter=10000)
                #print "### Training %s with arch = %d ###" % \
                #      (nn.search(), nn._architecture[1])
                #print nn._architecture

                nn.train()
                nn.saveNetwork(outputFile)

        fold +=1

    """
    cpu_count = multiprocessing.cpu_count() - 1
    print "%d available CPUs.\n" % (cpu_count)

    multiprocessingUtils.multiprocessTaskList(taskList, cpu_count)
    """
    """
    outputFile = "trainedNet_NerualNet_%s_arch%d_lambda%f.mat" % \
                 (dataFile.split("/")[-1].split(".")[0], archGrid[0], lambdaGrid[0])
    nn = NeuralNet(train_x.T, train_y[np.newaxis], architecture={1:archGrid[0]}, \
         LAMBDA=lambdaGrid[0], maxiter=10000)
    print "### Training %s with arch = %d, LAMBDA = %f ###" % \
          (nn.search(), nn._architecture[1], nn._LAMBDA)
    print nn._architecture
                
    nn.train()
    nn.saveNetwork(outputFile)
    """

if __name__ == "__main__":
    main()
