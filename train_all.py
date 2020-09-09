import decisiontree
import randomforest
import knn
import naivebayes
import svm
import nn_train


if __name__ == '__main__':
    decisiontree.run_loop()
    randomforest.run_loop()
    knn.run_loop()
    naivebayes.run_loop()
    svm.run_loop()
    nn_train.run_loop()
