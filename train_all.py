import decisiontree
import randomforest
import knn
import naivebayes
import svm


if __name__ == '__main__':
    decisiontree.run_loop()
    randomforest.run_loop()
    knn.run_loop()
    naivebayes.run_loop()
    svm.run_loop()