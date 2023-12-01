import argparse
import numpy as np
import configparser
import logging
#import log_reg

#Logging
logging.basicConfig(filename='AutoClassifierLog.log',level=logging.INFO,format='[%(asctime)s] %(levelname)s : %(message)s')
logging.info("--------------------------------------------------------------------------------------------------------")

#Config parser
config = configparser.ConfigParser()
config.read('autoclassifier_config.properties')
ds_location = config.get('Section','ds_location')
dataset_name = ds_location.split("\\")[-1]

def main():
    # Create an argument parser.
    parser = argparse.ArgumentParser()

    # Add autoclassify argument.
    parser.add_argument('--autoclassify', action='store_true', help='Classifies the dataset returns the Algorithm that gives best accuracy.')

    # Add logistic regression argument.
    parser.add_argument('--logistic_regression', action='store_true', help='Uses logistic regression to classify the dataset and returns accuracy score.')

    # Add SVM(Linear) classifier argument.
    parser.add_argument('--svm_linear', action='store_true', help='Uses SVM(kernel="linear") to classify the dataset and returns accuracy score.')

    # Add SVM(Non-Linear) classifier argument.
    parser.add_argument('--svm_non_linear', action='store_true', help='Uses SVM(kernel="rbf") to classify the dataset and returns accuracy score.')

    # Add KNN classifier argument.
    parser.add_argument('--knn', action='store_true', help='Uses KNN to classify the dataset and returns accuracy score.')

    # Add Naive Bayes classifier argument.
    parser.add_argument('--naive_bayes', action='store_true', help="Uses Naive Bayes to classify the dataset and returns accuracy score.")

    # Add Decision Tree classifier argument.
    parser.add_argument('--decision_tree', action='store_true', help='Uses Decision Tree to classify the dataset and returns accuracy score.')

    # Add Random Forest classifier argument.
    parser.add_argument('--random_forest', action='store_true', help='Uses Random Forest to classify the dataset and returns the accuracy score.')

    # Parse the arguments.
    args = parser.parse_args()

    # If the autoclassify argument is specified, classify the dataset and give the best algorithm.
    if args.autoclassify==True:
        print(f"Executing Autoclassifier on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Autoclassifier on the Dataset: {dataset_name}\n")
        import log_reg
        print("\n-------------------------------------\n")
        import svm_linear
        print("\n-------------------------------------\n")
        import kernel_svm
        print("\n-------------------------------------\n")
        import knn
        print("\n-------------------------------------\n")
        import naive_bayes
        print("\n-------------------------------------\n")
        import dt
        print("\n-------------------------------------\n")
        import rf
        print("\n-------------------------------------\n")
    
        d={}

        lr_ac=log_reg.lr_ac
        d['Logistic Regression'] = lr_ac

        svm_l_ac=svm_linear.svm_l_ac
        d['SVM Linear'] = svm_l_ac

        svc_nl_ac=kernel_svm.svc_nl_ac
        d['SVM Non-linear'] = svc_nl_ac

        knn_ac=knn.knn_ac
        d['KNN'] = knn_ac

        naive_bayes_ac=naive_bayes.naive_bayes_ac
        d['Naive Bayes'] = naive_bayes_ac

        dt_ac=dt.dt_ac
        d['Decision Tree'] = dt_ac

        rf_ac=rf.rf_ac
        d['Random Forest'] = rf_ac

        dict(sorted(d.items(),key=lambda item: item[1], reverse=True))
        best_algo=list(d.keys())[0]
        best_accuracy=list(d.values())[0]
        print(f"Among the above classification algorithms, {best_algo} gives the best accuracy of {best_accuracy}") 
        logging.info(f"Among the above classification algorithms, {best_algo} gives the best accuracy of {best_accuracy}")      
        logging.info("\n\n") 

        
    if args.logistic_regression==True:
        print(f"Executing Logistic Regression on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Logistic Regression on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import log_reg

    if args.svm_linear:
        print(f"Executing Support Vector Machine(Linear) on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Support Vector Machine(Linear) on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import svm_linear
    
    if args.svm_non_linear:
        print(f"Executing Support Vector Machine(Kernel) on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Support Vector Machine(Kernel) on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import kernel_svm
    
    if args.knn:
        print(f"Executing K-Nearest Neighbor on the Dataset: {dataset_name}\n")
        logging.info(f"Executing K-Nearest Neighbor on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import knn
    
    if args.naive_bayes:
        print(f"Executing Naive Bayes on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Naive Bayes on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import naive_bayes
    
    if args.decision_tree:
        print(f"Executing Decision Tree on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Decision Tree on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import dt
    
    if args.random_forest:
        print(f"Executing Random Forest on the Dataset: {dataset_name}\n")
        logging.info(f"Executing Random Forest on the Dataset: {dataset_name}\n")
        logging.info("\n\n")
        import rf
        
        
if __name__ == '__main__':
    main()