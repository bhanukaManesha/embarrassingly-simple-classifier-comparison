import pandas as pd
import json
from glob import glob
from tqdm import tqdm

# classifier
# experiment_name
# feature_extractor

# train-accuracy
# train-precision
# train-recall
# train-f1-score

# train-time

# val-accuracy
# val-precision
# val-recall
# val-f1-score

# pred-time

# top-class
# top-class-f1-score
# worst-class
# worst-class-f1-score


def extract_accuracy(path='results/decision-tree/mnasnet1_0-entropy/train_accuracy_report.csv'):
    df = pd.read_csv(path)
    accuracy = df.iloc[0,1]
    precision = df.iloc[2,1]
    recall = df.iloc[2, 2]
    f1_score = df.iloc[2, 3]
    return accuracy, precision, recall, f1_score

def extract_classification_report(path='results/decision-tree/mnasnet1_0-entropy/test-classification_report.csv'):
    df = pd.read_csv(path)
    top_class = df.iloc[0,0]
    top_class_f1_score = df.iloc[0,3]

    worst_class = df.iloc[len(df)-1, 0]
    worst_class_f1_score = df.iloc[len(df)-1,3]

    return top_class, top_class_f1_score, worst_class, worst_class_f1_score


def extract_logs(path='results/decision-tree/mnasnet1_0-entropy/log.json'):
    js = json.load(open(path))
    classifier = js['model_type']
    experiment_name = js['exp_name']
    feature_extractor = js['feature_extractor']
    train_time = js['train_time']
    pred_time = js['test_pred_time']
    return classifier, experiment_name, feature_extractor, train_time, pred_time

def extract_nn_logs(path='results/nn/mnasnet1_0-32-adamax-1e-05-wd/val-best/val-log.json'):
    js = json.load(open(path))
    classifier = js['params']['model_type']
    experiment_name = js['params']['exp_name']
    feature_extractor = js['params']['feature_extractor']
    train_time = js['metrics']['train_time']
    pred_time = js['metrics']['test_pred_time']
    return classifier, experiment_name, feature_extractor, train_time, pred_time

def extract_details(df, path):
    if '/nn/' not in path:
        train_accuracy, train_precision, train_recall, train_f1_score = extract_accuracy(
            path + 'train_accuracy_report.csv')
        test_accuracy, test_precision, test_recall, test_f1_score = extract_accuracy(path + 'test-accuracy_report.csv')
        top_class, top_class_f1_score, worst_class, worst_class_f1_score = extract_classification_report(
            path + 'test-classification_report.csv')
        classifier, experiment_name, feature_extractor, train_time, pred_time = extract_logs(path+'log.json')
    else:
        train_accuracy, train_precision, train_recall, train_f1_score = extract_accuracy(
            path + 'val-best/train_accuracy_report.csv')
        test_accuracy, test_precision, test_recall, test_f1_score = extract_accuracy(path + 'val-best/test-accuracy_report.csv')
        top_class, top_class_f1_score, worst_class, worst_class_f1_score = extract_classification_report(
            path + 'val-best/test-classification_report.csv')

        classifier, experiment_name, feature_extractor, train_time, pred_time = extract_nn_logs(path + 'val-best/val-log.json')

    df = df.append(pd.Series([classifier, experiment_name, feature_extractor, train_accuracy, train_precision, train_recall, train_f1_score, train_time, test_accuracy, test_precision, test_recall, test_f1_score, pred_time, top_class, top_class_f1_score, worst_class, worst_class_f1_score], index=df.columns), ignore_index=True)

    return df


def extract_results():

    main_df = pd.DataFrame(columns = ['Classifier' , 'Experiment Name', 'Feature Extractor' , 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score', 'Test Time', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score', 'Test Time', 'Top Class', 'Top Class F1 Score', 'Worst Class', 'Worst Class F1 Score'])

    model_types = glob("results/*/")
    for model_type in tqdm(model_types):
        experiments = glob(model_type+'*/')
        for experiment in experiments:
            main_df = extract_details(main_df, experiment)

    print(main_df)

    main_df.to_csv('results/results.csv')










if __name__ == '__main__':
    # extract_accuracy()
    # extract_classification_report()
    # extract_logs()
    # extract_nn_logs()
    extract_results()