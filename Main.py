import numpy as np
import pandas as pd
from knearestneighbor import KNearestNeighbor

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# help defining f - score credit: https://www.kaggle.com/code/meirnizri/covid-19-risk-prediction
def get_f_score(prediction, labels):
    true_positive = np.sum((prediction + labels) == 2)
    false_positive = np.sum((labels - prediction) == -1)
    false_negative = np.sum((prediction - labels) == -1)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    score = 2 * (precision * recall) / (precision + recall)
    return score


# this method makes sure that all the boolean features have valid values ie. 1 or 2
# it rids of all the invalid values ie. 97 or 99
def clean_data(csv_file):
    csv_file = csv_file.loc[(csv_file.CLASIFFICATION_FINAL < 4)]
    csv_file = csv_file.loc[(csv_file.SEX == 1) | (csv_file.SEX == 2)]
    csv_file = csv_file.loc[(csv_file.USMER == 1) | (csv_file.USMER == 2)]
    csv_file = csv_file.loc[(csv_file.PATIENT_TYPE == 1) | (csv_file.PATIENT_TYPE == 2)]
    csv_file = csv_file.loc[(csv_file.PNEUMONIA == 1) | (csv_file.PNEUMONIA == 2)]
    csv_file = csv_file.loc[(csv_file.DIABETES == 1) | (csv_file.DIABETES == 2)]
    csv_file = csv_file.loc[(csv_file.COPD == 1) | (csv_file.COPD == 2)]
    csv_file = csv_file.loc[(csv_file.ASTHMA == 1) | (csv_file.ASTHMA == 2)]
    csv_file = csv_file.loc[(csv_file.INMSUPR == 1) | (csv_file.INMSUPR == 2)]
    csv_file = csv_file.loc[(csv_file.HIPERTENSION == 1) | (csv_file.HIPERTENSION == 2)]
    csv_file = csv_file.loc[(csv_file.OTHER_DISEASE == 1) | (csv_file.OTHER_DISEASE == 2)]
    csv_file = csv_file.loc[(csv_file.CARDIOVASCULAR == 1) | (csv_file.CARDIOVASCULAR == 2)]
    csv_file = csv_file.loc[(csv_file.OBESITY == 1) | (csv_file.OBESITY == 2)]
    csv_file = csv_file.loc[(csv_file.RENAL_CHRONIC == 1) | (csv_file.RENAL_CHRONIC == 2)]
    csv_file = csv_file.loc[(csv_file.TOBACCO == 1) | (csv_file.TOBACCO == 2)]
    return csv_file


# this def converts all of the boolean features from 1 and 2 --> 0 and 1
# where 0 is no and 1 is yes
def convert_to_binary_data(cleaned_data):
    cleaned_data.DATE_DIED = cleaned_data.DATE_DIED.apply(lambda x: 0 if x == "9999-99-99" else 1)
    cleaned_data.SEX = cleaned_data.SEX.apply(lambda x: x if x == 1 else 0)
    cleaned_data.USMER = cleaned_data.USMER.apply(lambda x: x if x == 1 else 0)
    cleaned_data.PATIENT_TYPE = cleaned_data.PATIENT_TYPE.apply(lambda x: 0 if x == 1 else 1)
    cleaned_data.PNEUMONIA = cleaned_data.PNEUMONIA.apply(lambda x: x if x == 1 else 0)
    cleaned_data.DIABETES = cleaned_data.DIABETES.apply(lambda x: x if x == 1 else 0)
    cleaned_data.COPD = cleaned_data.COPD.apply(lambda x: x if x == 1 else 0)
    cleaned_data.ASTHMA = cleaned_data.ASTHMA.apply(lambda x: x if x == 1 else 0)
    cleaned_data.INMSUPR = cleaned_data.INMSUPR.apply(lambda x: x if x == 1 else 0)
    cleaned_data.HIPERTENSION = cleaned_data.HIPERTENSION.apply(lambda x: x if x == 1 else 0)
    cleaned_data.OTHER_DISEASE = cleaned_data.OTHER_DISEASE.apply(lambda x: x if x == 1 else -0)
    cleaned_data.CARDIOVASCULAR = cleaned_data.CARDIOVASCULAR.apply(lambda x: x if x == 1 else 0)
    cleaned_data.OBESITY = cleaned_data.OBESITY.apply(lambda x: x if x == 1 else 0)
    cleaned_data.RENAL_CHRONIC = cleaned_data.RENAL_CHRONIC.apply(lambda x: x if x == 1 else 0)
    cleaned_data.TOBACCO = cleaned_data.TOBACCO.apply(lambda x: x if x == 1 else 0)
    cleaned_data.PREGNANT = cleaned_data.PREGNANT.apply(lambda x: x if x == 1 else 0)
    cleaned_data.INTUBED = cleaned_data.INTUBED.apply(lambda x: x if x == 1 else 0)
    cleaned_data.ICU = cleaned_data.ICU.apply(lambda x: x if x == 1 else 0)
    return cleaned_data


def main():
    # reads in the file while keeping memory usage low
    covid_csv = pd.read_csv('Covid Data.csv', low_memory=False)
    cleaned_data = clean_data(covid_csv)
    usable_cleaned_data = convert_to_binary_data(cleaned_data)

    # create a new column to represent whether someone is high risk
    usable_cleaned_data['HIGH_RISK'] = usable_cleaned_data['DATE_DIED'] + usable_cleaned_data['INTUBED'] + \
                                       usable_cleaned_data['ICU']
    # 1 is used if someone has died, was in the ICU, and was on a ventilator o.w. 0 is used
    usable_cleaned_data.HIGH_RISK = usable_cleaned_data.HIGH_RISK.apply(lambda x: 1 if x > 0 else 0)

    # split the data for training
    train, test = train_test_split(usable_cleaned_data, test_size=0.1, shuffle=True)
    test_y = test.HIGH_RISK.to_numpy()
    test_x = test.drop(columns=['HIGH_RISK']).to_numpy()

    # under sampling the data set, credit for idea/implementation:
    # https://www.kaggle.com/code/meirnizri/covid-19-risk-prediction
    at_risk = train[train.HIGH_RISK == 1][:2000]
    non_risk = train[train.HIGH_RISK == 0][:2000]
    part_train = pd.concat([non_risk, at_risk])
    train_y = part_train.HIGH_RISK.to_numpy()
    train_x = part_train.drop(columns=['HIGH_RISK']).to_numpy()

    # initialize KNN parameters
    list_of_ks = [1, 5, 10, 25]
    knn_f_scores = {}
    knn_best_result = 0

    for k in list_of_ks:
        knn_classifier = KNearestNeighbor(k, train_x, train_y)
        prediction = knn_classifier.predict(test_x)

        knn_f_scores[k] = get_f_score(prediction, test_y)
        if knn_f_scores[k] > knn_best_result:
            knn_best_result = knn_f_scores[k]
            knn_best_model = knn_classifier

        print('K = ' + str(k) + ': Accuracy = ' + str(knn_f_scores[k]))

    print('*****PRINTING BEST RESULT WITH K VALUE USED*****')
    print('K = ' + str(knn_best_model.get_k_value()) + ': Accuracy = ' + str(knn_best_result))

    # multi class perceptron
    mcp_classifier = MLPClassifier(learning_rate='adaptive', max_iter=500)
    mcp_classifier.fit(train_x, train_y)
    prediction = mcp_classifier.predict(test_x)
    mcp_score = get_f_score(prediction, test_y)
    print(str(mcp_score))

    #  SVM classifiers
    list_of_kernels = ['linear', 'poly', 'rbf']
    svm_scores = {}

    for kernel_type in list_of_kernels:
        svm_classifier = SVC(kernel=kernel_type, class_weight='balanced')
        svm_classifier.fit(train_x, train_y)
        prediction = svm_classifier.predict(test_x)
        svm_scores[kernel_type] = get_f_score(prediction, test_y)
        print(kernel_type + ': Accuracy = ' + str(svm_scores[kernel_type]))


if __name__ == '__main__':
    main()
