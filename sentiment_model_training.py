# Sentiment CLassifier using Multi-nominal Naive Bayes classifier

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import re
import string
import configparser

lemmatizer = WordNetLemmatizer()
port = PorterStemmer()


#############Lemmatize/Stemminize######################
def lemmaStemma(text):
    return lemmatizer.lemmatize(text)


#######################Model Evaluation##############
def evaluation(predictions, predictions_prob, test_labels):
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error(MAE):', round(np.mean(errors), 2))

    """mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%')"""
    print('Accuracy:', round(accuracy_score(test_labels, predictions) * 100, 2), '%')

    '''confusion = precision_recall_fscore_support(test_labels, predictions, average='binary')
    print('Precision:', confusion[0])
    print('Recall:', confusion[1])
    print('F1:', confusion[2])'''

    print("****Confusion Matrix****")
    print(confusion_matrix(test_labels, predictions))


def main():

    # Read parameters from ini file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Parameter defining
    sentiment_training_data_path = config['TRAINING_PATH']['sentiment_training_data_path']
    sentiment_vector = config['MODEL']['sentiment_vector']
    sentiment_classifier = config['MODEL']['sentiment_classifier']
    classification_test_size = config['PARAMETERS']['classification_test_size']


    df = pd.read_csv(sentiment_training_data_path, header=0, encoding='unicode_escape').dropna()
    df['comment'] = df['comment'].str.lower()
    df['comment'] = df['comment'].map(lambda x: re.sub(r'\W+', ' ', x))
    df['comment'] = df['comment'].map(lambda x: re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', x))

    df['comment'] = df['comment'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    features = df["comment"]
    labels = df["label"]

    vectorizer = CountVectorizer(max_features=1500, min_df=2, max_df=0.7, stop_words="english")
    Answer_vect_fit = vectorizer.fit(features)
    Answer_vect = Answer_vect_fit.transform(features).toarray()

    # Save Vectorizer
    print("Saving santiment vectorizer....")
    pkl_filename = sentiment_vector
    with open(pkl_filename, 'wb') as file:
        pickle.dump(Answer_vect_fit, file)


    print("Data Splitting")
    ##Split train/test
    train_features, test_features, train_labels, test_labels, train_org, test_org = train_test_split(Answer_vect,
                                                                                                     labels,
                                                                                                     df["comment"],
                                                                                                     stratify=labels,
                                                                                                     test_size=float(classification_test_size),
                                                                                                     random_state=42)

    # MNB CLASSIFICATION
    mnb = MultinomialNB()
    mnbModel = mnb.fit(train_features, train_labels)

    mnb_prediction = mnbModel.predict(test_features)
    mnb_prob = mnbModel.predict_proba(test_features)

    mnb_results = np.array(list(zip(
                                    test_org,
                                    test_labels,
                                    mnb_prediction,
                                    mnb_prob[:, 0],
                                    mnb_prob[:, 1],
                                    mnb_prob[:, 2])
                                )
                        )

    mnb_results = pd.DataFrame(mnb_results, columns=['test_org',
                                                     'actual',
                                                     'prediction',
                                                     'negative',
                                                     'neutral',
                                                     'positive']
                               )

    print("****Model Evaluation****")
    evaluation(mnb_prediction, mnb_prob, test_labels)

     # Save Model
    print("Saving the sentiment model....")
    pkl_filename = sentiment_classifier
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mnbModel, file)


if __name__ == '__main__':
    main()