# This the scoring file which use to classify overall sentiment and splitted sentiments and return the results as a json using azure-sdk.

import pandas as pd
import numpy as np
from azureml.core.model import Model
import pickle
import re
import json

# Split long sentences into multiple
def split_sentences(dataset):
    comments_remarks = []
    split_sentence = []
    sentenceEnders = re.compile('but|[.!?]|however|that\s|Wherever|and')

    for _, row in dataset.iterrows():
        comments_remarks.append(row.comments_remarks)
        #split_sentence.append(nltk.tokenize.sent_tokenize(str(row.comments_remarks)))
        split_sentence.append(map(str.strip, sentenceEnders.split(row.comments_remarks.lower())))

    split_list = pd.DataFrame({
            "comments_remarks": comments_remarks,
            "split_sentence": split_sentence
            })

    combined = [dataset, split_list]
    combined_df = pd.concat(combined, axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()] # Remove duplicate columns

    emp_id = []
    comment_full = []
    explode_sentence = []

    for _, row in combined_df.iterrows():
        for sentence in row.split_sentence:
            emp_id.append(row.person_number)
            comment_full.append(row.comments_remarks)
            explode_sentence.append(sentence)

    split_explode = pd.DataFrame({
            "person_number": emp_id,
            "comments_remarks": comment_full,
            "split_explode": explode_sentence
            })

    return split_explode


def sentiment_classifier_model(sentiment_vector, sentiment_classifier, df, comment_col):

    # Open Classifier model
    pkl_filename = sentiment_vector
    # print("Sentiment vectorizer Opening....")
    with open(pkl_filename, 'rb') as file:
        vectorizer = pickle.load(file)

    # Open Classifier model
    pkl_filename = sentiment_classifier
    # print("Sentiment Model Opening....")
    with open(pkl_filename, 'rb') as file:
        classifier = pickle.load(file)

    df[comment_col] = df[comment_col].str.lower()
    df[comment_col] = df[comment_col].map(lambda x: re.sub(r'\W+', ' ', x))

    features = df[comment_col]
    Answer_vect = vectorizer.transform(features).toarray()

    # Make prediction
    prediction = classifier.predict(Answer_vect)
    predictionProb = classifier.predict_proba(Answer_vect)


    mnb_results = np.array(list(
                            zip(df[comment_col]
                                ,prediction
                                ,predictionProb[:, 0]
                                ,predictionProb[:, 1]
                                ,predictionProb[:, 2])
                                )
                            )

    mnb_results = pd.DataFrame(mnb_results,
                               columns=['comment',
                                        'prediction_sentiment',
                                        'negative',
                                        'neutral',
                                        'positive']
                               )

    # Calculate sentiment logic
    mnb_results["sentiment_classifier"] = np.where(
                                                    mnb_results['prediction_sentiment'] == '0',
                                                        'negative',
                                           np.where(
                                                    mnb_results['prediction_sentiment'] == '1',
                                                        'neutral',
                                            np.where(
                                                    mnb_results['prediction_sentiment'] == '2',
                                                            'positive', 'none'
                                                    )
                                                )
                                            )

    # Get max probability among sentiment columns
    mnb_results["sentiment_prob"] = mnb_results[['negative', 'neutral', 'positive']].max(axis=1)

    return mnb_results

# This is standard function which required to define models when deploy on AML services
def init():
    global sentiment_vector
    global sentiment_classifier

    sentiment_vector = Model.get_model_path('./sentiment_vector.pkl')
    sentiment_classifier = Model.get_model_path('./sentiment_classifier.pkl')


def run(text):

    data_lst = text.split(":")[1].replace('[', '').replace(']', '').strip().split(",")
    dataset = pd.DataFrame(data_lst, columns=['comments_remarks'])
    dataset['person_number'] = dataset.index + 1
    dataset['person_number'] = dataset['person_number'].astype(str)
    dataset_org = dataset['comments_remarks']

    ############Multiple sentiments per sentence##################
    # Split sentences
    spliter = split_sentences(dataset)
    splitter_org = spliter[['person_number', 'comments_remarks', 'split_explode']]

    # Calling sentiment classifier method
    multiple_sentiment = sentiment_classifier_model(sentiment_vector, sentiment_classifier, spliter,
                                              'split_explode')

    # Combine with original dataframe
    multiple_sentiment_all = [splitter_org,
                              multiple_sentiment[['prediction_sentiment',
                                                  'negative',
                                                  'neutral',
                                                  'positive',
                                                  'sentiment_classifier',
                                                  'sentiment_prob']]
                              ]

    multiple_sentiment_combined = pd.concat(multiple_sentiment_all, axis=1)

    # Overall Sentiment classification using classification model
    sentiment_classification = sentiment_classifier_model(sentiment_vector, sentiment_classifier, dataset,
                                                    'comments_remarks')

    # Combine both Overall sentiment classifier model and area classification model
    combined = [dataset_org,
                sentiment_classification[['prediction_sentiment',
                                          'negative',
                                          'neutral',
                                          'positive',
                                          'sentiment_classifier',
                                          'sentiment_prob']]
                ]

    combined_df = pd.concat(combined, axis=1)

    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    result = combined_df.merge(multiple_sentiment_combined, how='left', left_on=['comments_remarks'],
                               right_on=['comments_remarks'])

    final_result = result[['comments_remarks',
                           'sentiment_classifier_x',
                           'sentiment_prob_x',
                           'negative_x',
                           'positive_x',
                           'neutral_x',
                           'split_explode',
                           'sentiment_classifier_y',
                           'sentiment_prob_y'
                           ]]

    final_result.columns = ['comment',
                            'overall_sentiment',
                            'overall_sentiment_prob',
                            'negative',
                            'positive',
                            'neutral',
                            'split_explode',
                            'sentiment',
                            'sentiment_prob']

    final_result['overall_sentiment_prob'] = final_result['overall_sentiment_prob'].astype('str')
    final_result['sentiment_prob'] = final_result['sentiment_prob'].astype('str')

    final_json = (final_result.groupby(['comment',
                                        'overall_sentiment',
                                        'overall_sentiment_prob',
                                        'negative',
                                        'positive',
                                        'neutral'], as_index=False)
                  .apply(lambda x: x[['split_explode', 'sentiment', 'sentiment_prob']].to_dict('r'))
                  .reset_index()
                  .rename(columns={0: 'split_sen'}))
    # .to_json(orient='records'))

    json_obj = final_json.to_dict('records')

    return json_obj