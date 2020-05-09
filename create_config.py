import configparser

config = configparser.ConfigParser()

config['MODEL'] = {
    'sentiment_vector': 'models/sentiment_vector.pkl',
    'sentiment_classifier': 'models/sentiment_classifier.pkl'
}

config['TRAINING_PATH'] = {
    'sentiment_training_data_path': 'training_data/sentiment_sample.csv'
}

config['PARAMETERS'] = {
    'classification_test_size': '0.2'
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)