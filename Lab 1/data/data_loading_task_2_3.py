import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Text'] = data['Text'].str.lower()
    data['Text'] = data['Text'].str.replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Text'] = data['Text'].str.replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Text'] = data['Text'].str.replace(r'[^\w\s]','')                                                       # remove special characters
    data['Text'] = data['Text'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Text'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_._append({
            "index": row['index'],
            "Score": row['Score'],
            "Text": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data


def prepare_task_2_and_3_data():
    # Load the data
    data = pd.read_csv('/home/convergent/PycharmProjects/Labs D7047E/Lab 1/data/Reviews.csv')

    # Columns in the data:
    #   Id,
    #   ProductId,
    #   UserId,
    #   ProfileName,
    #   HelpfulnessNumerator,
    #   HelpfulnessDenominator,
    #   Score,
    #   Time,
    #   Summary,
    #   Text

    # Drops all columns that is not The review nor score
    data = data.drop(['Id', 'ProductId',
                      'UserId', 'ProfileName',
                      'HelpfulnessNumerator',
                      'HelpfulnessDenominator',
                      'Time'], axis=1)

    # Prepare all text and score in numpy ndarray before converting to a torch.Tenosr
    reviews_text = data['Text'].values
    reviews_score = data['Score'].values

    # Split the data into training and validation sets
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        reviews_text,
        reviews_score,
        test_size=0.20,
        random_state=0,
        shuffle=True
    )


    #data_percentage_to_use = 0.01

    # Use only 2 percent of the data
    #training_data = training_data[:int(len(training_data) * data_percentage_to_use)]
    #training_labels = training_labels[:int(len(training_labels) * data_percentage_to_use)]
    #validation_data = validation_data[:int(len(validation_data) * data_percentage_to_use)]
    #validation_labels = validation_labels[:int(len(validation_labels) * data_percentage_to_use)]

    # Make sure the amount of labeled datapoints is uniform for training and validation sets
    num_of_datpoints_per_class = 1000

    # Get the indices of the datapoints that are labeled with 1, 2, 3, 4, 5
    indices_1 = np.where(training_labels == 1)[0]
    indices_2 = np.where(training_labels == 2)[0]
    indices_3 = np.where(training_labels == 3)[0]
    indices_4 = np.where(training_labels == 4)[0]
    indices_5 = np.where(training_labels == 5)[0]

    print(len(indices_1), len(indices_2), len(indices_3), len(indices_4), len(indices_5))

    # Get the data for the new training set
    new_data_training = []

    new_data_training.extend(training_data[indices_1][:num_of_datpoints_per_class])
    new_data_training.extend(training_data[indices_2][:num_of_datpoints_per_class])
    new_data_training.extend(training_data[indices_3][:num_of_datpoints_per_class])
    new_data_training.extend(training_data[indices_4][:num_of_datpoints_per_class])
    new_data_training.extend(training_data[indices_5][:num_of_datpoints_per_class])

    # Get the labels for the new data
    new_data_training_labels = []

    new_data_training_labels.extend([1 for _ in range(num_of_datpoints_per_class)])
    new_data_training_labels.extend([2 for _ in range(num_of_datpoints_per_class)])
    new_data_training_labels.extend([3 for _ in range(num_of_datpoints_per_class)])
    new_data_training_labels.extend([4 for _ in range(num_of_datpoints_per_class)])
    new_data_training_labels.extend([5 for _ in range(num_of_datpoints_per_class)])

    # Get the indices of the datapoints that are labeled with 1, 2, 3, 4, 5
    new_data_validation = []

    # Get the indices of the datapoints that are labeled with 1, 2, 3, 4, 5
    indices_1 = np.where(validation_labels == 1)[0]
    indices_2 = np.where(validation_labels == 2)[0]
    indices_3 = np.where(validation_labels == 3)[0]
    indices_4 = np.where(validation_labels == 4)[0]
    indices_5 = np.where(validation_labels == 5)[0]

    # Get the data for the new validation set
    new_data_validation.extend(validation_data[indices_1][:num_of_datpoints_per_class])
    new_data_validation.extend(validation_data[indices_2][:num_of_datpoints_per_class])
    new_data_validation.extend(validation_data[indices_3][:num_of_datpoints_per_class])
    new_data_validation.extend(validation_data[indices_4][:num_of_datpoints_per_class])
    new_data_validation.extend(validation_data[indices_5][:num_of_datpoints_per_class])

    # Get the labels for the new validation data
    new_data_validation_labels = []

    # Get the labels for the new data
    new_data_validation_labels.extend([1 for _ in range(num_of_datpoints_per_class)])
    new_data_validation_labels.extend([2 for _ in range(num_of_datpoints_per_class)])
    new_data_validation_labels.extend([3 for _ in range(num_of_datpoints_per_class)])
    new_data_validation_labels.extend([4 for _ in range(num_of_datpoints_per_class)])
    new_data_validation_labels.extend([5 for _ in range(num_of_datpoints_per_class)])

    # Convert all new_data to numpy arrays
    training_data = np.array(new_data_training)
    training_labels = np.array(new_data_training_labels)
    validation_data = np.array(new_data_validation)
    validation_labels = np.array(new_data_validation_labels)

    print(type(training_data), type(training_labels), type(validation_data), type(validation_labels))

    # vectorize data using TFIDF and transform for PyTorch for scalability (max_Features is DANGEROUS DO NOT SET THIS TOO HIGH)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=500, max_df=0.5, use_idf=True,
                                      norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)  # transform texts to sparse matrix
    training_data = training_data.todense()  # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.LongTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels))
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.LongTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels))

    # y_tensors will be converted to class indicies
    train_y_tensor = train_y_tensor - 1
    validation_y_tensor = validation_y_tensor - 1

    # Prints to confirm the data is loaded and pre-processed
    print("Training data: ", train_x_tensor.shape)
    print("Training labels: ", train_y_tensor.shape)
    print("Validation data: ", validation_x_tensor.shape)
    print("Validation labels: ", validation_y_tensor.shape)
    print("Vocabulary size: ", vocab_size)
    print("Data loaded and pre-processed successfully")

    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer

