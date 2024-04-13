import time
from typing import List

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
        filtered_sent = [w for w in word_tokens if w not in stopwords.words('english')]
        df_ = df_._append({
            "index": row['index'],
            "Score": row['Score'],
            "Text": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data


def prepare_task_2_and_3_data():

    import pathlib
    data_path = pathlib.Path("Reviews.csv")

    # Load the data
    data = pd.read_csv(str(data_path))
    vocab = set()
    import multiprocessing

    from multiprocessing import managers
    manager = managers.SyncManager()
    manager.start()
    shared_tkn_count = manager.dict()

    def progress_worker(queue: multiprocessing.Queue, freq):
        """
        Prints a progress bar and the number of items in the queue
        """
        start_size = queue.qsize()
        ASCII_PREV_LINE = "\033[F"
        ASCII_NEXT_LINE = "\033[E"
        ASCII_CLEAR_FROM_CURSOR = "\033[0J"

        while queue.qsize() > 0:
            print(ASCII_CLEAR_FROM_CURSOR, end='')
            # Print bar as (start_size - current_size) * =
            curr_size = queue.qsize()
            # start_size >= curr_size
            progress = (start_size - curr_size) / start_size
            progress *= 100
            progress = int(progress)
            prog = "=" * progress
            remaining = ">" + " " * (100 - progress - 1)
            print(f"|{prog}{remaining}|")
            print(f"Queue size: {curr_size}")
            print(ASCII_PREV_LINE * 2, end='')
            time.sleep(1/freq)
        print()

    def count_tokens(data: pd.DataFrame, shared_tkn_count: dict):
        for index, row in data.iterrows():
            word_tokens = word_tokenize(row['Text'])
            for w in word_tokens:
                shared_tkn_count[w] = shared_tkn_count.setdefault(w, 0) + 1

    def token_counter(shared_tkn_count: dict, queue: multiprocessing.Queue):
        while True:
            try:
                data = queue.get()
                count_tokens(data, shared_tkn_count)
            # except queue.Empty:
            except Exception as e:
                if not queue.empty():
                    import logging
                    logging.getLogger(__name__).error(f"Error: {e}")
                break

    # 1 process per core
    queue = multiprocessing.Queue()
    pool = [multiprocessing.Process(target=token_counter, args=(shared_tkn_count, queue)) for _ in range(multiprocessing.cpu_count())]
    prog_process = multiprocessing.Process(target=progress_worker, args=(queue, 18))

    # Split up the data and put it in the queue
    data: pd.DataFrame
    chunk_size = 100
    for i in range(0, len(data), chunk_size):
        queue.put(data[i:i+chunk_size])

    # Start the processes
    for p in pool:
        p.start()

    prog_process.start()

    # Wait for the processes to finish
    for p in pool:
        # With a timeout
        p.join()

    shared_tkn_count = dict(shared_tkn_count)

    print(f"Vocab size: {len(shared_tkn_count)}",
          f"Token items: ",
          *sorted(shared_tkn_count.items(), key=lambda e: e[1], reverse=True), sep='\n')

    vocab = list(shared_tkn_count.keys())
    special_tokens = ['<PAD>', '<UNK>', '<STP_WORD>', '<EOS>']
    vocab = special_tokens + vocab
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}



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

    # Convert string data in the '_data' arrays to token indices
    def tokenize(data):
        res = list()
        for word in word_tokenize(data):
            if res in stopwords.words('english'):
                res.append(word2idx['<STP_WORD>'])
                continue

            res.append(word2idx.get(word, word2idx['<UNK>']))

        return res


    training_data = torch.tensor([tokenize(review) for review in training_data], dtype=torch.int64)
    validation_data = torch.tensor([tokenize(review) for review in validation_data], dtype=torch.int64)
    training_labels, validation_labels = torch.tensor(training_labels, dtype=torch.int64), torch.tensor(validation_labels, dtype=torch.int64)

    print(training_data)

    # Convert labels to class indices
    training_labels -= 1
    validation_labels -= 1

    # Convert the data to a torch.Tensor

    print(training_labels)

prepare_task_2_and_3_data()