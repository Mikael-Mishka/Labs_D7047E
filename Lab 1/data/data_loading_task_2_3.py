import os
import pickle
import queue
import re
import sys
import time
from typing import List

import numpy
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import queue
import multiprocessing
from multiprocessing import freeze_support
import nltk
from multiprocessing import managers
import pathlib
import logging


def progress_worker(tok_queue: multiprocessing.Queue, dct_queue: multiprocessing.Queue, freq):
    """
    Prints a progress bar and the number of items in the queue
    """
    start_size = tok_queue.qsize()
    ASCII_PREV_LINE = "\033[F"
    ASCII_NEXT_LINE = "\033[E"
    ASCII_CLEAR_FROM_CURSOR = "\033[0J"
    prev_size = start_size
    dct_prev_size = -1
    while tok_queue.qsize() > 0 or dct_queue.qsize() > 0:
        # Print bar as (start_size - current_size) * =
        curr_size = tok_queue.qsize()
        dct_curr_size = dct_queue.qsize()
        if curr_size != prev_size or dct_prev_size != dct_curr_size:
            # start_size >= curr_size
            print(ASCII_CLEAR_FROM_CURSOR, end='', flush=True)
            progress = (start_size - curr_size) / start_size
            progress *= 100
            progress = int(progress)
            prog = "=" * progress
            remaining = ">" + " " * (100 - progress - 1)
            print(f"|{prog}{remaining}|", flush=True)
            print(f"Queue size: {curr_size}|\tDict operation queue:{dct_queue.qsize()}", flush=True)
            print(ASCII_PREV_LINE * 2, end='', flush=True)
            prev_size = curr_size
            dct_prev_size = dct_curr_size
        time.sleep(1 / freq)

    print(f"\nDONE!, Items processed: {start_size} out of {start_size}{ASCII_CLEAR_FROM_CURSOR}")


def count_tokens(data: pd.DataFrame, dict_queue: multiprocessing.Queue, swords):
    words = set()
    # data['Text'] = data['Text'].str.lower()
    # data['Text'] = data['Text'].str.replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)  # remove emails
    # data['Text'] = data['Text'].str.replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '',
    #                                        regex=True)  # remove IP address
    # data['Text'] = data['Text'].replace(r'[^\w\s]', '', regex=True)  # remove special characters
    # data['Text'] = data['Text'].replace(r'\d', '', regex=True)
    token_pat = re.compile(r'\w+', re.IGNORECASE)
    for index, row in data.iterrows():
        word_tokens = token_pat.findall(row['Text'])
        for w in word_tokens:
            w = w.lower()
            if w in words:
                continue

            words.add(w)
            dict_queue.put(w)


def token_counter(dict_queue: multiprocessing.Queue, tok_queue: multiprocessing.Queue, swords):
    while tok_queue.qsize() > 0:
        try:
            data = tok_queue.get(True, 1)
            count_tokens(data, dict_queue, swords)
        except Exception as e:
            if tok_queue.qsize() <= 0 or tok_queue.empty() or isinstance(e, queue.Empty):
                break


def dict_handler(set_res: list, dict_queue: multiprocessing.Queue):
    tokens = set()
    while dict_queue.qsize() > 0:
        try:
            w = dict_queue.get(block=True, timeout=10)
            tokens.add(w)
        except Exception as e:
            if dict_queue.qsize() <= 0 or dict_queue.empty() or isinstance(e, queue.Empty):
                break

    set_res.append(tokens)


def prepare_task_2_and_3_data():
    data_path = pathlib.Path("Reviews.csv")

    # Load the data
    data = pd.read_csv(str(data_path))
    vocab = set()

    manager = managers.SyncManager()
    manager.start()
    shared_set_result = manager.list()
    nltk.download('punkt')
    nltk.download('stopwords')
    swords = set(stopwords.words('english'))

    # 1 process per core
    tok_queue = multiprocessing.Queue()
    dict_queue = multiprocessing.Queue()
    pool = [multiprocessing.Process(target=token_counter, args=(dict_queue, tok_queue, swords)) for _ in
            range(multiprocessing.cpu_count())]
    dct_pool = [multiprocessing.Process(target=dict_handler, args=(shared_set_result, dict_queue))
                for _ in range(multiprocessing.cpu_count())]
    pool.extend(dct_pool)
    dct_pool.clear()

    prog_process = multiprocessing.Process(target=progress_worker, args=(tok_queue, dict_queue, 7))

    # Split up the data and put it in the queue
    data: pd.DataFrame
    chunk_size = 10000
    import math
    print(f"Creating {math.ceil(len(data) / chunk_size)} chunks")
    print(f"There are {math.ceil(len(data) / chunk_size) / len(pool)} tasks per process")
    for i in range(math.ceil(len(data) / chunk_size)):
        tok_queue.put(data[i * chunk_size:(i + 1) * chunk_size])

    # Start the processes
    print(f"Starting {len(pool)} processes")
    prog_process.start()
    for p in pool:
        p.start()

    for p in pool:
        p.join()
        p.close()

    pool.clear()
    import gc
    gc.collect()

    ASCII_CLEAR_FROM_CURSOR = "\033[0J"
    print(f"Token workers finished! {ASCII_CLEAR_FROM_CURSOR}")

    prog_process.join()
    prog_process.close()

    shared_set_result = list(shared_set_result)
    shared_tkns = set()
    for i, st in enumerate(shared_set_result, start=1):
        print(f"Handling dictionary {i}/{len(shared_set_result)}\r")
        shared_tkns |= st

    print(f"Vocab size: {len(shared_tkns)}",
          f"Token items: ",
          *(list(sorted(shared_tkns, key=len, reverse=True))[:5] + ["..."]), sep='\n')

    print("Tokenization done")
    vocab = list(map(lambda s: s.lower(), shared_tkns))
    vocab.sort()
    special_tokens = ['<PAD>', '<UNK>', '<STP_WORD>', '<EOS>']
    vocab = special_tokens + vocab

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
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
    token_pat = re.compile(r'\w+', re.IGNORECASE)
    pad_len = int(np.percentile(list(map(lambda x: len(token_pat.findall(x)), data["Text"].values)),  99))
    print(f"PAD size: {pad_len}")

    def prepare_chunk(reviews_text, reviews_score, i):
        training_data, validation_data, training_labels, validation_labels = train_test_split(
            reviews_text,
            reviews_score,
            test_size=0.20,
            shuffle=True
        )

        # Convert string data in the '_data' arrays to token indices
        def tokenize(data):
            res = list()
            for word in token_pat.findall(data):
                word = word.lower()
                res.append(word2idx.get(word, word2idx['<UNK>']))

            return res

        def pad(data: list):
            nonlocal pad_len
            if pad_len == -1:
                pad_len = len(max(data, key=len))

            for i, e in enumerate(data):
                if pad_len > len(e):
                    data[i].extend([word2idx["<PAD>"]] * (pad_len - len(e)))
                elif pad_len < len(e):
                    del data[i][pad_len:]

            return data

        print(training_data.shape)
        padded_data = pad([tokenize(review) for review in training_data])
        with open(f"padded_data_p{i}.pkl", "wb") as f:
            pickle.dump(padded_data, f)
        training_data = torch.tensor(pad([tokenize(review) for review in training_data]), dtype=torch.int64)
        validation_data = torch.tensor(pad([tokenize(review) for review in validation_data]), dtype=torch.int64)
        training_labels, validation_labels = torch.tensor(training_labels, dtype=torch.int64), torch.tensor(
            validation_labels, dtype=torch.int64)

        # Convert labels to class indices
        training_labels -= 1
        validation_labels -= 1

        with open(f"training_tensors_p{i}.pkl", "wb") as f:
            pickle.dump((training_data, training_labels), f)

        with open(f"validation_tensors_p{i}.pkl", "wb") as f:
            pickle.dump((validation_data, validation_labels), f)

        print(training_data)

        # Convert the data to a torch.Tensor

        print(training_labels)

    chunk_size = 69_000
    print(f"Creating {len(data)} chunks")
    chunk = data
    prepare_chunk(chunk["Text"].values, chunk["Score"].values, 0)


if __name__ == "__main__":
    freeze_support()
    prepare_task_2_and_3_data()
