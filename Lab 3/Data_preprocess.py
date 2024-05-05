import cv2
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import numpy as np


def image_preprocess():
    fldr1 = './Flicker8k_Dataset'
    files = os.listdir(fldr1)

    images = []
    name = []

    # Recording the images and the corresponding filenames as they are the only way to match
    # the images with their descriptions

    count = 0
    for file in files:
        name.append(file)
        image = cv2.imread(fldr1 + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        images.append(image)

        count += 1
        if count % 1000 == 0:
            print('Images processed:', count)

    print('Total images processed:', count)
    name_np = np.array(name)
    images_np = np.array(images)

    return name_np, images_np


def text_extractor():
    print('Text extracting started')
    file = './Flicker8k_text/Flickr8k.token.txt'
    f = open(file)
    lines = f.readlines()
    main_df = pd.DataFrame()

    # The descriptions are present in the file Flickr8k.token.txt.
    # The format of representation is "file_imagename.jpg#description_no    description"
    # The below code splits the format and extracts the image file name and corresponding description
    # It also adds the "StartSeq" and "EndSeq" as the starting and Ending token

    for line in lines:
        sent = "Startseq "
        word1, word2 = line.split("\t")
        pic, index = word1.split('#')
        clean_sentence = word2.replace("[^a-zA-Z0-9]", " ")
        sent += clean_sentence[:-3]
        sent += " Endseq"
        # print(sent)
        df = pd.DataFrame([[pic, sent]], columns=['image', 'description'])
        # print(df)
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat([main_df, df])

    main_df.head()
    main_df.to_csv('desc.csv', index=False)
    print('Text extracting done')


# Applying Preprocessing to the language
# Removing any character which is not alphabet and whose length is not greater than 1
# Converting all words to lower case
def text_preprocess(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    words = text.split()
    words_sm = [word.lower() for word in words]

    words_la = [word for word in words_sm if len(word) > 1]

    words_f = [word for word in words_la if word.isalpha()]

    text = ""

    for word in words_f:
        text = text + word + " "

    return text


def tokenizer(y_f):
    # Word map is a dictionary with the word as the key and the corresponding Representing
    # integer and the count of its occurence in the whole dataset
    print('Tokenization started')
    word_map = {}
    count = 1

    for description_grps in y_f:
        for descriptions in description_grps:
            words = descriptions.split()
            for word in words:

                if word in word_map.keys():
                    word_map[word]['count'] = word_map[word]['count'] + 1

                else:
                    word_map[word] = {}
                    word_map[word]['Rep'] = count
                    word_map[word]['count'] = 1
                    count += 1

    print('Different words mapped:', count - 1)
    out_file = open("./word_map.json", "w")
    json.dump(word_map, out_file)
    out_file.close()
    print('Tokenization done')


def caption_preprocess():
    text_extractor()
    df = pd.read_csv("./desc.csv")
    df.head()
    print('Text preprocessing started')
    df['description'] = df['description'].apply(text_preprocess)
    print('Text preprocessing done')
    df.head(20)
    mapping = {}

    # Creating a dictionary which has the image file name as key
    # The dictionary has the corresponding descriptions of the files
    # And the corresponding images as numpy arrays.
    print('Word mapping started')

    i = 0
    while i < len(df):
        if df.iloc[i]['image'] in mapping.keys():
            mapping[df.iloc[i]['image']]['desc'].append(df.iloc[i]['description'])
        else:
            mapping[df.iloc[i]['image']] = {}
            mapping[df.iloc[i]['image']]['desc'] = [df.iloc[i]['description']]

        i += 1
        if i % 5000 == 0:
            print('Captions processed:', i)

    print('Total captions processed:', i)
    print('Word mapping done')
    print('Image mapping started')

    names, images = image_preprocess()

    name_l = list(names)
    keys = [key for key in mapping.keys()]
    for key in keys:
        try:
            ind = name_l.index(key)
            mapping[key]['image'] = images[ind]
        except:
            mapping.pop(key, None)

    print('Image Mapping done')

    X = []
    y = []

    for k in mapping.keys():
        X.append(mapping[k]['image'])
        y.append(mapping[k]['desc'])

    X_f = np.array(X)
    y_f = np.array(y)

    np.save('./X_f.npy', X_f)
    np.save('./y_f.npy', y_f)

    tokenizer(y_f)


def word_embedding():
    print('Word embedding started')

    X_f = np.load('./X_f.npy')
    y_f = np.load('./y_f.npy')
    out_file = open("./word_map.json", "r")
    word_map = json.load(out_file)
    out_file.close()

    Y = []
    for description_grps in y_f:

        Y_temp = []
        for descriptions in description_grps:
            words = descriptions.split()
            temp_sent = []
            for word in words:
                temp_sent.append(word_map[word]['Rep'])
            temp_sent = np.pad(np.array(temp_sent), (0, 40 - len(temp_sent)))
            Y_temp.append(temp_sent)

        #print(Y_temp)
        Y.append(Y_temp)

    Y_n = np.array(Y)

    # X_f_2=X_f/255

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_f, Y_n, test_size=0.2)

    # Creating embedding matrix for the word vectors
    emb_dim = 50
    vocab = len(word_map) + 1
    emb_mat = np.zeros((vocab, emb_dim))

    # For embedding we use Glove6B by the standford NLP
    # This code picks up every word in the dataset and picks its 50d embedding from the glove 6b.
    # So, as we have 9385 words our embedding will have a dimension of 9385x50
    with open('./glove.6B.50d.txt', encoding="utf8") as f:
        for line in f:
            word, *emb = line.split()
            if word in word_map.keys():
                emb_mat[word_map[word]['Rep']] = np.array(emb, dtype="float32")[:emb_dim]

    print('Word embedding done')

    np.save('./emb_mat.npy', emb_mat)
    return X_train, X_test, Y_train, Y_test, emb_mat
