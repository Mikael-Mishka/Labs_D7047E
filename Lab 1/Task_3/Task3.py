import torch
import pickle
import pathlib
import random


"""
BELOW IS JANK CODE!
DO NOT USE IN THE FUTURE UWU.
"""

# Get path
abs_path = pathlib.Path().absolute()

# Remove the last part of the path
path = abs_path.parent.parent

path = path / "Lab 1" / "data" / "picklies" / "vocab.pkl"

print(path)

# Unpickle the vocab.pkl file
with open(path, "rb") as f:
    vocab = pickle.load(f)

word2idx = {word: idx for idx, word in enumerate(vocab)}

# Unpickle the re_path.pkl file
path = abs_path.parent.parent / "Lab 1" / "data" / "picklies" / "re_pat.pkl"

with open(path, "rb") as f:
    re_pat = pickle.load(f)

print(re_pat)
print(type(re_pat))


def pad(data: list):
    pad_len = 405
    if pad_len == -1:
        pad_len = len(max(data, key=len))

    for i, e in enumerate(data):
        if pad_len > len(e):
            data[i].extend([word2idx["<PAD>"]] * (pad_len - len(e)))
        elif pad_len < len(e):
            del data[i][pad_len:]

    return data


def tokenize(text, re_pattern):

    res = list()
    for word in re_pattern.findall(text):
        word = word.lower()
        res.append(word2idx.get(word, word2idx['<UNK>']))

    return res


def prepare_transformer_user_input(text: str):

    ids = tokenize(text, re_pat)

    padded = pad([ids])

    user_input_integer_representation = torch.tensor(padded, dtype=torch.int64)

    return user_input_integer_representation


def modified_ANN_interaction():
    from ModifiedSimpleChatbot import main as ModifiedMain

    model = ModifiedMain()

    # pickle the model
    with open("best_ANN.pth", "wb") as f:
        torch.save(model, f)

    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1

    # Define responses
    responses = {
        POSITIVE: ["Positive! You are welcome in the future! Would you like to submit another review?",
                   "Welcome in the future! Glad you liked it! Would you like to submit another review?"],

        NEUTRAL: ["Okay I see. Feel free to submit another review!",
                  "Next time we hope your experience will be better! Would you like to submit another review?"],

        NEGATIVE: ["We are sorry you didn't like it. Would you like to submit another review?",
                   "We will try to improve in the future so your next experience will be better! Would you like to submit another review?"]
    }

    # Define the input loop
    running = True

    print(f"<ANN> Welcome! Type 'exit' if you want to leave.")

    while running:
        user_input = input("<User> ")
        if user_input == "exit":
            running = False
        else:
            user_tensor = prepare_transformer_user_input(user_input)
            output = model(user_tensor)

            response = None

            if output < 0.5:
                response = NEGATIVE

            elif output == 0.5:
                response = NEUTRAL

            else:
                response = POSITIVE

            print(f"<ANN> {responses[response][random.randint(0, 1)]}")


def transformer_interaction():

    # get abs path
    abs_path = pathlib.Path().absolute()

    # Remove "/Task_3" from the path
    path = abs_path.parent.parent

    path = path / "Lab 1" / "Task_2" / "best_transformer.pth"
    print(path)
    # Load the pth file for the transformer model
    model = torch.load(path)

    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1

    # Define responses
    responses = {
        POSITIVE: ["Positive! You are welcome in the future! Would you like to submit another review?",
                   "Welcome in the future! Glad you liked it! Would you like to submit another review?"],

        NEUTRAL: ["Okay I see. Feel free to submit another review!",
                  "Next time we hope your experience will be better! Would you like to submit another review?"],

        NEGATIVE: ["We are sorry you didn't like it. Would you like to submit another review?",
                   "We will try to improve in the future so your next experience will be better! Would you like to submit another review?"]
    }

    # Define the input loop
    running = True

    print(f"<Transformer> Welcome! Type 'exit' if you want to leave.")

    while running:
        user_input = input("<User> ")
        if user_input == "exit":
            running = False
        else:
            user_tensor = prepare_transformer_user_input(user_input)
            output_softmaxed = torch.nn.functional.softmax(model(user_tensor))

            output_softmaxed = output_softmaxed.tolist()[0]

            #print(output_softmaxed)
            #print(type(output_softmaxed))

            response = None

            # Sum of all neurons except the last one
            sum_of_probs = sum(output_softmaxed[:4])

            if sum_of_probs <= 0.55:
                response = NEGATIVE

            elif sum_of_probs > 0.55 and sum_of_probs < 0.65:
                response = NEUTRAL

            else:
                response = POSITIVE

            print(f"<Transformer> {responses[response][random.randint(0, 1)]}")




#transformer_interaction()
modified_ANN_interaction()
#prepare_transformer_user_input("This review sucks")