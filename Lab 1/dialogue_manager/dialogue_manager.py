import numpy as np

def compare_answers(predicted_ans, expected_ans):
    return predicted_ans == expected_ans

# Call answer_tracker every time a new answer was predicted by the system

# prev_answers is a list of all the guesses made, with the predicted 
# answer_cnt an array of 2 integers: the number of wrong answers and the number of right answers 
# new_predicted is the current predicted answer: 0 = negative; 1 = positive
# new_expected is the current real answer: 0 = negative; 1 = positive
def answer_tracker(new_predicted, new_expected, prev_answers = [], answer_cnt = np.zeros(2)):
    answer_cnt[1 if compare_answers(new_predicted, new_expected) == True else 0] += 1
    prev_answers.append([int(answer_cnt[0]+answer_cnt[1]), new_predicted, new_expected, compare_answers(new_predicted, new_expected)])
    return prev_answers, answer_cnt


def display_answers(prev_answers, answer_cnt):
    print("Networks's answers: {} right; {} wrong.\n".format(answer_cnt[1], answer_cnt[0]))
    for row in range(len(prev_answers)):
        print("\tNb {} ;\tPredicted: {};\tExpected:{};\tIs the answer satisfying ? {}".format(prev_answers[row][0], ("Negative" if prev_answers[row][1] == 0 else "Positive"), ("Negative" if prev_answers[row][2] == 0 else "Positive"), ("No" if prev_answers[row][3] == 0 else "Yes")))
