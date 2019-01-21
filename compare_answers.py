import numpy as np
import argparse


def load_answers(txt_path):
    result_pairs = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            words = line.split()
            nonterminal = words[0]
            pairs = np.array(list(map(int, words[1:]))).reshape(-1, 2)
            indices = np.lexsort((pairs[:, 1], pairs[:, 0]))
            result_pairs[nonterminal] = pairs[indices]
    return result_pairs


def check_answers(result_path, correct_path):
    result_pairs, correct_pairs = load_answers(result_path), load_answers(correct_path)
    is_correct = True
    for nonterminal, true_pairs in correct_pairs.items():
        test_pairs = result_pairs.get(nonterminal, None)
        if nonterminal not in result_pairs.keys():
            print("Nonterminal {} from correct pairs isn't contained in yours".format(nonterminal))
            is_correct = False
        elif true_pairs.shape[0] != test_pairs.shape[0]:
            print("Correct amount of points {}, yours: {}".format(true_pairs.shape[0],
                                                                  test_pairs.shape[0]))
            is_correct = False
        elif np.sum(true_pairs - test_pairs) > 0:
            print("Amount of points is the same, but points differ.")
            is_correct = False
    return is_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str, help='path to txt with deferred output')
    parser.add_argument('correct_answer', type=str, help='path to txt with correct answer')
    args = parser.parse_args()

    result = check_answers(args.result, args.correct_answer)
    if result:
        print("Correct.")
    else:
        print("".join(["-"] * 20))
        print("Wrong.")
