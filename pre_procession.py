import os
import csv
import random
import nltk
from nltk import word_tokenize
nltk.download('punkt')

data_dir = './data'


def analyze_dataset():
    question_sets = []  # list of sets
    dissimilar_question_sets = []
    recorded_questions = {}  # {'index': index, 'set_index': set_index}
    sentences = {}
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            idx1 = int(row[1])
            idx2 = int(row[2])
            is_similar = int(row[5])
            if idx1 in recorded_questions.keys():
                temp = recorded_questions[idx1]
                if is_similar:
                    if idx2 not in question_sets[temp]:
                        question_sets[temp].add(idx2)
                else:
                    if idx2 not in dissimilar_question_sets[temp]:
                        dissimilar_question_sets[temp].add(idx2)
            elif idx2 in recorded_questions.keys():
                temp = recorded_questions[idx2]
                if is_similar:
                    if idx1 not in question_sets[temp]:
                        question_sets[temp].add(idx1)
                else:
                    if idx1 not in dissimilar_question_sets[temp]:
                        dissimilar_question_sets[temp].add(idx1)
            else:
                temp = len(question_sets)
                recorded_questions[idx1] = temp
                recorded_questions[idx2] = temp
                question_sets.append(set([idx1]))
                dissimilar_question_sets.append(set())
                if is_similar:
                    question_sets[temp].add(idx2)
                else:
                    dissimilar_question_sets[temp].add(idx2)
                    question_sets.append(set([idx2]))
                    dissimilar_question_sets.append(set([idx1]))
            sentences[idx1] = row[3]
            sentences[idx2] = row[4]
    train_queries = []
    train_docs = []
    validate_queries = []
    validate_docs = []
    validate_ground_truths = []
    test_queries = []
    test_docs = []
    test_ground_truths = []
    validate_set_size = 2000

    for i, question_set in enumerate(question_sets):
        dissimilar_question_set = dissimilar_question_sets[i]
        if (len(question_set) >= 2 and len(dissimilar_question_set) >= 1) and (len(question_set) >= 3 or len(dissimilar_question_set) >= 2):
            question_set = list(question_set)
            dissimilar_question_set = list(dissimilar_question_set)
            train_queries.append(' '.join(word_tokenize(sentences[question_set[0]])) + '\n')
            train_docs.append(' '.join(word_tokenize(sentences[question_set[1]])) + '\n')
            train_docs.append(' '.join(word_tokenize(sentences[dissimilar_question_set[0]])) + '\n')
            # debug
            # test_queries.append(' '.join(word_tokenize(sentences[question_set[0]])) + '\n')
            # test_docs.append(' '.join(word_tokenize(sentences[question_set[1]])) + '\n')
            # test_ground_truths.append('1\n')
            # test_queries.append(' '.join(word_tokenize(sentences[question_set[0]])) + '\n')
            # test_docs.append(' '.join(word_tokenize(sentences[dissimilar_question_set[0]])) + '\n')
            # test_ground_truths.append('0\n')
            if len(question_set) >= 3:
                test_docs.append(' '.join(word_tokenize(sentences[question_set[2]])) + '\n')
                test_queries.append(' '.join(word_tokenize(sentences[question_set[0]])) + '\n')
                test_ground_truths.append('1\n')
            if len(dissimilar_question_set) >= 2:
                test_docs.append(' '.join(word_tokenize(sentences[dissimilar_question_set[1]])) + '\n')
                test_queries.append(' '.join(word_tokenize(sentences[question_set[0]])) + '\n')
                test_ground_truths.append('0\n')
    test_queries = test_queries[:validate_set_size*2]
    test_docs = test_docs[:validate_set_size*2]
    test_ground_truths = test_ground_truths[:validate_set_size*2]
    validate_queries = test_queries[:validate_set_size]
    validate_docs = test_docs[:validate_set_size]
    validate_ground_truths = test_ground_truths[:validate_set_size]
    test_queries = test_queries[validate_set_size:]
    test_docs = test_docs[validate_set_size:]
    test_ground_truths = test_ground_truths[validate_set_size:]

    print("train query number: %d, validate query number: %d, test query number: %d" % (len(train_queries), len(validate_queries), len(test_queries)))

    with open(os.path.join(data_dir, 'queries.txt'), 'w') as f:
        f.writelines(train_queries)
    with open(os.path.join(data_dir, 'docs.txt'), 'w') as f:
        f.writelines(train_docs)
    with open(os.path.join(data_dir, 'test_queries.txt'), 'w') as f:
        f.writelines(test_queries)
    with open(os.path.join(data_dir, 'validate_queries.txt'), 'w') as f:
        f.writelines(validate_queries)
    with open(os.path.join(data_dir, 'test_docs.txt'), 'w') as f:
        f.writelines(test_docs)
    with open(os.path.join(data_dir, 'validate_docs.txt'), 'w') as f:
        f.writelines(validate_docs)
    with open(os.path.join(data_dir, 'test_ground_truths.txt'), 'w') as f:
        f.writelines(test_ground_truths)
    with open(os.path.join(data_dir, 'validate_ground_truths.txt'), 'w') as f:
        f.writelines(validate_ground_truths)


def create_dataset_deprecated():
    lines = []
    dis_similar = []
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if int(line[5]) != 1:
                dis_similar.append(line)
            else:
                lines.append(line)

    query_num = 10000
    neg_num = 4
    test_num = 250 * 2  # test set and validate set
    random.shuffle(lines)
    test_lines = lines[query_num:query_num + test_num // 2]
    lines = lines[:query_num]

    queries = []
    docs = []
    idxs = {}
    for line in lines:
        idx1 = int(line[1])
        idx2 = int(line[2])
        if idx1 not in idxs.keys():
            idxs[idx1] = [idx2]
        else:
            idxs[idx1].append(idx2)
        queries.append(' '.join(word_tokenize(line[3])) + '\n')
        docs.append(' '.join(word_tokenize(line[4])) + '\n')
        negs_count = 0
        while negs_count < neg_num:
            rand_idx = random.randint(0, query_num - 1)
            candidate = lines[rand_idx]
            i1 = int(candidate[1])
            i2 = int(candidate[2])
            if i1 == idx1 or i2 == idx1 or i1 in idxs[idx1] or i2 in idxs[idx1]:
                continue
            docs.append(' '.join(word_tokenize(candidate[3])) + '\n')
            negs_count += 1
    with open(os.path.join(data_dir, 'queries.txt'), 'w') as f:
        f.writelines(queries)
    with open(os.path.join(data_dir, 'docs.txt'), 'w') as f:
        f.writelines(docs)

    random.shuffle(dis_similar)
    test_lines = test_lines + dis_similar[test_num // 2]
    # debug
    temp = test_lines[test_num // 2]
    test_queries = [(' '.join(word_tokenize(row[3])) + '\n') for row in test_lines]
    test_docs = [(' '.join(word_tokenize(row[4])) + '\n') for row in test_lines]
    test_ground_truths = [(row[5] + '\n') for row in test_lines]

    validate_queries = test_queries[:test_num // 4] + test_queries[test_num // 2:test_num * 3 // 4]
    validate_docs = test_docs[:test_num // 4] + test_docs[test_num // 2:test_num * 3 // 4]
    validate_ground_truths = test_ground_truths[:test_num // 4] + test_ground_truths[test_num // 2:test_num * 3 // 4]
    test_queries = test_queries[test_num // 4:test_num // 2] + test_queries[test_num * 3 // 4:]
    test_docs = test_docs[test_num // 4:test_num // 2] + test_docs[test_num * 3 // 4:]
    test_ground_truths = test_ground_truths[test_num // 4:test_num // 2] + test_ground_truths[test_num * 3 // 4:]

    with open(os.path.join(data_dir, 'test_queries.txt'), 'w') as f:
        f.writelines(test_queries)
    with open(os.path.join(data_dir, 'validate_queries.txt'), 'w') as f:
        f.writelines(validate_queries)
    with open(os.path.join(data_dir, 'test_docs.txt'), 'w') as f:
        f.writelines(test_docs)
    with open(os.path.join(data_dir, 'validate_docs.txt'), 'w') as f:
        f.writelines(validate_docs)
    with open(os.path.join(data_dir, 'test_ground_truths.txt'), 'w') as f:
        f.writelines(test_ground_truths)
    with open(os.path.join(data_dir, 'validate_ground_truths.txt'), 'w') as f:
        f.writelines(validate_ground_truths)

if __name__ == '__main__':
    # create_dataset()
    analyze_dataset()
