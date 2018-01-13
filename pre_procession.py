import os
import csv
import random
import nltk
from nltk import word_tokenize
nltk.download('punkt')

data_dir = './data'


def create_dataset():
    similar = {}
    dissimilar = {}
    data = []
    with open(os.path.join(data_dir, 'all.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            value = {'index': [int(row[2])], 'query': row[3], 'sentence': row[4]}
            if int(row[5]) == 1:  # similar
                if int(row[1]) not in similar.keys():
                    similar[int(row[1])] = value
            else:
                if int(row[1]) not in dissimilar.keys():
                    dissimilar[int(row[1])] = value
            data.append(row)
    train_queries = []
    train_docs = []
    train_indexes = set()
    validate_queries = []
    validate_docs = []
    validate_ground_truths = []
    test_queries = []
    test_docs = []
    test_ground_truths = []
    for key, value in dissimilar.items():
        if key not in similar.keys():
            continue
        similar_value = similar[key]
        train_queries.append(' '.join(word_tokenize(similar_value['query'])) + '\n')
        train_docs.append(' '.join(word_tokenize(similar_value['sentence'])) + '\n')
        train_docs.append(' '.join(word_tokenize(value['sentence'])) + '\n')
        train_indexes.add(key)
    train_queries_num = len(train_queries)
    validate_queries_num = train_queries_num // 4
    test_queries_num = train_queries_num // 4
    validate_similar_count = 0
    test_similar_count = 0
    for line in data:
        if (len(validate_queries) == validate_queries_num) and (len(test_queries) == test_queries_num):
            break
        if (int(line[1]) not in train_indexes) and (int(line[2]) not in train_indexes):
            if int(line[5]) == 1:
                if validate_similar_count < validate_queries_num // 2:
                    validate_queries.append(' '.join(word_tokenize(line[3])) + '\n')
                    validate_docs.append(' '.join(word_tokenize(line[4])) + '\n')
                    validate_ground_truths.append(line[5] + '\n')
                    validate_similar_count += 1
                elif test_similar_count < test_queries_num // 2:
                    test_queries.append(' '.join(word_tokenize(line[3])) + '\n')
                    test_docs.append(' '.join(word_tokenize(line[4])) + '\n')
                    test_ground_truths.append(line[5] + '\n')
                    test_similar_count += 1
            else:
                if len(validate_queries) - validate_similar_count < validate_queries_num // 2:
                    validate_queries.append(' '.join(word_tokenize(line[3])) + '\n')
                    validate_docs.append(' '.join(word_tokenize(line[4])) + '\n')
                    validate_ground_truths.append(line[5] + '\n')
                elif len(test_queries) - test_similar_count < test_queries_num // 2:
                    test_queries.append(' '.join(word_tokenize(line[3])) + '\n')
                    test_docs.append(' '.join(word_tokenize(line[4])) + '\n')
                    test_ground_truths.append(line[5] + '\n')

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
    create_dataset()
    # analyze_dataset()
