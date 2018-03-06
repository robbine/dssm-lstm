import os
import csv
import random

data_dir = './data'


def analyze_dataset():
    idxs = {}
    # debug
    iter = 0
    with open(os.path.join(data_dir, 'know_question_table.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) not in idxs.keys():
                idxs[int(row[0])] = [int(row[2])]
            else:
                idxs[int(row[0])].append(int(row[2]))
            # debug
            iter += 1
            print("No.%d" % iter)
    # debug
    print("Finished reading. Writing...")
    with open(os.path.join(data_dir, 'dataset_info.txt'), 'w') as f:
        for idx, values in idxs.iteritems():
            f.write('similar to ' + str(idx) + ': ' + ','.join(str(v) for v in values) + '\n')



def create_dataset():
    lines = []
    dis_similar = []
    know2idx = {}
    docs2idx = {}
    with open(os.path.join(data_dir, 'know_question_table.csv'), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line + ['1'])
            know2idx[line[1]] = int(line[0])
            docs2idx[line[3]] = int(line[2])
    max_know_idx = max(know2idx.values())
    max_docs_idx = max(docs2idx.values())
    with open(os.path.join(data_dir, 'right_know_ask_table_log2resample.csv'), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if not line[0]:
                continue
            if not line[1]:
                continue
            know_idx = know2idx[line[0]] if line[0] in know2idx.keys() else ++max_know_idx
            know2idx[line[0]] = know_idx
            doc_idx = docs2idx[line[1]] if line[1] in docs2idx.keys() else ++max_docs_idx
            docs2idx[line[1]] = doc_idx
            lines.append([know_idx, line[0], doc_idx, line[1]], '1')
    with open(os.path.join(data_dir, 'wrong_know_ask_table_log2resample.csv'), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if not line[0]:
                continue
            if not line[1]:
                continue
            know_idx = know2idx[line[0]] if line[0] in know2idx.keys() else ++max_know_idx
            know2idx[line[0]] = know_idx
            doc_idx = docs2idx[line[1]] if line[1] in docs2idx.keys() else ++max_docs_idx
            docs2idx[line[1]] = doc_idx
            dis_similar.append([know_idx, line[0], doc_idx, line[1], '0'])

    query_num = 10000
    neg_num = 4
    test_num = 2500
    random.shuffle(lines)
    test_lines = lines[query_num:int(query_num + test_num / 2)]
    lines = lines[:query_num]

    query_id_dict = {}

    queries = []
    docs = []
    idxs = {}
    for line in lines:
        idx1 = int(line[0])
        idx2 = int(line[2])
        if idx1 not in idxs.keys():
            idxs[idx1] = [idx2]
        else:
            print("appending\n")
            idxs[idx1].append(idx2)
        queries.append(line[1])
        docs.append(line[3])
        negs_count = 0
        while negs_count < neg_num:
            rand_idx = random.randint(0, query_num - 1)
            candidate = lines[rand_idx]
            i1 = int(candidate[0])
            i2 = int(candidate[2])
            if i1 == idx1 or i2 == idx1 or i1 in idxs[idx1] or i2 in idxs[idx1]:
                continue
            docs.append(candidate[3])
            negs_count += 1
    with open(os.path.join(data_dir, 'queries.txt'), 'w') as f:
        f.writelines(queries)
    with open(os.path.join(data_dir, 'docs.txt'), 'w') as f:
        f.writelines(docs)

    random.shuffle(dis_similar)
    test_lines = test_lines + dis_similar[:int(test_num / 2)]
    test_queries = [row[1] for row in test_lines]
    test_docs = [row[3] for row in test_lines]
    test_ground_truths = [(row[5] + '\n') for row in test_lines]

    with open(os.path.join(data_dir, 'test_queries.txt'), 'w') as f:
        f.writelines(test_queries)
    with open(os.path.join(data_dir, 'test_docs.txt'), 'w') as f:
        f.writelines(test_docs)
    with open(os.path.join(data_dir, 'test_ground_truths.txt'), 'w') as f:
        f.writelines(test_ground_truths)


if __name__ == '__main__':
    create_dataset()
