data: https://www.kaggle.com/c/quora-question-pairs/data, only use train.csv

word vector: glove.6B(300d)

rename train.csv to all.csv, rename the word vector file to vector.txt, and put them under data dir

then run pre_procession.py to generate queries.txt, docs.txt, test_queries.txt, test_docs.txt and test_ground_truths.txt

pre_procession is written under py3 because I'm too lazy that I only get nltk installed under my py3 environment, I don't know if it works under py2. the other part of the program is written in py2

in test we use log loss according to this page: https://www.kaggle.com/wiki/LogLoss