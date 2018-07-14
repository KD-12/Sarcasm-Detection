# Sarcasm-Detection
************************************************************************************************

I. Dataset:
Self Annotated Reddit Corpus: hosted at http://nlp.cs.princeton.edu/SARC/0.0/pol/
It is a labelled data of reddit comments, with 0 as non-sarcastic and 1 as sarcastic. The data we used is balanced, i.e the ratio of sarcastic and non-sarcastic comments is equal.
Specifically, we downloaded: train-balanced.csv.bz2


II. Code files being submitted:
1. ViewData.py - File for generating required input text files from the bz2 downloaded of SARC.
2. build_train.py - File for making train data per user, alongwith negative samples. The output is a pickle file with the data extracted and word embeddings learnt. The word embeddings are made by using code we made in Assignment 1, alongwith all the code in the template provided to us. (For the Assigment1 code that we reused, the submission does not include the model pickle files (we have only included the python file whose code was run) as they would not get uploaded to Blackboard.)
3. user2vec.py - File which generates user embedddings by taking in the train data made from previous file. The output is a pickle file containing the trained user embeddings.
4. cnn.py - File which implements CUE-CNN by taking as input the user embeddings formed from the previous file.
5. Intermediate text and pickle files are generated, which are read in the next stages appropriately. All files will be created in the same folder where the code is placed. The intermediate text files are being submitted, the pickle files are not.


III. Libraries used:
Tensorflow along with Python 3.5. For making the train-test split of the user data, we used the functions from sklearn as also for calculation of scores.


IV. How to run:
1. For running the system, the files would be run in the order as:
ViewData.py -> build_train.py -> user2vec.py -> cnn.py
The final output are scores for binary classification of the input text as sarcastic and non-sarcastic.


V. References:
Code:
We have referred the code created by the original authors of the paper for creating the user embeddings.
It is hosted at: https://github.com/samiroid/usr2vec

Theory:
https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer
http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/convnets/
https://stackoverflow.com/questions/44151760/received-a-label-value-of-1-which-is-outside-the-valid-range-of-0-1-python
https://stats.stackexchange.com/questions/255105/why-is-the-validation-accuracy-fluctuating
https://github.com/samiroid/CUE-CNN/tree/master/code
http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
https://www.quora.com/What-might-be-the-cause-of-the-many-fluctuations-in-validation-accuracy-during-training-of-my-deep-neural-network
