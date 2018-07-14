# Code reffered and adapted from ASSignment 1 Assignment 3 and https://github.com/samiroid/usr2vec
#and https://github.com/samiroid/CUE-CNN/tree/master/code
#and # http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# coding: utf-8

# In[40]:


import pickle
import numpy as np
import tensorflow as tf
import parameters
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk import tokenize
import re
import math
max_len = 20
batch_size = 10
hidden_size = 25
embedding_size = 128
num_distinct_heights = 3
channels = 1
num_classes = 2
num_filters = 100 # 200 400 600
lam = 1e-4   #1e-3 #1e-2 # ##  1e-1  10
drop_prob = 0.5 # 0.3 0.1 0.0
Heights = [1,3,5] #[5,7,9]# [2,4,6] #  [3,5,7] [4,6,8]
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[41]:


def create_sent_corpus(file):
    line_split = []
    message_list = defaultdict(list)
    with open(file,"r") as fid:
            c = 0
            for line in fid:
                c+=1
                if len(line) > 1:
                    line_split = line.split(" ")
                    message_line = " ".join(line_split[2:])
                    message_list[line_split[0]].append((message_line,line_split[1]))
            print(" total ",c," lines read fron data file")
            return  message_list


# In[42]:


def genTrainExamples(message_list, max_len, wrd2idx):
    feature_list = []
    c = 0
    sent_list = []
    max_len = 20 # keeping length of sentence fixed for the CNN model
    tokenizer = TweetTokenizer(preserve_case=False)
    regex = re.compile(r'[\.\]\%\[\'",\?\*!\}\{<>\^-]')
    for key, message_line_list in message_list.items():
        for sent,label in message_line_list:
            content = tokenizer.tokenize(sent)
            content = [word for word in content if not regex.match(word)]
            for w in content:
                if w in wrd2idx:
                    sent_list.append(wrd2idx[w])
            while len(sent_list) < max_len:         # if len smaller then fill it with unknown words id.
                sent_list.append(0)
            while len(sent_list) > max_len:         # if greater then pop last few words.
                sent_list.pop()
            assert len(sent_list) == max_len
            feature_list.append((key,sent_list,label))
            sent_list = []
    return feature_list




# In[69]:


class CNNModel(object):


    def __init__(self, graph, E, U):


        self.build_graph(graph, E, U)

    def build_graph(self, graph, E, U):

        with graph.as_default():
            total_h_grams = num_distinct_heights * num_filters
            self.embeddings = tf.Variable(E, dtype=tf.float32)
            self.drop_out = tf.placeholder(tf.float32)

            ###### train variables and place holders ####
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size,max_len])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size,])
            train_labels = tf.one_hot(self.train_labels, num_classes) #converting it into categorical class of sarcasm/not sarcasm
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.expand_dims(embed, -1)  # making it 4d for conv2d function

            ################################################# Tune ##############################################
            self.tune_inputs = tf.placeholder(tf.int32, shape=[batch_size,max_len])
            self.tune_labels = tf.placeholder(tf.int32, shape=[batch_size,])
            tune_labels = tf.one_hot(self.tune_labels, num_classes)
            tune_embed = tf.nn.embedding_lookup(self.embeddings, self.tune_inputs)
            tune_embed = tf.expand_dims(tune_embed, -1)

            ############################################## Test #################################################
            self.test_inputs = tf.placeholder(tf.int32, shape=[None,max_len])
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.expand_dims(test_embed, -1)

            #### Initialising variables and place holders for hidden and output layer after CNN##############
            self.user_embeddings = tf.Variable(U, dtype=tf.float32)

            ################################# train placeholders and variables ##########################################
            self.user_id = tf.placeholder(tf.int32, shape=[batch_size,])
            user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_id)

            ######################### tune placeholders and variables ###################################################
            self.tune_user_id = tf.placeholder(tf.int32, shape=[batch_size,])
            tune_user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.tune_user_id)

            ################################# test placeholders and variables ###########################################
            self.test_user_id = tf.placeholder(tf.int32, shape=[None,])
            test_user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.test_user_id)

            ##### hidden layer for CNN with only word embeddings for classification ##########
            #hidden_layer_weights = tf.Variable(tf.truncated_normal([hidden_size, total_h_grams],
            #                                                          stddev=1.0 / math.sqrt(hidden_size)))

            ########## hidden layer for user features concatenated with word features ######
            hidden_layer_weights = tf.Variable(tf.truncated_normal([hidden_size, total_h_grams+ embedding_size],
                                                                      stddev=1.0 / math.sqrt(hidden_size)))

            hidden_layer_bias = tf.Variable(tf.zeros([hidden_size,1]))



            output_layer_weights = tf.Variable(tf.truncated_normal([num_classes ,hidden_size ],

                                                                   stddev=1.0 / math.sqrt(hidden_size)))
            output_layer_bias = tf.Variable(tf.zeros([num_classes,1]))


            # CALLING NEURAL NET CNN for train #####
            self.prediction, filter_weights= self.find_preds(embed,user_embed,hidden_layer_weights,hidden_layer_bias
                                                    ,total_h_grams,output_layer_weights,output_layer_bias)

            self.loss = self.find_loss(self.prediction, train_labels,filter_weights, hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias)

            ## finding loss ########
            optimizer = tf.train.AdadeltaOptimizer(1.0,0.95,1e-08) #with decay rate of 0.95

            #with constant learning Rate
            #optimizer = tf.train.AdamOptimizer(1e-6)

            #optimizer = tf.train.GradientDescentOptimizer(1e-6)

            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

             ############################################ finding accuracy for train set #################################
            self.train_prediction = tf.nn.softmax(tf.transpose(self.prediction))


            ############################# CALLING NEURAL NET CNN for tune ##############################################
            self.tune_preds, tune_filter_weights= self.find_preds(tune_embed,tune_user_embed,hidden_layer_weights,hidden_layer_bias,
                                                                  total_h_grams,output_layer_weights,output_layer_bias)

            self.tune_loss = self.find_loss(self.tune_preds, tune_labels,filter_weights, hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias)
            ############################################ finding accuracy for tune set #################################
            self.tune_prediction = tf.nn.softmax(tf.transpose(self.tune_preds))


            ############################# CALLING NEURAL NET CNN for test ##############################################
            self.test_preds, test_filter_weights= self.find_preds(test_embed,test_user_embed,hidden_layer_weights,hidden_layer_bias,
                                                                  total_h_grams,output_layer_weights,output_layer_bias)

            ########################################### finding accuracy for test set ##################################
            self.test_prediction = tf.nn.softmax(tf.transpose(self.test_preds))


            self.init = tf.global_variables_initializer()


    def find_preds(self, embed, user_embed, hidden_layer_weights, hidden_layer_bias, total_h_grams, output_layer_weights,output_layer_bias):
        pooled_outputs = []
        for f_size in Heights:
            filter_weights = tf.Variable(tf.truncated_normal([f_size, embedding_size, channels, num_filters], stddev=0.1))
            bias_conv = tf.Variable(tf.zeros(num_filters))
            conv = tf.nn.conv2d(embed,filter_weights,strides=[1, 1, 1, 1],padding="VALID")
            additive_bias = tf.nn.bias_add(conv, bias_conv)
            Relu_layer = tf.nn.relu(additive_bias)
            pooled = tf.nn.max_pool(Relu_layer,ksize=[1, max_len - f_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID')
            pooled_outputs.append(pooled)



        concatenate_map_filters = tf.concat(pooled_outputs,num_distinct_heights)
        pooled_outputs = []
        Cs_matrix_before_drop_out = tf.reshape(concatenate_map_filters,[total_h_grams, -1])
        Cs_matrix = tf.nn.dropout(Cs_matrix_before_drop_out, self.drop_out)
        reshaped_user_embed = tf.reshape(user_embed,[embedding_size,-1])

        ##### matrix multiplication without concatenated user multiplication ####
        #H_Cs_U = tf.matmul(hidden_layer_weights,Cs_matrix)
        rnn_input = tf.concat([reshaped_user_embed,Cs_matrix],0)
        H_Cs_U = tf.matmul(hidden_layer_weights,rnn_input)
        hidden_layer_output = tf.add(H_Cs_U,hidden_layer_bias)
        activation_layer_output = tf.nn.relu(hidden_layer_output)
        preds = tf.add(tf.matmul(output_layer_weights,activation_layer_output),output_layer_bias)
        return preds, filter_weights


    def find_loss(self,logits, labels,filter_weights, hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias):

        cue_cnn_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(logits),
                                                                                labels=labels)

        ### L2 Regularizer #########
        filter_weight_regularizer = tf.nn.l2_loss(filter_weights)

        h_weight_regularizer = tf.nn.l2_loss(hidden_layer_weights)
        h_bias_regularizer = tf.nn.l2_loss(hidden_layer_bias)

        output_layer_weights_regularizer = tf.nn.l2_loss(output_layer_weights)
        output_layer_bias_regularizer = tf.nn.l2_loss(output_layer_bias)

        E_regularizer = tf.nn.l2_loss(self.embeddings)
        U_regularizer = tf.nn.l2_loss(self.user_embeddings)

        loss = tf.reduce_mean(cue_cnn_loss + (lam) *  filter_weight_regularizer
                                                + (lam) *  h_weight_regularizer
                                                + (lam) *  h_bias_regularizer
                                                + (lam) *  output_layer_weights_regularizer
                                                + (lam) *  output_layer_bias_regularizer
                                                + (lam) *  E_regularizer
                                                + (lam) *  U_regularizer

                                                )
        return loss

    def train(self, sess, trainFeats, max_len ):
        self.init.run()
        print(" Training Initailized")
        c = 0
        max_num_steps = 1001
        user_idx = {}
        user  = []
        sent  = []
        label = []
        for tuple_i in trainFeats:
            try:
                u_id = user_idx[tuple_i[0]]
            except KeyError:
                user_idx[tuple_i[0]] = len(user_idx)
                u_id = user_idx[tuple_i[0]]
            if u_id > 100: ################################### ADDED for 100 user in training of user2vec
                break
            user.append(u_id)

            sent_i , label_i = tuple_i[1], tuple_i[2]
            sent.append(sent_i)
            label.append(int(float(label_i)))
        average_loss = 0

        split = int( len(user) * 0.9) # 90-10 split of train and  validation set.
        sent_train = sent[:split]
        sent_tune  = sent[split:]
        user_train = user[:split]
        user_tune  = user[split:]
        label_train = label[:split]
        label_tune  = label[split:]
        for step in range(max_num_steps):
            predictions = []
            average_loss = 0
            average_loss_v = 0
            predictions_v = []
            for start in range(0, len(user_train), batch_size):
                end = (start + batch_size) % len(user_train)
                if end < start:
                    start -= end
                    old_end = end
                    end = len(user_train)
                batch_user, batch_inputs, batch_labels = user_train[start:end], sent_train[start:end], label_train[start:end]
                feed_dict = {self.user_id.name:batch_user, self.train_inputs.name: batch_inputs,
                                                       self.train_labels.name: batch_labels, self.drop_out: 0.5 }

                _, loss_val, preds_train = sess.run([self.app, self.loss, self.train_prediction], feed_dict=feed_dict)
                if start % batch_size != 0:

                    preds_train = list(preds_train)
                    for i in range(0, old_end):
                        preds_train.pop(0)
                    preds_train = np.array(preds_train)

                average_loss += loss_val
                predictions.extend(list(np.argmax(preds_train, 1)))

            if step % 5 == 0:

                print("Average loss at step ", step, ": ", average_loss)

                #print("Classsification Report:\n",classification_report(label_train, predictions, target_names=["Sarcasm", "Not Sarcasm"]))


                print("Train Accuracy:", 100.0 * accuracy_score(label_train, predictions), "%\n")

            if step % 10 == 0:
                for start_v in range(0, len(user_tune), batch_size):

                    end_v = (start_v + batch_size) % len(user_tune)
                    if end_v < start_v:
                        start_v -= end_v
                        old_end_v = end_v
                        end_v = len(user_tune)
                    batch_user_v, batch_inputs_v, batch_labels_v = user_tune[start_v:end_v], sent_tune[start_v:end_v], label_tune[start_v:end_v]

                    feed_dict = {self.tune_user_id.name:batch_user_v, self.tune_inputs.name: batch_inputs_v,
                                                                      self.tune_labels.name: batch_labels_v,
                                                                      self.drop_out: 1.0 }

                    tune_loss ,preds_tune = sess.run([self.tune_loss,self.tune_prediction], feed_dict=feed_dict)

                    if start_v % batch_size != 0:

                        preds_tune = list(preds_tune)

                        for i in range(0, old_end_v):
                            preds_tune.pop(0)
                        preds_tune = np.array(preds_tune)
                    average_loss_v += tune_loss
                    predictions_v.extend(list(np.argmax(preds_tune, 1)))
                print("Validation Loss:", average_loss_v)
                #print("Validation Report:\n",classification_report(label_tune, predictions_v, target_names=["Sarcasm", "Not Sarcasm"]))
                print("Validation Accuracy: ", 100.0 * accuracy_score(label_tune, predictions_v), "%\n" )
        print("Train Finished.")

    def test(self, sess, testFeats, max_len ):
        self.init.run()
        print(" Testing Initailized")
        c = 0
        max_num_steps = 1001
        user_idx = {}
        user  = []
        sent  = []
        label = []
        for tuple_i in testFeats:
            try:
                u_id = user_idx[tuple_i[0]]
            except KeyError:
                user_idx[tuple_i[0]] = len(user_idx)
                u_id = user_idx[tuple_i[0]]

            user.append(u_id)

            sent_i , label_i = tuple_i[1], tuple_i[2]
            sent.append(sent_i)
            label.append(int(float(label_i)))
        average_loss = 0

        predictions = []

        for start in range(0, len(user), batch_size):
            end = (start + batch_size) % len(user)
            if end < start:
                start -= end
                old_end = end
                end = len(user)
            batch_user, batch_inputs, batch_labels = user[start:end], sent[start:end], label[start:end]
            feed_dict = {self.test_user_id.name:batch_user, self.test_inputs.name: batch_inputs,
                                                       self.drop_out: 0.5 }

            preds_test = sess.run([self.test_prediction], feed_dict=feed_dict)
            preds_test = preds_test[0]
            if start % batch_size != 0:

                preds_test = list(preds_test)
                for i in range(0, old_end):
                    preds_test.pop(0)
                preds_test = np.array(preds_test)


            predictions.extend(list(np.argmax(preds_test, 1)))


        print("Test Accuracy:", 100.0 * accuracy_score(label, predictions), "%\n")
        print("Test Report:\n",classification_report(label, predictions, target_names=["Sarcasm", "Not Sarcasm"]))


# In[70]:


def init_cnn():
    E,unigram_prob,wrd2idx,word_counter,n_users = pickle.load(open('train_embeddings.pkl', 'rb'))
    U = pickle.load(open("user_embeddings_100.pkl","rb"))
    #print(U.shape)
    message_list = create_sent_corpus("cleaned_data.txt")
    print("Generating Traning Examples")
    trainFeats= genTrainExamples(message_list, max_len, wrd2idx)
    #print(trainFeats)
    print("Done.")
    test_message_list = create_sent_corpus("cleaned_data_test.txt")
    print("Generating Testing Examples")
    testFeats= genTrainExamples(test_message_list, max_len, wrd2idx)
    #print(trainFeats)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()

    model = CNNModel(graph, E, U)

    with tf.Session(graph=graph) as sess:

        model.train(sess, trainFeats, max_len)
        model.test(sess, testFeats,max_len)




# In[ ]:


init_cnn()
