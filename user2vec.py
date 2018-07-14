# Code reffered and adapted from ASSignment 1 Assignment 3 and https://github.com/samiroid/usr2vec
#and https://github.com/samiroid/CUE-CNN/tree/master/code

# coding: utf-8

# In[1]:


import tensorflow as tf
import collections
from collections import namedtuple
import pickle
import parameters
import numpy as np


# In[2]:


User2Vec = namedtuple('User2Vec', ['user_ids', 'app', 'sent_ids', 'neg_ids', 'optimizer', 'loss', 'normalized_U', 'score'])


# In[3]:


batch_size = 10
embedding_size = 128
max_len = 10


# In[4]:


user_train_data = pickle.load(open("user_train_data.pkl", 'rb'))

def generate_data():

    train_user_ids = []
    test_user_ids = []
    train_sents = []
    test_sents = []
    neg_samples = []
    user_dict = {}

    for user_id, train, test, neg_words in user_train_data:
        try:
            uid = user_dict[user_id]
        except KeyError:
            user_dict[user_id] = len(user_dict)
            uid = user_dict[user_id]

        for index in range(len(train)):
            train_user_ids.append(uid)

            # make length of all train sents equal
            if len(train[index]) < max_len:
                train[index].extend([0] * (max_len - len(train[index])))
            elif len(train[index]) > max_len:
                for i in range(len(train[index]) - max_len):
                    train[index].pop()
            train_sents.append(train[index])

            neg_samples.append(neg_words[index])

        for index in range(len(test)):
            test_user_ids.append(uid)

            if len(test[index]) < max_len:
                test[index].extend([0] * (max_len - len(test[index])))
            elif len(test[index]) > max_len:
                for i in range(len(test[index]) - max_len):
                    test[index].pop()
            test_sents.append(test[index])

    return train_user_ids, test_user_ids, train_sents, test_sents, neg_samples


# In[5]:


def hinge_loss(user_embeds, word_embeds, neg_sample_ids):
    pos_score = tf.matmul(user_embeds, word_embeds, transpose_b = True)
#    print('pos_score: ', pos_score)

    neg_sample_ids_t = tf.transpose(neg_sample_ids)

    neg_score = tf.matmul(user_embeds, neg_sample_ids, transpose_b=True)
#    print('neg_score: ', neg_score)

    loss = tf.maximum(0.0, 1 - tf.add(pos_score,neg_score))

    return loss


# In[6]:


def build_model(sess, graph, embed_matrix_rows, n_users, embed_matrix):
    lam = 1e-8
    with graph.as_default():
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)

            # u_j Placeholder 0
            user_ids = tf.placeholder(tf.int32, shape=[batch_size,], name='user_ids')
#            print('user_ids: ', user_ids)
            U = tf.Variable(tf.random_uniform([n_users, embedding_size], -1.0, 1.0))
#            print('U: ', U)
            user_embeds = tf.nn.embedding_lookup(U, user_ids)
#            print('user_embed: ', user_embeds)
            # e_i Placeholder 1
            E = tf.Variable(embed_matrix, dtype=tf.float32)
#            print('E: ', E)
            sent_ids = tf.placeholder(tf.int32, shape=[batch_size,max_len], name='sent_ids')
#            print('sent_ids: ', sent_ids)
            word_embeds = tf.nn.embedding_lookup(E, sent_ids)
#            print('word_embeds :', word_embeds)
            word_embeds = tf.reshape(word_embeds, [-1, embedding_size])
#            print('after reshape word_embeds :', word_embeds)

            # e_l Placeholder 2
            neg_ids = tf.placeholder(tf.int32, shape=[batch_size,max_len], name='neg_ids')
#            print("neg ids: ", neg_ids)
            neg_sample_ids = tf.nn.embedding_lookup(E, neg_ids)
#            print("neg sample ids: ", neg_sample_ids)
            neg_sample_ids = tf.reshape(neg_sample_ids, [-1, embedding_size])
#            print("after reshape neg sample ids: ", neg_sample_ids)

            hinge_loss_1 =  hinge_loss(user_embeds, word_embeds, neg_sample_ids)
            #U_regularizer = tf.nn.l2_loss(U)
            #E_regularizer = tf.nn.l2_loss(E)

            loss = tf.reduce_mean(hinge_loss_1 )#+ (lam/2) *  U_regularizer + (lam/2) *  E_regularizer)


        # Construct the SGD optimizer using a learning rate of 1.0.
        #optimizer = tf.train.GradientDescentOptimizer(1e-6).minimize(loss, global_step=global_step)


        optimizer = tf.train.GradientDescentOptimizer(1e-6)
        grads = optimizer.compute_gradients(loss)
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)

        #AdamOptimizer

        #optimizer = tf.train.AdamOptimizer(1e-6)
        #grads = optimizer.compute_gradients(loss)
        #clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        #app = optimizer.apply_gradients(clipped_grads)

        #MomentumOptimizer
        #optimizer = tf.train.MomentumOptimizer(1e-5,0.9)
        #grads = optimizer.compute_gradients(loss)
        #clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        #app = optimizer.apply_gradients(clipped_grads)


        norm = tf.sqrt(tf.reduce_sum(tf.square(U), 1, keep_dims=True))
        normalized_U = U / norm

        # generating score by adding the probabilities with which wrd2vec and user2vec were trained to do evaluation
        # and finding the U with best score to stored for future use in CUE-CNN
        step_1 = tf.matmul(E,U,transpose_b = True)
        #print("step_1", step_1)
        step_2 = tf.nn.softmax(tf.transpose(step_1))
        step_3= tf.log(step_2)
        #print("step_2", step_2)
        score  = tf.reduce_mean(step_3,1)
        #print("score", score)

        tf.global_variables_initializer().run()
        print(" Train Initialized")

    model = User2Vec(user_ids, app, sent_ids, neg_ids, optimizer, loss, normalized_U, score)
    return model


# In[7]:


def train(sess, model, n_users):
    max_num_steps = 1001
    average_loss_step = 100

    train_user_ids, test_user_ids, train_sents, test_sents, neg_samples = generate_data()
    print("data: ", len(train_user_ids), len(test_user_ids), len(train_sents), len(test_sents), len(neg_samples))

    max_score = float('-inf')

    for step in range(max_num_steps):
            predictions = []
            average_loss = 0
            average_loss_v = 0
            cnt = 0
            for start in range(0, len(train_user_ids), batch_size):
                end = (start + batch_size) % len(train_user_ids)
                if end < start:
                    start -= end
                    old_end = end
                    end = len(train_user_ids)

                users_in_batch, batch_inputs, batch_neg_samples = train_user_ids[start:end], train_sents[start:end], neg_samples[start:end]
                feed_dict = {model.user_ids.name:users_in_batch, model.sent_ids.name: batch_inputs,
                                                       model.neg_ids.name: batch_neg_samples}

                _, loss_val = sess.run([model.app, model.loss], feed_dict=feed_dict)
                average_loss += loss_val
                cnt += 1

            if step % average_loss_step == 0:
                if step > 0:
                    average_loss /= cnt * average_loss_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            cnt = 0
            total_score = 0
            for start in range(0, len(test_user_ids), batch_size):
                end = (start + batch_size) % len(test_user_ids)
                if end < start:
                    start -= end
                    end = len(test_user_ids)

                users_in_batch, batch_test = test_user_ids[start:end], test_sents[start:end]
                feed_dict = {model.user_ids.name: users_in_batch, model.sent_ids.name: batch_test}

                score = sess.run([model.score], feed_dict = {model.user_ids.name: users_in_batch,
                                                             model.sent_ids.name: batch_test})
                print("score: ", score)
                score = np.mean(score)
                total_score += score
                cnt += 1

            average_score = total_score / cnt
            print("average score: ", average_score)
            if average_score > max_score:
                max_score = average_score
                user_embeddings = model.normalized_U.eval()

#    print("user embeddings: ", type(user_embeddings))
    print("Returning Best U")
    return user_embeddings


# In[9]:


if __name__ == '__main__':

    # load pickled word embeddings
    # because we want the number of users which we pickled here
    embed_matrix, unigram_prob, wrd2idx, word_counter, n_users = pickle.load(open("train_embeddings.pkl", 'rb'))

    user_train_data = pickle.load(open("user_train_data.pkl", 'rb'))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        model = build_model(sess, graph, embed_matrix.shape[0], n_users, embed_matrix)
        user_embeddings = train(sess, model, n_users)

    pickle.dump(user_embeddings, open('user_embeddings.pkl', 'wb'))
