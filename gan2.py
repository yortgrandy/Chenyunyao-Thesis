
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import Loadd
import pickle

numinD=100
numinG=100
numD1=100
numD2=80
numD3=1
numG1=200
numG2=300
numG3=300
numG4=200
numG5=100
numG6=100
EPOCH=3000


BATCH_SIZE=5000
lr=0.0001
X = tf.placeholder(tf.float32, shape=[None, numinD])
D_W1 = tf.Variable(tf.contrib.layers.xavier_initializer()([numinD, numD1]))
D_b1 = tf.Variable(tf.zeros(shape=[numD1]))

D_W2 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD1, numD2]))
D_b2 = tf.Variable(tf.zeros(shape=[numD2]))

D_W3 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD2, numD3]))
D_b3 = tf.Variable(tf.zeros(shape=[numD3]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


Z = tf.placeholder(tf.float32, shape=[None, numinG])

G_W1 = tf.Variable(tf.contrib.layers.xavier_initializer()([numinG, numG1]))
G_b1 = tf.Variable(tf.zeros(shape=[numG1]))

G_W2 = tf.Variable(tf.contrib.layers.xavier_initializer()([numG1, numG2]))
G_b2 = tf.Variable(tf.zeros(shape=[numG2]))

G_W3 = tf.Variable(tf.contrib.layers.xavier_initializer()([numG2, numG3]))
G_b3 = tf.Variable(tf.zeros(shape=[numG3]))

G_W4 = tf.Variable(tf.contrib.layers.xavier_initializer()([numG3, numG4]))
G_b4 = tf.Variable(tf.zeros(shape=[numG4]))

G_W5 = tf.Variable(tf.contrib.layers.xavier_initializer()([numG4, numG5]))
G_b5 = tf.Variable(tf.zeros(shape=[numG5]))

G_W6 = tf.Variable(tf.contrib.layers.xavier_initializer()([numG5, numG6]))
G_b6 = tf.Variable(tf.zeros(shape=[numG6]))

theta_G = [G_W1, G_W2, G_W3, G_W4,G_W5,G_W6,G_b1, G_b2, G_b3, G_b4, G_b5, G_b6]

def sample_Z(m, n):
    #return np.random.uniform(-2., 2., size=[m, n])
    return np.random.normal(0, 1, size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_h5 = tf.nn.relu(tf.matmul(G_h4, G_W5) + G_b5)
    G_log_prob = tf.matmul(G_h5, G_W6) + G_b6
    G_prob = tf.nn.softmax(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit



    
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G)
'''
D_solver = tf.train.GradientDescentOptimizer(learning_rate=lrD).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.GradientDescentOptimizer(learning_rate=lrG).minimize(G_loss, var_list=theta_G)
'''


def train(w):
    global BATCH_SIZE,lr
    BATCH_SIZE=5000
    lr=0.0001
    psit=Loadd.loadd(w)
    for count in range(30000):
        psit[count]=psit[count]*psit[count]

    TRAIN_DATASIZE,_=psit.shape
    PERIOD = int(TRAIN_DATASIZE/BATCH_SIZE)
    answery=np.zeros([20,3000,numinG])
    cc=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCH):
            idxs = np.random.permutation(TRAIN_DATASIZE) #shuffled ordering
            X_random = psit[idxs]

            for i in range(PERIOD):
                batch_X = X_random[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_X, Z: sample_Z(BATCH_SIZE, numinG)})
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(BATCH_SIZE, numinG)})
            if e>1500:
                lr=0.00001
                BATCH_SIZE=30000
                PERIOD = int(TRAIN_DATASIZE/BATCH_SIZE)
            if e % 100== 0:
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
            if e % 200==0:
                answer,accuracy=sess.run([G_sample,D_real], feed_dict={X: psit,Z: sample_Z(10, numinG)})
                print('answer[0]:',answer[0][:5])
                print('answer[9]:',answer[9][:5])
                print('mean(accuracy):',np.mean(accuracy))
            if e > 2000:
                if (e+1)%50==0:
                    answery[cc]=sess.run(G_sample, feed_dict={X: psit,Z: sample_Z(3000, numinG)})
                    cc=cc+1
        accuracy=sess.run(D_real, feed_dict={X: psit,Z: sample_Z(5000, numinG)})
        
    output1 = open('answer_%s.pkl'%(w), 'wb')
    output2 = open('accuracy_%s.pkl'%(w), 'wb')


    pickle.dump(answery, output1)
    pickle.dump(accuracy, output2)


    output1.close()
    output2.close()

    return 'complete'
