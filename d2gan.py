
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
numG2=175
numG3=150
numG4=150
numG5=125
numG6=100
EPOCH=6000
alpha=1.
beta=1.

BATCH_SIZE=6000
lr=0.0001
X = tf.placeholder(tf.float32, shape=[None, numinD])
D1_W1 = tf.Variable(tf.contrib.layers.xavier_initializer()([numinD, numD1]))
D1_b1 = tf.Variable(tf.zeros(shape=[numD1]))

D1_W2 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD1, numD2]))
D1_b2 = tf.Variable(tf.zeros(shape=[numD2]))

D1_W3 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD2, numD3]))
D1_b3 = tf.Variable(tf.zeros(shape=[numD3]))
theta_D1 = [D1_W1, D1_W2, D1_W3, D1_b1, D1_b2, D1_b3]

D2_W1 = tf.Variable(tf.contrib.layers.xavier_initializer()([numinD, numD1]))
D2_b1 = tf.Variable(tf.zeros(shape=[numD1]))

D2_W2 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD1, numD2]))
D2_b2 = tf.Variable(tf.zeros(shape=[numD2]))

D2_W3 = tf.Variable(tf.contrib.layers.xavier_initializer()([numD2, numD3]))
D2_b3 = tf.Variable(tf.zeros(shape=[numD3]))
theta_D2 = [D2_W1, D2_W2, D2_W3, D2_b1, D2_b2, D2_b3]

theta_D=theta_D1+theta_D2

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
    return np.random.normal(0, 0.5, size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_h5 = tf.nn.relu(tf.matmul(G_h4, G_W5) + G_b5)
    G_log_prob = tf.matmul(G_h5, G_W6) + G_b6
    G_prob = tf.nn.softmax(G_log_prob)

    return G_prob


def discriminator1(x):
    D1_h1 = tf.nn.leaky_relu(tf.matmul(x, D1_W1) + D1_b1)
    D1_h2 = tf.nn.leaky_relu(tf.matmul(D1_h1, D1_W2) + D1_b2)
    D1_logit = tf.matmul(D1_h2, D1_W3) + D1_b3
    D1_prob = tf.nn.softplus(D1_logit)

    return D1_prob

def discriminator2(x):
    D2_h1 = tf.nn.leaky_relu(tf.matmul(x, D2_W1) + D2_b1)
    D2_h2 = tf.nn.leaky_relu(tf.matmul(D2_h1, D2_W2) + D2_b2)
    D2_logit = tf.matmul(D2_h2, D2_W3) + D2_b3
    D2_prob = tf.nn.softplus(D2_logit)

    return D2_prob


    
G_sample = generator(Z)



D1_loss_real = tf.reduce_mean(tf.log(discriminator1(X)))
D1_loss_fake = tf.reduce_mean(discriminator1(G_sample))
D1_loss = -alpha*D1_loss_real + D1_loss_fake

D2_loss_real = tf.reduce_mean(discriminator2(X))
D2_loss_fake = tf.reduce_mean(tf.log(discriminator2(G_sample)))
D2_loss= D2_loss_real-beta*D2_loss_fake
D_loss=D1_loss+D2_loss
                              
G_loss = beta*tf.reduce_mean(tf.log(discriminator2(G_sample)))+tf.reduce_mean(-discriminator1(G_sample))
                        
D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G)


def train(w):
    global BATCH_SIZE,lr
    BATCH_SIZE=6000
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
            '''
            if e>1500:
                lr=0.00001
                BATCH_SIZE=30000
                PERIOD = int(TRAIN_DATASIZE/BATCH_SIZE)
            '''
            if e % 10== 0:
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
            if e % 10==0:
                answer=sess.run(G_sample, feed_dict={X: psit,Z: sample_Z(10, numinG)})
                print('answer[0]:',answer[0][:5])
                print('answer[9]:',answer[9][:5])
                #print('mean(accuracy):',np.mean(accuracy))
            if e > 5000:
                if (e+1)%50==0:
                    answery[cc]=sess.run(G_sample, feed_dict={X: psit,Z: sample_Z(3000, numinG)})
                    cc=cc+1
        #accuracy=sess.run(D_real, feed_dict={X: psit,Z: sample_Z(5000, numinG)})
        
    output1 = open('answer_%s.pkl'%(w), 'wb')
    output2 = open('accuracy_%s.pkl'%(w), 'wb')


    pickle.dump(answery, output1)
    pickle.dump(accuracy, output2)


    output1.close()
    output2.close()

    return 'complete'
