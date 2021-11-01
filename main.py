from numpy.core.fromnumeric import shape
import tensorflow.compat.v1 as tf
import numpy as np
import os
from tensorflow.keras.datasets import mnist as data
from matplotlib import pyplot as plt
from optimizer import Gradient, ConOpt, CESP, ConOpt_with_CESP
tf.compat.v1.disable_eager_execution()
import time
import math

(X_train_old, y_train_old), (X_test, y_test) = data.load_data()

target1 = 0
target2 = 1
X_train = []
y_train = []  # it is useless, but you can make sure the target1 and target2 were saved

for i in range(X_train_old.shape[0]):
    if y_train_old[i] == target1 or y_train_old[i] == target2:
        X_train.append(X_train_old[i])
        y_train.append(y_train_old[i])

X_train = np.array(X_train)
if len(X_train.shape) == 3:
    X_train.resize((X_train.shape[0], X_train.shape[1],X_train.shape[2],1))
print(X_train.shape)
for i in range(len(y_train)):
    if y_train[i] != target1 and y_train[i]!= target2:
        print('It existed not {} or {}'.format(target1, target2), y_train[i])

IMG_ROW = X_train.shape[1]
IMG_COLUMN = X_train.shape[2]
IMG_CHANNEL = X_train.shape[3]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
class GAN_Model(object):
    def __init__(self, batch_size=128, 
                    unit_size=100, 
                    lr=0.005, 
                    epoch=2000, 
                    gamma=0.1,              # learning rate of consensus optimization
                    alpha=0.5,              # learning rate of negative curvature
                    img_shape=(28, 28, 1)):
        self.batch_size = batch_size
        self.unit_size = unit_size
        self.learning_rate = lr
        self.epoch = epoch
        self.gamma = gamma
        self.alpha = alpha
        self.img_shape = img_shape
        self.row = img_shape[0]
        self.column = img_shape[1]
        self.channel = img_shape[2]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # generative model
    def generatorModel(self, noise_img, out_size, lrelu_lr=0.2):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            FC = tf.layers.dense(noise_img, self.unit_size)
            reLu = tf.nn.leaky_relu(FC, lrelu_lr)
            logits = tf.layers.dense(reLu, out_size)
            outputs = tf.tanh(logits)
            return logits, outputs

    # discriminator model
    def discriminatorModel(self, images, lrelu_lr=0.2, reuse=False):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            FC = tf.layers.dense(images, self.unit_size)
            reLu = tf.nn.leaky_relu(FC, lrelu_lr)
            logits = tf.layers.dense(reLu, 1)
            outputs = tf.sigmoid(logits)
            return logits, outputs

    # loss function
    """
    The goal of discriminator is:
    1. For real image, we will expect D return 1 as the label
    2. For fake image, we will expect D return 0 as the label
    The goal of generator is: For generative image, G will hope D return 1 as the label
    """
    def loss_function(self, real_logits, fake_logits):
        # generator hope discriminator return 1
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                        labels=tf.ones_like(fake_logits)))
        # Given discriminator a fake image, D will hope return 0
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                        labels=tf.zeros_like(fake_logits)))
        # Given discriminator a real image, D will hope return 1
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                        labels=tf.ones_like(real_logits)))
        # The loss of discriminator
        D_loss = tf.add(fake_loss, real_loss)
        return G_loss, fake_loss, real_loss, D_loss


    # Train
    def train(self):
        optimizer = ['Gradient', 'ConOpt', 'CESP', 'ConOpt_with_CESP']              # All optimizer
        image_size = self.X_train.shape[1]                  #self.mnist.train.images[0].shape[0]            # 784
        real_images = tf.placeholder(tf.float32, [None, image_size])
        fake_images = tf.placeholder(tf.float32, [None, image_size])

        # Generate a fake image as G_output
        G_logits, G_output = self.generatorModel(fake_images, image_size)
        # discriminative real image
        real_logits, real_output = self.discriminatorModel(real_images)
        # discriminative fake image
        fake_logits, fake_output = self.discriminatorModel(G_output, reuse=True)
        # compute the loss function
        G_loss, real_loss, fake_loss, D_loss = self.loss_function(real_logits, fake_logits)
        times = []
        for opt in optimizer:
            # Start training
            start = time.time()

            D_loss_list = []
            G_loss_list = []
            d_grads_list = []
            g_grads_list = []
            nc_d_list = []
            nc_g_list = []
            eigen_d_list = []
            eigen_g_list = []
            try:
                os.makedirs('image_{}'.format(opt))
            except FileExistsError:
                print('existed')
            # Optimizer
            if opt =='Gradient':
                G_optimizer, D_optimizer, d_grads, g_grads, e_d_max, e_g_min = Gradient(G_loss, D_loss, self.learning_rate)
            elif opt == 'ConOpt':
                G_optimizer, D_optimizer, d_grads, g_grads, e_d_max, e_g_min = ConOpt(G_loss, D_loss, self.learning_rate, self.gamma)
            elif opt == 'CESP':
                G_optimizer, D_optimizer, d_grads, g_grads, nc_step_d, nc_step_g, e_d_max, e_g_min = CESP(G_loss, D_loss, self.learning_rate, self.alpha)
            elif opt == 'ConOpt_with_CESP':
                G_optimizer, D_optimizer, d_grads, g_grads, nc_step_d, nc_step_g, e_d_max, e_g_min = ConOpt_with_CESP(G_loss, D_loss, self.learning_rate, self.gamma, self.alpha)
            #saver = tf.train.Saver()
            step = 0
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for epoch in range(self.epoch+1):
                    for batch_i in range(self.X_train.shape[0]//self.batch_size): #self.mnist.train.num_examples // self.batch_size):
                        batch_image = self.X_train[0+batch_i*self.batch_size:self.batch_size*(batch_i+1)]#self.mnist.train.next_batch(self.batch_size)
                        # scale to (-1,1) for all batch image since generator is used tanh fucntion
                        batch_image = batch_image * 2 -1
                        # noise of generative model
                        noise_image = np.random.uniform(-1, 1, size=(self.batch_size, image_size))
                        # run optimizer
                        session.run(G_optimizer, feed_dict={fake_images:noise_image})
                        session.run(D_optimizer, feed_dict={real_images: batch_image, fake_images: noise_image})
                        step = step + 1
                    # loss function of D 
                    loss_D = session.run(D_loss, feed_dict={real_images: batch_image, fake_images:noise_image})
                    # loss of D for real image
                    loss_real =session.run(real_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
                    # loss of D for fake image
                    loss_fake = session.run(fake_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
                    # loss of G
                    loss_G = session.run(G_loss, feed_dict={fake_images: noise_image})

                    D_loss_list.append(loss_D)
                    G_loss_list.append(loss_G)
                    if opt == 'CESP' or opt == 'ConOpt_with_CESP':         
                        d_grad = session.run(d_grads, feed_dict = {real_images: batch_image, fake_images: noise_image})
                        g_grad = session.run(g_grads, feed_dict = {fake_images: noise_image})
                        nc_g = session.run(nc_step_g, feed_dict = {fake_images: noise_image}) 
                        nc_d = session.run(nc_step_d, feed_dict = {real_images: batch_image, fake_images: noise_image})
                        eigen_max_d = session.run(e_d_max, feed_dict = {real_images: batch_image, fake_images: noise_image})
                        eigen_min_g = session.run(e_g_min, feed_dict = {fake_images:noise_image})
                        d_grads_list.append(np.linalg.norm(d_grad[3], 2))
                        g_grads_list.append(np.linalg.norm(g_grad[3], 2))
                        nc_d_list.append(np.linalg.norm(nc_d[3], 2))
                        nc_g_list.append(np.linalg.norm(nc_g[3], 2))
                        eigen_d_list.append(eigen_max_d)
                        eigen_g_list.append(eigen_min_g)
                    elif opt == 'Gradient' or opt == 'ConOpt':
                        d_grad = session.run(d_grads, feed_dict = {real_images: batch_image, fake_images: noise_image})
                        g_grad = session.run(g_grads, feed_dict = {fake_images: noise_image})
                        eigen_max_d = session.run(e_d_max, feed_dict = {real_images: batch_image, fake_images: noise_image})
                        eigen_min_g = session.run(e_g_min, feed_dict = {fake_images:noise_image})
                        d_grads_list.append(np.linalg.norm(d_grad[3], 2))
                        g_grads_list.append(np.linalg.norm(g_grad[3], 2))
                        eigen_d_list.append(eigen_max_d)
                        eigen_g_list.append(eigen_min_g)

                    print('epoch:', epoch, 'loss_D:', loss_D, ' loss_real', loss_real, ' loss_fake', loss_fake, ' loss_G', loss_G)
                    #model_path = os.getcwd() + os.sep + "mnist.model"
                    #saver.save(session, model_path, global_step=step)
                    if (epoch) % 1000 == 0:              # generate a picture for every 100 epoch
                        gen_imgs = session.run(G_output, feed_dict={fake_images: noise_image})
                        gen_imgs = gen_imgs.reshape((self.batch_size, self.column, self.row, self.channel))
                        """
                        plt.imshow(gen_imgs[0,:,:,0], cmap='gray')
                        plt.title('{} epoch'.format(epoch))
                        plt.savefig('/content/gdrive/MyDrive/paper_idea/image_{}/mnist_{}.png'.format(opt, epoch))
                        plt.close()
                        """
                        fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
                        ax = ax.flatten()
                        for j in range(25):
                            ax[j].imshow(gen_imgs[j,:,:,0], cmap='gray_r') 
                        plt.savefig('./image_{}/mnist_{}.png'.format(opt, epoch))
                        plt.close()
                        
                plt.figure(1)
                self.plot_list(D_loss_list, 'discriminator loss', opt)
                plt.figure(2)
                self.plot_list(G_loss_list, 'generator loss', opt)
                plt.figure(3)
                self.plot_list(d_grads_list, 'gradient of Discriminator', opt)
                plt.figure(4)
                self.plot_list(g_grads_list, 'gradient of Generator', opt)
                plt.figure(5)
                self.plot_list(eigen_d_list, 'max eigenvalue of Discriminator', opt)
                plt.figure(6)
                self.plot_list(eigen_g_list, 'min eigenvalue of Generator', opt)
                if opt == 'CESP' or opt == 'ConOpt_with_CESP':
                    plt.figure(7)
                    self.plot_list(nc_d_list, 'negative curvature of Discriminator', opt)
                    plt.figure(8)
                    self.plot_list(nc_g_list, 'negative curvature of Generator', opt)
        
            # end of training one opt
            end = time.time()

            # training time
            train_time = end - start

            days = train_time/60/60/24 
            hour = (days - math.floor(days))*24
            mini = (hour - math.floor(hour))*60
            sec = (mini - math.floor(mini))*60

            times.append([math.floor(days), math.floor(hour), math.floor(mini), round(sec)])

        #### Total trained time

        print('Gradient', times[0][0], 'days', times[0][1], 'hour', times[0][2], 'minute', times[0][3], 'sec')
        print('ConOpt', times[1][0], 'days', times[1][1], 'hour', times[1][2], 'minute', times[1][3], 'sec')
        print('CESP', times[2][0], 'days', times[2][1], 'hour', times[2][2], 'minute', times[2][3], 'sec')
        print('ConOpt_with_CESP', times[3][0], 'days', times[3][1], 'hour', times[3][2], 'minute', times[3][3], 'sec')    

    def plot_list(self, target_list, name, opt):
        plt.title(name)
        plt.plot([i for i in range(self.epoch+1)], target_list)
        plt.savefig('image_{}/{}_{}.png'.format(opt, name, opt))
        plt.close()

def main(argv=None):
    gan = GAN_Model(epoch=30000, img_shape=(IMG_ROW,IMG_COLUMN,IMG_CHANNEL))
    gan.train()

if __name__ == '__main__':
    tf.app.run()