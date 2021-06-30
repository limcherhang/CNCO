import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from optimizer import Gradient, ConOpt, CESP, ConOpt_with_CESP

BATCH_SIZE = 1000
UNITS_SIZE = 100
LEARNING_RATE = 0.005
EPOCH = 10000
SMOOTH = 0.1
GAMMA = 0.1     # learning rate of consensus optimization
ALPHA = 0.01     # learning rate of negative curvature
IMG_ROW = 28
IMG_COLUMN = 28
IMG_CHANNEL = 1

mnist = input_data.read_data_sets('/mnist_data/', one_hot=True)

# generative model
def generatorModel(noise_img, units_size, out_size, alpha=0.01):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        FC = tf.layers.dense(noise_img, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
        drop = tf.layers.dropout(reLu, rate=0.2)
        logits = tf.layers.dense(drop, out_size)
        outputs = tf.tanh(logits)
        return logits, outputs

# discriminator model
def discriminatorModel(images, units_size, alpha=0.01, reuse=False):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        FC = tf.layers.dense(images, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
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
def loss_function(real_logits, fake_logits, smooth):
    # generator hope discriminator return 1
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                    labels=tf.ones_like(fake_logits)*(1-smooth)))
    # Given discriminator a fake image, D will hope return 0
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                       labels=tf.zeros_like(fake_logits)))
    # Given discriminator a real image, D will hope return 1
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                       labels=tf.ones_like(real_logits)*(1-smooth)))
    # The loss of discriminator
    D_loss = tf.add(fake_loss, real_loss)
    return G_loss, fake_loss, real_loss, D_loss


# Train
def train(mnist):
    optimizer = ['Gradient', 'ConOpt', 'CESP', 'ConOpt_with_CESP']           # All optimizer
    image_size = mnist.train.images[0].shape[0] #784
    real_images = tf.placeholder(tf.float32, [None, image_size])
    fake_images = tf.placeholder(tf.float32, [None, image_size])

    # Generate a fake image as G_output
    G_logits, G_output = generatorModel(fake_images, UNITS_SIZE, image_size)
    # discriminative real image
    real_logits, real_output = discriminatorModel(real_images, UNITS_SIZE)
    # discriminative fake image
    fake_logits, fake_output = discriminatorModel(G_output, UNITS_SIZE, reuse=True)
    # compute the loss function
    G_loss, real_loss, fake_loss, D_loss = loss_function(real_logits, fake_logits, SMOOTH)
    for opt in optimizer:
        D_loss_list = []
        G_loss_list = []
        try:
            os.makedirs('image_{}'.format(opt))
        except FileExistsError:
            print('existed')
        # Optimizer
        if opt =='Gradient':
            G_optimizer, D_optimizer = Gradient(G_loss, D_loss, LEARNING_RATE)
        elif opt == 'ConOpt':
            G_optimizer, D_optimizer = ConOpt(G_loss, D_loss, LEARNING_RATE, GAMMA)
        elif opt == 'CESP':
            G_optimizer, D_optimizer = CESP(G_loss, D_loss, LEARNING_RATE, ALPHA)
        elif opt == 'ConOpt_with_CESP':
            G_optimizer, D_optimizer = ConOpt_with_CESP(G_loss, D_loss, LEARNING_RATE, GAMMA, ALPHA)
        #saver = tf.train.Saver()
        step = 0
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(EPOCH+1):
                for batch_i in range(mnist.train.num_examples // BATCH_SIZE):
                    batch_image, _ = mnist.train.next_batch(BATCH_SIZE)
                    # scale to (-1,1) for all batch image since generator is used tanh fucntion
                    batch_image = batch_image * 2 -1
                    # noise of generative model
                    noise_image = np.random.uniform(-1, 1, size=(BATCH_SIZE, image_size))
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

                print('epoch:', epoch, 'loss_D:', loss_D, ' loss_real', loss_real, ' loss_fake', loss_fake, ' loss_G', loss_G)
                #model_path = os.getcwd() + os.sep + "mnist.model"
                #saver.save(session, model_path, global_step=step)
                if (epoch) % 100 == 0:              # generate a picture for every 100 epoch
                    gen_imgs = session.run(G_output, feed_dict={fake_images: noise_image})
                    gen_imgs = gen_imgs.reshape((BATCH_SIZE, IMG_COLUMN, IMG_ROW, IMG_CHANNEL))
                    plt.imshow(gen_imgs[0,:,:,0], cmap='gray')
                    plt.savefig('image_{}/mnist_{}.png'.format(opt, epoch))
                    plt.close()
                    
            plt.figure(1)
            plot(EPOCH+1, D_loss_list, 'discriminator loss', opt)
            plt.figure(2)
            plot(EPOCH+1, G_loss_list, 'generator loss', opt)

def plot(epochs, loss, loss_name, opt):
    plt.title(loss_name)
    plt.plot([i for i in range(EPOCH+1)], loss)
    plt.savefig('image_{}/{}.png'.format(opt, loss_name))

def main(argv=None):
    train(mnist)

if __name__ == '__main__':
    tf.app.run()