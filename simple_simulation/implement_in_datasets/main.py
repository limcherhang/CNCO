import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from optimizer import basic, conopt

BATCH_SIZE = 1000
UNITS_SIZE = 100
LEARNING_RATE = 0.0001
EPOCH = 150000
SMOOTH = 0.1
GAMMA = 0.1     # learning rate of consensus optimization
ALPHA = 0.1     # learning rate of negative curvature
IMG_ROW = 28
IMG_COLUMN = 28
IMG_CHANNEL = 1

mnist = input_data.read_data_sets('/mnist_data/', one_hot=True)

# 生成模型
def generatorModel(noise_img, units_size, out_size, alpha=0.01):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        FC = tf.layers.dense(noise_img, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
        drop = tf.layers.dropout(reLu, rate=0.2)
        logits = tf.layers.dense(drop, out_size)
        outputs = tf.tanh(logits)
        return logits, outputs

# 判別模型
def discriminatorModel(images, units_size, alpha=0.01, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        FC = tf.layers.dense(images, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
        logits = tf.layers.dense(reLu, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs

# 損失函式
"""
判別器的目的是：
1. 對於真實圖片，D要為其打上標籤1
2. 對於生成圖片，D要為其打上標籤0
生成器的目的是：對於生成的圖片，G希望D打上標籤1
"""
def loss_function(real_logits, fake_logits, smooth):
    # 生成器希望判別器判別出來的標籤為1; tf.ones_like()建立一個將所有元素都設定為1的張量
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                    labels=tf.ones_like(fake_logits)*(1-smooth)))
    # 判別器識別生成器產出的圖片，希望識別出來的標籤為0
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                       labels=tf.zeros_like(fake_logits)))
    # 判別器判別真實圖片，希望判別出來的標籤為1
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                       labels=tf.ones_like(real_logits)*(1-smooth)))
    # 判別器總loss
    D_loss = tf.add(fake_loss, real_loss)
    return G_loss, fake_loss, real_loss, D_loss


# 訓練
def train(mnist):
    try:
        os.makedirs('image')
    except FileExistsError:
        print('existed')
    D_loss_list = []
    G_loss_list = []
    opt = 'basic'
    image_size = mnist.train.images[0].shape[0] #784
    real_images = tf.placeholder(tf.float32, [None, image_size])
    fake_images = tf.placeholder(tf.float32, [None, image_size])

    #呼叫生成模型生成影象G_output
    G_logits, G_output = generatorModel(fake_images, UNITS_SIZE, image_size)
    # D對真實影象的判別
    real_logits, real_output = discriminatorModel(real_images, UNITS_SIZE)
    # D對G生成影象的判別
    fake_logits, fake_output = discriminatorModel(G_output, UNITS_SIZE, reuse=True)
    # 計算損失函式
    G_loss, real_loss, fake_loss, D_loss = loss_function(real_logits, fake_logits, SMOOTH)
    # 優化
    if opt =='basic':
        G_optimizer, D_optimizer = basic(G_loss, D_loss, LEARNING_RATE)
    elif opt == 'conopt':
        G_optimizer, D_optimizer = conopt(G_loss, D_loss, LEARNING_RATE, GAMMA)
    #saver = tf.train.Saver()
    step = 0
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            for batch_i in range(mnist.train.num_examples // BATCH_SIZE):
                batch_image, _ = mnist.train.next_batch(BATCH_SIZE)
                # 對影象畫素進行scale，tanh的輸出結果為(-1,1)
                batch_image = batch_image * 2 -1
                # 生成模型的輸入噪聲
                noise_image = np.random.uniform(-1, 1, size=(BATCH_SIZE, image_size))
                #
                session.run(G_optimizer, feed_dict={fake_images:noise_image})
                session.run(D_optimizer, feed_dict={real_images: batch_image, fake_images: noise_image})
                step = step + 1
            # 判別器D的損失
            loss_D = session.run(D_loss, feed_dict={real_images: batch_image, fake_images:noise_image})
            # D對真實圖片
            loss_real =session.run(real_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
            # D對生成圖片
            loss_fake = session.run(fake_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
            # 生成模型G的損失
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
                plt.savefig('image/mnist_{}.png'.format(epoch))
                plt.close()
                
        plt.figure(1)
        plot(EPOCH, D_loss_list, 'discriminator loss')
        plt.figure(2)
        plot(EPOCH, G_loss_list, 'generator loss')

def plot(epochs, loss, loss_name):
    plt.title(loss_name)
    plt.plot([i for i in range(EPOCH)], loss)
    plt.savefig('{}.png'.format(loss_name))

def main(argv=None):
    train(mnist)

if __name__ == '__main__':
    tf.app.run()