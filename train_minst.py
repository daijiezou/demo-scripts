import argparse
import datetime

import tensorflow as tf

from tensorflow import optimizers, metrics
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument("--num_epochs", default=2, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--data_dir", default="data/mnist.npz")
parser.add_argument("--train_dir", default="./model")

args = parser.parse_args()


def load_mnist(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def pre_process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = load_mnist(args.data_dir)
# x = tf.cast(x[0], dtype=tf.float32) / 255.
# x = tf.reshape(x, [-1, 28 * 28])
batchSize = args.batch_size
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(pre_process).shuffle(10000).batch(batchSize)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(pre_process).shuffle(10000).batch(batchSize)
db_iter = iter(db)
sample = next(db_iter)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b,784] => [b,256]
    layers.Dense(128, activation=tf.nn.relu),  # [b,256] => [b,128]
    layers.Dense(64, activation=tf.nn.relu),  # [b,128] => [b,64]
    layers.Dense(32, activation=tf.nn.relu),  # [b,64] => [b,32]
    layers.Dense(10)  # [b,32] => [b,10]
])
model.build(input_shape=[None, 28 * 28])  # build 可以在模型训练之前看到模型的各层之间的参数
model.summary()  # 输出模型的层数

optimizer = optimizers.Adam(learning_rate=1e-3)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = args.train_dir + "/" + current_time

summary_writer = tf.summary.create_file_writer(log_dir)
# get x from (x,y)
sample_img = next(iter(db))[0]
# get first image instance
sample_img = sample_img[0]

sample_img = tf.reshape(sample_img, [1, 28, 28, 1])

with summary_writer.as_default():
    tf.summary.image("Training sample:", sample_img, step=0)


def main():
    for epoch in range(args.num_epochs):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_pred=logits, y_true=y, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, "loss:", loss)
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss), step=step)
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]
        acc = total_correct / total_num
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(acc), step=step)
        print(epoch, "test acc:", acc)

    model.save(args.train_dir)


if __name__ == '__main__':
    main()
