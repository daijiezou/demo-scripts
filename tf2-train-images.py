import datetime
import os

import tensorflow as tf
import pathlib
import time
import argparse
from tensorflow import keras

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument("--mode", default="train", help="train or test")
parser.add_argument('--model', default='mobileNet',
                    help='mobileNet or vgg19 or  resNet50 or resNet152 or denseNet121 or nasNetLarge ')
parser.add_argument("--num_epochs", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--data_dir", default="/Users/daijun/PycharmProjects/pythonProject1/testdata/mydata1")
parser.add_argument("--train_dir", default="./model")

args = parser.parse_args()
gpus = tf.config.list_physical_devices(device_type='GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.MirroredStrategy()


def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image, label


def main():
    data_root = pathlib.Path(args.data_dir)
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    image_count = len(all_image_paths)

    # 确定每张图片的标签
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    classes = len(label_names)
    # 为每个标签分配索引：
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    # 创建一个列表，包含每个文件的标签索引：
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()  # 算出step的真实数量
    # cache1 = image_label_ds.cache()
    ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10))
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    print("=====================================")
    print("imageTotalCount:", image_count)
    print("steps_per_epoch:", steps_per_epoch)
    print("batch_size:", BATCH_SIZE)
    print("epoch:", args.num_epochs)
    print("classes:", classes)
    print("gpus:", gpus)
    print("=====================================")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.train_dir + "/" + current_time
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    with strategy.scope():

        if args.model == "resNet50":
            model = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=[224, 224, 3], classes=classes)
        elif args.model == "vgg19":
            model = tf.keras.applications.vgg19.VGG19(weights=None, input_shape=[224, 224, 3], classes=classes)
        elif args.model == "resNet152":
            model = tf.keras.applications.ResNet152(weights=None, input_shape=[224, 224, 3], classes=classes)
        elif args.model == "denseNet121":
            model = tf.keras.applications.DenseNet121(weights=None, input_shape=[224, 224, 3], classes=classes)
        elif args.model == "nasNetLarge":
            model = tf.keras.applications.NASNetLarge(weights=None, input_shape=[224, 224, 3], classes=classes)
        elif args.model == "mobileNet":
            model = tf.keras.applications.MobileNet(weights=None, input_shape=[224, 224, 3], classes=classes)
        else:
            print("no  model")
            return

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.train_dir, checkpoint_name="model.ckpt",
                                         max_to_keep=2)
    if args.mode == "test":
        model.fit(ds, epochs=args.num_epochs, steps_per_epoch=2, callbacks=[tensorboard_callback])
        manager.save()
    else:
        model.fit(ds, epochs=args.num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
        manager.save()

    print("==========train end============")


if __name__ == '__main__':
    start = time.time()
    main()
    costTime = time.time() - start
    costTime = int(costTime)
    print("costTime:%dmin%ds" % (costTime // 60, costTime % 60))
