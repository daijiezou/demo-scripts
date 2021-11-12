import datetime

import tensorflow as tf
import pathlib
import time
import argparse
import os
import json

from tensorflow import keras

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument("--num_epochs", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--data_dir", default="/data/dataset")
parser.add_argument("--train_dir", default="./my_model")
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--rank", default=0, type=int)

args = parser.parse_args()
AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.list_physical_devices(device_type='GPU')
num_workers = args.num_workers
workList = []
for i in range(num_workers):
    work = os.environ.get("GEMINI_HOST_IP_taskrole1_%d" % i) + ":" + os.environ.get(
        "GEMINI_taskrole1_%d_http_PORT" % i)
    workList.append(work)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': workList
    },
    'task': {'type': 'worker', 'index': args.rank}
})

print("TF_CONFIG:", os.environ.get("TF_CONFIG"))
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def main():
    data_root = pathlib.Path(args.data_dir)
    if len(gpus) != 0:
        BATCH_SIZE_PER_REPLICA = args.batch_size * len(gpus)
    else:
        BATCH_SIZE_PER_REPLICA = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * args.num_workers
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
    image_label_ds = ds.map(load_and_preprocess_from_path_label)

    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()  # 算出step的真实数量
    # cache1 = image_label_ds.cache()
    ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10))
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.train_dir + "/" + current_time
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    print("=====================================")
    print("imageTotalCount:", image_count)
    print("steps_per_epoch:", steps_per_epoch)
    print("batch_size:", BATCH_SIZE)
    print("epoch:", args.num_epochs)
    print("classes:", classes)
    print("gpus:", gpus)
    print("=====================================")
    with strategy.scope():
        model = tf.keras.applications.vgg19.VGG19(weights=None, input_shape=[224, 224, 3], classes=classes)
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory=args.train_dir, checkpoint_name="model.ckpt",
                                             max_to_keep=1)
        # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=args.train_dir)]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
    model.fit(ds, epochs=args.num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
    if args.rank == 0:
        manager.save()
    print("==========train end============")


if __name__ == '__main__':
    start = time.time()
    main()
    costTime = time.time() - start
    costTime = int(costTime)
    print("costTime:%dmin%ds" % (costTime // 60, costTime % 60))
