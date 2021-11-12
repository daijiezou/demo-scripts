# gemini-demo-script


## 快速开始（TensorFlow2）
所需数据集在已经在仓库中

启动命令：`python train_minst.py`

## TF2
### 参数说明：
num_epochs:对数据集进行多少次训练（可根据此参数调整任务训练时间）
batch_size:每次读取多少张图片(根据显存大小调节,图片大小调节)
data_dir:数据集所在路径
train_dir：生成模型的存储路径
model：指定训练的模型（mobileNet、vgg19、resNet50、resNet152、denseNet121）


### 启动命令
`python3.6 $GEMINI_RUN/tf2-train-images.py  --mode train --model resNet50 --num_epochs 3 --batch_size 128 --data_dir $GEMINI_DATA_IN1/testData --train_dir $GEMINI_DATA_OUT`

## Pytorch
### 参数说明
epochs：对数据集进行多少次训练
batch-size：每次读取多少张图片
world-size：训练所需节点数(如两节点的分布式任务--world-size 2 )
### 启动命令
`python3.6 -u $GEMINI_RUN/torch1.4.py -a resnet18 --batch-size 64 --epochs 3 --dist-url "tcp://$GEMINI_HOST_IP_taskrole1_0:$GEMINI_taskrole1_0_http_PORT" --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank $GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX $GEMINI_DATA_IN1/ImageNet_ILSVRC2012_3G`