#超参数设定
class Config:
    #每次梯度更新所用的样本数量
    batch_size = 64

    lr = 1e-3  #学习率
    # 动量项，加速 SGD 收敛并减少震荡，代码实际使用 Adam 优化器，此参数未被使用（可能冗余）
    momentum = 0.9

    # L2 正则化系数，防止过拟合
    weights_decay = 1e-4
    # 分类类别数（0-9 共 10 个数字 + 1 个空白符）
    class_num = 11
    # 每训练 1 个 epoch 后，在验证集上评估模型性能。
    eval_interval = 1
    # 每 1 个 epoch 保存一次模型权重
    checkpoint_interval = 1
    # 每 50 个 batch 打印一次训练损失和准确率,并使用余弦退火更新调整学习率
    print_interval = 50
    #模型权重保存路径
    checkpoints = './user_data/model_data/checkpoints/'  #这个需要自己创建一个文件夹用来储存权重

    #预训练模型路径，用于加载已有权重继续训练。
    pretrained = None #'/user_data/model_data/checkpoints/epoch-resnet18-52-bn-acc-73.86.pth'

    start_epoch = 0  #从预训练模型继续训练时的起始 epoch（避免从 0 开始计数）

    epoches = 80  #总训练轮数（可能提前停止，但上限为 100）
    # 标签平滑系数，缓解过拟合
    smooth = 0.1
    # 随机擦除（Random Erasing）概率，增强数据多样性
    erase_prob = 0.5
