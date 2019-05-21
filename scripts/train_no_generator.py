# -*-coding:utf-8-*-

# -*-coding:utf-8-*-
"""
该脚本用于大内存用户训练模型
64g一下内存不要使用
"""
from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from config import PATIENCE, TRAIN_SAMPLES, BATCH_SIZE, VALID_SAMPLES, EPOCH
from utils import get_available_gpus, get_available_cpus
from model import build_model
from data_generator import data_all


def parse_command_params():
    """
    命令行参数解析器
    :return:
    """
    ap = ArgumentParser()  # 创建解析器
    ap.add_argument('-p', '--pretrained', help="path of your model files expected .h5 or .hdf5")
    ap.add_argument('-s', '--show', default='no', help="if you want to visualize training process")
    args_ = vars(ap.parse_args())
    return args_


def get_model(pretrained):
    """
    按照机器配置构建合理模型
    :return:
    """
    if get_available_gpus() > 1:
        model = build_model()
        if pretrained:
            my_model.load_weights(pretrained)
        else:
            pass
        model = multi_gpu_model(model, get_available_gpus())

    else:
        model = build_model()
        if pretrained:
            model.load_weights(pretrained)

    return model


def get_callbacks():
    """
    设置一些回调
    :return:
    """
    model_checkpoint = ModelCheckpoint('../models/training_best_model.h5',
                                       monitor='val_loss', verbose=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.1, patience=PATIENCE//4, verbose=True)

    callbacks_ = [model_checkpoint, early_stopping, reduce_lr]
    return callbacks_


if __name__ == '__main__':
    args = parse_command_params()
    my_model = get_model(args['pretrained'])

    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True, clipnorm=0.5)
    verbose = True if args['show'] == 'yes' else False
    callbacks = get_callbacks()

    my_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    x, y = data_all()
    my_model.fit(x, y, batch_size=64, epochs=EPOCH, verbose=verbose, callbacks=callbacks, )
    my_model.save_weights('../models/my_model_weights.h5')








