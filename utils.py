#!/usr/bin/env python

import sys
import re
import os
import numpy as np
import tensorflow as tf
import argparse

def get_command_line_args(Config):
    ap = argparse.ArgumentParser()

    ap.add_argument("--weight_decay", type=float, nargs=1, default=None)
    ap.add_argument("--max_grad_norm", type=float, nargs=1, default=None)
    ap.add_argument("--drop_i", type=float, nargs=1, default=None)
    ap.add_argument("--drop_h", type=float, nargs=1, default=None)
    ap.add_argument("--drop_o", type=float, nargs=1, default=None)
    ap.add_argument("--hidden_size", type=int, nargs=1, default=None)
    ap.add_argument("--mask", type=float, nargs=1, default=None)
    ap.add_argument("--num_steps", type=int, nargs=1, default=None)
    ap.add_argument("--init_scale", type=float, nargs=1, default=None)
    ap.add_argument("--state_gate", type=bool, nargs=1, default=None)
    ap.add_argument("--init_bias", type=float, nargs=1, default=None)
    ap.add_argument("--num_layers", type=int, nargs=1, default=None)
    ap.add_argument("--depth", type=int, nargs=1, default=None)
    ap.add_argument("--out_size", type=int, nargs=1, default=None)
    ap.add_argument("--adaptive_optimizer", type=str, nargs=1, default=None)
    ap.add_argument("--reset_weights_flag", type=bool, nargs=1, default=None)
    ap.add_argument("--start_time", type=int, nargs=1, default=None)
    ap.add_argument("--wind_step_size", type=int, nargs=1, default=None)
    ap.add_argument("--switch_to_asgd", type=int, nargs=1, default=None)
    ap.add_argument("--decay_epochs", type=int, nargs='*', default=None)
    ap.add_argument("--learning_rate", type=float, nargs=1, default=None)
    ap.add_argument("--lr_decay", type=float, nargs='*', default=None)
    ap.add_argument("--max_max_epoch", type=int, nargs=1, default=None)
    ap.add_argument("--DB_name", type=str, nargs=1, default=None)
    ap.add_argument("--concat_tar_2_feat", type=bool, nargs=1, default=None)
    ap.add_argument("--random", type=bool, nargs=1, default=True)
    ap.add_argument("--server", type=bool, nargs=1, default=False)
    ap.add_argument("--gpu", type=int, nargs='*', default=-1)
    ap.add_argument("--num_of_proc", type=int, nargs=1, default=1)

    args = ap.parse_args()
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr is not None:
            if type(attr) == list:
                if hasattr(Config, arg):
                    if type(getattr(Config, arg)) == list:
                        setattr(Config, arg, attr)
                    elif len(attr) == 1:
                        setattr(Config, arg, attr[0])
                    else:
                        print('you enter list where its not suppose to be list.. exiting')
                        print('you enter ' + str(attr))
                        print('original ' + str(getattr(Config, arg)))
                        sys.exit()
                elif len(attr) == 1:
                    setattr(Config, arg, attr[0])
                else:
                    setattr(Config, arg, attr)
            else:
                setattr(Config, arg, attr)

# def isRandom():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--random", type=bool, nargs=1, default=True)
#     args = ap.parse_args()
#     return args.random

def get_relevant_features_idx(arg):
    if arg == "all":
        return np.array([i for i in range(385)])
    elif arg == "opt_a":
        l1 = np.array([i for i in range(132)])
        l2 = np.array([i for i in range(289,293)])
        return np.concatenate([l1,l2])
    elif arg == "opt_b":
        return np.array([i for i in range(132,385)])
    elif arg == "old":
        return np.array([i for i in range(179)])
    else:
        print("wrong argument for \'get_relevant_features_idx\'!! exiting")
        exit()


def get_scores_mask(y, config):
    return np.array(abs(y) > config.mask, dtype=np.float32)



def get_noise(m, drop_i, drop_h, drop_o):
    keep_i, keep_h, keep_o = 1.0 - drop_i, 1.0 - drop_h, 1.0 - drop_o
    if keep_i < 1.0:
        noise_i = (np.random.random_sample((m.batch_size, m.in_size, m.num_layers)) < keep_i).astype(np.float32) / keep_i
    else:
        noise_i = np.ones((m.batch_size, m.in_size, m.num_layers), dtype=np.float32)
    if keep_h < 1.0:
        noise_h = (np.random.random_sample((m.batch_size, m.size, m.num_layers)) < keep_h).astype(np.float32) / keep_h
    else:
        noise_h = np.ones((m.batch_size, m.size, m.num_layers), dtype=np.float32)
    if keep_o < 1.0:
        noise_o = (np.random.random_sample((m.batch_size, 1, m.size)) < keep_o).astype(np.float32) / keep_o
    else:
        noise_o = np.ones((m.batch_size, 1, m.size), dtype=np.float32)
    return noise_i, noise_h, noise_o


def reset_optimizer(session, name):
    print("reseting optimizer " + name)
    optimizer_scope = [v for v in tf.global_variables() if name in v.name]
    session.run(tf.variables_initializer(optimizer_scope))

def reset_weights():
    print("reseting weights")
    tf.global_variables_initializer().run()


def get_gpu_device_list(args_gpu):
    if type(args_gpu) == int:
        return str(args_gpu)
    if type(args_gpu) == list:
        return [str(g) for g in args_gpu]

# def get_num_of_proc():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--num_of_proc", type=int, nargs=1, default=1)
#     args = ap.parse_args()
#     return args.num_of_proc

# def isServer():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--server", type=bool, nargs=1, default=False)
#     args = ap.parse_args()
#     print('')
#     print("#############################")
#     if args.server:
#         print("# running on server!!!")
#     else:
#         print("# NOT running on server!!!")
#     print("#############################")
#     print('')
#     return args.server

class Logger(object):
    def __init__(self, file_path_and_name):
        self.terminal = sys.stdout
        self.log_file_name = file_path_and_name
        log = open(self.log_file_name, "w")
        log.close()

    def write(self, message):
        self.terminal.write(message)
        log = open(self.log_file_name, "a")
        log.write(message)
        log.close()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def get_rand_seed():
    proc_id = os.getpid()
    num = np.random.randint(np.iinfo(np.int32).max)
    for i in range(proc_id):
        num = np.random.randint(np.iinfo(np.int32).max)
    return num


def targets_nan_to_num(targets):
    print("in targets_nan_to_num")
    idxes = np.array(np.where(np.isnan(targets[:, :, 0])))
    for i in range(idxes.shape[1]):
        targets[idxes[0, i], idxes[1, i], 0] = 2
        targets[idxes[0, i], idxes[1, i], 1] = 3
        targets[idxes[0, i], idxes[1, i], 2] = 0.0
    return targets

def features_nan_to_num(features):
    print("in features_nan_to_num")
    for i in range(features.shape[2]):
        features[:, :, i] = np.nan_to_num(features[:, :, i])
        if i%50 == 0:
            print("finished converting %d features" % (i+1))
    return features

def get_documentation(config, Config):
    simulation_name = 'gru__hid_size=' + str(config.hidden_size) + \
                      '__DB=' + config.DB_name + \
                      '__mask=' + str(config.mask).replace('.','_') + \
                      '__wind=' + str(config.wind_step_size)

    if config.state_gate:
        simulation_name = simulation_name + '_stg'

    if config.reset_weights_flag:
        simulation_name = simulation_name + '_Rst'

    if Config.random:
        simulation_name = simulation_name + '_Rnd'

    if Config.server:
        simulation_name = simulation_name + '_Srv'

    count = 0
    documentation_dir = os.path.join('./documentation/' + simulation_name)
    while os.path.isdir(documentation_dir + '__' + str(count)):
        count += 1
    simulation_name = simulation_name + '__' + str(count)

    documentation_dir = os.path.join('./documentation/' + simulation_name)
    os.makedirs(documentation_dir)
    sys.stdout = Logger(documentation_dir + "/logfile.log")
    os.makedirs(documentation_dir + '/saver')
    print('simulation is saved to %s' % documentation_dir)
    print("process id = " + str(os.getpid()))

    # documentation of configurations
    print('')
    print('')
    print("######################################  CONFIGURATIONS  ######################################")
    text_file = open(documentation_dir + "/configurations.txt", "w")
    for attr, value in sorted(vars(Config).items()):
        if str(attr).startswith("__"): continue
        line = str(attr) + '=' + str(value)
        print("# " + line)
        text_file.write(line + '\n')
    text_file.close()
    print("##############################################################################################")
    print('')
    print('')
    return simulation_name, documentation_dir

def get_sess_config(config):
    if config.server:
        sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                     inter_op_parallelism_threads=2,
                                     intra_op_parallelism_threads=8)
        sess_config.gpu_options.visible_device_list = get_gpu_device_list(config.gpu)
    else:
        sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True
    return sess_config

def train_windows_producer(config, max_time):
    print("generationg training windows:")
    end_list = [config.start_time]
    start_list = [0]
    while (end_list[-1] + config.wind_step_size) < max_time:
        print(start_list[-1], end_list[-1])
        end_list.append(end_list[-1] + config.wind_step_size)
        start_list.append(0)
    print(start_list[-1], end_list[-1])
    return zip(start_list, end_list)