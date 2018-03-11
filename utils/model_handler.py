# !/usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals
from preprocessing.data_handler import Places
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from skimage import io, transform, color
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch as t
import numpy as np
import model as models
import logging
import math
import copy
import os

logger = logging.getLogger()


class ModelHandler(object):
    def __init__(self, model_params):
        # self.model = getattr(models, model_params.get("model_name"))(**model_params).eval()
        self.model = getattr(models, model_params.get("model_name"))().eval()
        self.model_initial_param = copy.deepcopy(self.model.state_dict())
        # print self.model

    # for step 0: configure model
    def _load_pretrained_model(self, pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None:
            pretrained_dict = t.load(pretrained_model_path)
            model_dict = copy.deepcopy(self.model_initial_param)

            # 1. filter out unnecessary keys
            useful_pretrained_dict = {}
            unused_pretrained_dict = []
            for key, value in pretrained_dict.items():
                if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                    useful_pretrained_dict.update({key: value})
                else:
                    unused_pretrained_dict.append(key)
            print "Unused pretrained layer:", unused_pretrained_dict

            # 2. overwrite entries in the existing state dict
            model_dict.update(useful_pretrained_dict)

            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
        else:
            model_dict = copy.deepcopy(self.model_initial_param)
            self.model.load_state_dict(model_dict)

    def _load_pretrained_model_for_prototype_2(self, pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None:
            pretrained_dict = t.load(pretrained_model_path)
            pretrained_dict = {key.split("module.", 1)[-1]: value
                               for key, value in pretrained_dict["state_dict"].items()}
            model_dict = copy.deepcopy(self.model_initial_param)
            # for i in sorted(pretrained_dict.keys()):
            #     print i, pretrained_dict[i].shape
            # for k, v in pretrained_dict["state_dict"].items():
            #     print k.split("module.", 1)[-1], v.shape
            # for k in sorted(model_dict.keys()):
            #         print k, model_dict[k].shape

            # 1. filter out unnecessary keys
            useful_pretrained_dict = {}
            unused_pretrained_dict = []
            for key, value in pretrained_dict.items():
                if key == "conv1.weight":
                    tmp_value = value.cpu().numpy()
                    tmp_value = np.mean(tmp_value, axis=1)
                    tmp_value = np.expand_dims(tmp_value, 1)
                    mean_value = t.from_numpy(tmp_value).cuda()
                    useful_pretrained_dict.update({key: mean_value})
                else:
                    if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                        # print "pretrained layer"
                        useful_pretrained_dict.update({key: value})
                    else:
                        unused_pretrained_dict.append(key)
            print "Used pretrained layer:\n", len(useful_pretrained_dict.keys())
            print "Unused pretrained layer:\n", len(unused_pretrained_dict)

            # 2. overwrite entries in the existing state dict
            model_dict.update(useful_pretrained_dict)

            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
        else:
            model_dict = copy.deepcopy(self.model_initial_param)
            self.model.load_state_dict(model_dict)

    def _load_pretrained_model_for_prototype_3(self, pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None:
            pretrained_dict = t.load(pretrained_model_path)
            pretrained_dict = {key.split("module.", 1)[-1]: value
                               for key, value in pretrained_dict.items()}
            model_dict = copy.deepcopy(self.model_initial_param)
            # for i in sorted(pretrained_dict.keys()):
            #     print i, pretrained_dict[i].shape
            # for k, v in pretrained_dict["state_dict"].items():
            #     print k.split("module.", 1)[-1], v.shape
            # for k in sorted(model_dict.keys()):
            #         print k, model_dict[k].shape

            # 1. filter out unnecessary keys
            useful_pretrained_dict = {}
            unused_pretrained_dict = []
            for key, value in pretrained_dict.items():
                if key == "conv1.weight":
                    tmp_value = value.cpu().numpy()
                    tmp_value = np.mean(tmp_value, axis=1)
                    tmp_value = np.expand_dims(tmp_value, 1)
                    mean_value = t.from_numpy(tmp_value).cuda()
                    useful_pretrained_dict.update({key: mean_value})
                else:
                    if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                        # print "pretrained layer"
                        useful_pretrained_dict.update({key: value})
                    else:
                        unused_pretrained_dict.append(key)
            print "Used pretrained layer:\n", len(useful_pretrained_dict.keys())
            print "Unused pretrained layer:\n", len(unused_pretrained_dict)

            # 2. overwrite entries in the existing state dict
            model_dict.update(useful_pretrained_dict)

            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
        else:
            model_dict = copy.deepcopy(self.model_initial_param)
            self.model.load_state_dict(model_dict)

    @staticmethod
    def _transfer_data_to_gpu(var, visiable_gpu_device_num):
        var_class = str(var.__class__)
        if "model" in var_class and visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
            var = nn.DataParallel(var, device_ids=visiable_gpu_device_num)
            var.cuda()
        elif "torch.autograd.variable.Variable" in var_class:
            if visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
                return var.cuda()
            else:
                return var
        elif "torch" in var_class and "Tensor" in var_class:
            if visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
                return Variable(var.cuda())
            else:
                return Variable(var)
        else:
            raise TypeError("No matched var class for transferring to GPU.")

    # for step 3: criterion and optimizer
    @staticmethod
    def _get_loss_function(loss_function_name):
        return eval(loss_function_name)

    def _get_optimizer(self, optimizer_name, **kwargs):
        optimizer = eval(optimizer_name)
        if "Adam" in optimizer_name:
            optimizer = optimizer(self.model.parameters(),
                                  lr=float(kwargs.get("lr")),
                                  weight_decay=float(kwargs.get("weight_decay")))
        else:
            optimizer = optimizer(self.model.parameters(),
                                  momentum=kwargs.get("momentum"),
                                  lr=float(kwargs.get("lr")),
                                  weight_decay=float(kwargs.get("weight_decay")))

        return optimizer

    def _get_loss(self, score, target, criterion):
        loss = criterion(score, target)

        return loss

    def _get_accuracy(self, confusion_matrix, classes_num):
        cm_value = confusion_matrix.value()

        correct_sum = 0.0
        for i in range(classes_num):
            correct_sum += cm_value[i][i]
        if cm_value.sum() != 0:
            accuracy = 1.0 * correct_sum / cm_value.sum()
        else:
            accuracy = 0.0

        return accuracy

    def train(self, training_params):
        k_folder = training_params["k_folder"]
        for i in [0]:
            print i, "~~~~~~~~~~~~~~~~~~~~~~~~"
            tb_dir_i = os.path.join(training_params.get("tensorboard_dir"), str(i))
            os.system("rm -rf {}".format(tb_dir_i))
            if not os.path.exists(tb_dir_i):
                os.makedirs(tb_dir_i)

            writer = SummaryWriter(tb_dir_i)
            json_filepath = os.path.join(tb_dir_i, "all_scalars.json")

            # step 1: get dataloader
            image_dir = training_params.get("multi_thread_data_dir")
            print "Image Dir:", image_dir
            data_pattern = training_params.get("data_pattern")
            annotation_file_path = training_params.get("annotation_file_path")
            hard_example_mining_file_path = training_params.get("hard_example_mining_file_path")
            repeat_times = training_params.get("repeat_times")
            batch_size = training_params.get("batch_size")
            gpu_num = training_params.get("gpu_num")
            thread_num = training_params.get("thread_num")
            train_data = Places(image_dir, data_pattern, annotation_file_path,
                                is_training=True,
                                is_testing=False,
                                hard_example_mining_file_path=hard_example_mining_file_path,
                                repeat_times=repeat_times)
            # val_data = LungNodule(image_dir, data_pattern, annotation_file_path, is_training=False, is_testing=False)
            train_dataloader = DataLoader(train_data, batch_size * gpu_num, shuffle=True, num_workers=thread_num)

            # step 2: configure model and map to gpu
            pretrained_model_path = training_params.get("pretrained_model_path")
            unnecessary_keys = training_params.get("unnecessary_keys")
            if isinstance(pretrained_model_path, list) and \
                    len(unnecessary_keys) != 0 and isinstance(unnecessary_keys[0], list):
                pretrained_model_path = pretrained_model_path[i]
                unnecessary_keys = unnecessary_keys[i]
            print "Pretrained Model Path:", pretrained_model_path

            special_model_load_method = training_params.get("special_model_load_method")
            if special_model_load_method == 2:
                self._load_pretrained_model_for_prototype_2(pretrained_model_path=pretrained_model_path,
                                                            unnecessary_keys=unnecessary_keys)
            if special_model_load_method == 3:
                self._load_pretrained_model_for_prototype_2(pretrained_model_path=pretrained_model_path,
                                                            unnecessary_keys=unnecessary_keys)
            else:
                self._load_pretrained_model(pretrained_model_path=pretrained_model_path,
                                            unnecessary_keys=unnecessary_keys)

            gpu_num = training_params.get("gpu_num")
            gpu_device_num = training_params.get("gpu_device_num")
            if gpu_device_num is not None and len(gpu_device_num) == gpu_num:
                visiable_gpu_device_num = range(gpu_num)
            else:
                print "Gpu num is not the same as len of gpu_device_num."
                visiable_gpu_device_num = range(gpu_num)
            print "Visiable gpu number:", t.cuda.device_count(), ", numbers: ", gpu_device_num

            self._transfer_data_to_gpu(self.model, visiable_gpu_device_num=visiable_gpu_device_num)

            # step 3: criterion and optimizer
            loss_function_name = training_params.get("loss_function_name")
            criterion = self._get_loss_function(loss_function_name)

            optimizer_name = training_params.get("optimizer_name")
            momentum = training_params.get("momentum")
            lr = training_params.get("learning_rate")
            weight_decay = training_params.get("weight_decay")
            optimizer = self._get_optimizer(optimizer_name, momentum=momentum, lr=lr, weight_decay=weight_decay)

            # step 4: meters for measure network
            loss_meter = meter.AverageValueMeter()
            confusion_matrix = meter.ConfusionMeter(training_params.get("classes_num"))
            previous_loss = 1e100

            # train
            self.model.train()

            for epoch in range(training_params.get("max_epoch")):
                loss_meter.reset()
                confusion_matrix.reset()

                for ii, (input_data, target_data, _, _, _) in enumerate(train_dataloader):
                    input = self._transfer_data_to_gpu(input_data.float(), visiable_gpu_device_num=visiable_gpu_device_num)
                    target = self._transfer_data_to_gpu((target_data * 200.0 - 100.).float(), visiable_gpu_device_num=visiable_gpu_device_num)
                    optimizer.zero_grad()

                    # score = self.model(input)
                    score = nn.parallel.data_parallel(self.model, input, device_ids=visiable_gpu_device_num)

                    # print score, score.size()
                    total_score = score.view(score.size()[0], -1)
                    total_target = target.contiguous().view(target.size()[0], -1)

                    # print score
                    loss = self._get_loss(total_score, total_target, criterion)
                    loss.backward()
                    optimizer.step()

                    print "Epoch {}, Step {}, loss: {:.5f} \t\t\t\t\t Sample Num: {}".format(epoch, ii,
                                                                                             float(loss.data[0]),
                                                                                             input_data.shape[0])

                    # meters update and visualize
                    loss_meter.add(loss.data[0])

                    if ii % training_params.get("print_freq") == training_params.get("print_freq") - 1:
                        training_n_iter = epoch * (int(math.ceil(1.0 * len(train_data) / (batch_size * gpu_num)))) \
                                          + ii + 1
                        writer.add_scalars("data/loss", {"train": float(loss.data[0])}, training_n_iter)

                if (epoch + 1) % int(training_params.get("learning_rate_decay_step")) == 0:
                    lr = float(lr) * float(training_params.get("learning_rate_decay"))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                if epoch % training_params.get("save_frequency") == training_params.get("save_frequency") - 1:
                    name = os.path.join(training_params.get("model_dir"),
                                          "{}_{}.pth".format(training_params.get("save_name"), epoch + 1))
                    t.save(self.model.state_dict(), name)

                previous_loss = loss_meter.value()[0]

                writer.export_scalars_to_json(json_filepath)

    def test(self, testing_params):
        # _mean_then_sofemax
        model_accuracy = []
        k_folder = testing_params["k_folder"]
        for i in [0]:
            print i, "~~~~~~~~~~~~~~~~~~~~~~~~"

            # gpu config
            gpu_num = testing_params.get("gpu_num")
            gpu_device_num = testing_params.get("gpu_device_num")
            if gpu_device_num is not None and len(gpu_device_num) == gpu_num:
                visiable_gpu_device_num = range(gpu_num)
            else:
                print "Gpu num is not the same as len of gpu_device_num."
                visiable_gpu_device_num = range(gpu_num)
            print "Visiable gpu number:", t.cuda.device_count(), ", numbers: ", gpu_device_num

            # model --> gpu
            self._transfer_data_to_gpu(self.model, visiable_gpu_device_num=visiable_gpu_device_num)

            # set output config
            model_dir = testing_params.get("pretrained_model_path")
            image_dir = testing_params.get("multi_thread_data_dir")

            output_data_path = testing_params.get("output_data_path")
            result_file = testing_params.get("result_file")
            retrain_sample_file = testing_params.get("retrain_sample_file")
            all_sample_file = testing_params.get("all_sample_file")

            train_or_test = "train" if testing_params.get("flag_training_dataset") else "test"
            model_number = model_dir.strip().strip("/").split("/")[-1]
            result_dir = os.path.join(output_data_path, "result_analysis", model_number, train_or_test, str(i))
            if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            with open(os.path.join(result_dir, result_file), 'w') as f:
                f.write(image_dir)
                f.write("\n")
            print "Image Dir:", image_dir

            data_pattern = testing_params.get("data_pattern")
            annotation_file_path = testing_params.get("annotation_file_path")
            flag_training_dataset = testing_params.get("flag_training_dataset")
            flag_testing_dataset = testing_params.get("flag_testing_dataset")
            test_data = Places(image_dir, data_pattern, annotation_file_path,
                               is_training=flag_training_dataset,
                               is_testing=flag_testing_dataset)
            test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False,
                                         num_workers=testing_params.get("thread_num"))

            # run each model
            model_path_list = []
            if os.path.isdir(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for filename in files:
                        # if filename.strip().endswith(".pth"):
                        # todo:
                        if filename.strip().startswith("color"):
                            model_path_list.append(os.path.join(root, filename))
            else:
                model_path_list = [os.path.abspath(model_dir)]

            if len(model_path_list) == 0:
                raise

            for model_path in model_path_list:
                print model_path
                self.model.load_state_dict(t.load(model_path))

                self.model.eval()

                train_or_test = "train" if flag_training_dataset else "test"

                for ii, (data, ground_truth, filename, min_value, max_value) in enumerate(test_dataloader):
                    if data.size()[0] != 1:
                        print "Error."

                    if testing_params.get("gpu_num") is not None and testing_params.get("gpu_num") != 0:
                        input = Variable(data.float(), volatile=True).cuda()
                    else:
                        input = Variable(data.float(), volatile=True)

                    score = nn.parallel.data_parallel(self.model, input, device_ids=visiable_gpu_device_num)

                    total_score = np.stack([data[0][0].numpy() * 100.,
                                            score.cpu().data[0][0].numpy(),
                                            score.cpu().data[0][1].numpy()], axis=0).transpose((1, 2, 0))
                    total_score = color.lab2rgb(total_score)
                    # print np.max(total_score), np.min(total_score)
                    # plt.imshow(total_score)
                    # plt.show()

                    # print output_data_path, "result", model_number, train_or_test, "0", filename[0]
                    original_image = np.stack([data[0][0].numpy() * 100.,
                                               ground_truth[0][0].numpy() * 200. - 100.,
                                               ground_truth[0][1].numpy() * 200. - 100.], axis=0).transpose((1, 2, 0))
                    original_image = color.lab2rgb(original_image)

                    image_dir = os.path.join(output_data_path, "result", model_number, train_or_test, "0")
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    output_path = os.path.join(image_dir, filename[0])

                    self.plot_image(data[0][0].numpy(), original_image, total_score, output_path)
                    # plt.imsave(output_path, total_score)

    @staticmethod
    def plot_image(img_array_1, img_array_2, img_array_3, output_image_filepath):
            plt.figure(figsize=(12, 4))
            plt.title(u"result")

            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow(img_array_1, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Original")
            plt.imshow(img_array_2)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Output")
            plt.imshow(img_array_3)
            plt.axis('off')

            plt.savefig(output_image_filepath, dpi=100)
            # plt.show()
