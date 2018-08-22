import os

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.optim import lr_scheduler

import mlp
from data_reader import BatchImageGenerator
from utils import sgd, crossentropyloss, fix_seed, write_log, compute_accuracy


class ModelBaseline:
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # fix the random seed or not
        fix_seed()

        self.setup_path(flags)

        self.network = mlp.MLPNet(num_classes=flags.num_classes)

        self.network = self.network.cuda()

        print(self.network)
        print('flags:', flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags.state_dict)

        self.configure(flags)

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = ['art_painting_train_features.hdf5',
                      'cartoon_train_features.hdf5',
                      'photo_train_features.hdf5',
                      'sketch_train_features.hdf5']

        val_data = ['art_painting_val_features.hdf5',
                    'cartoon_val_features.hdf5',
                    'photo_val_features.hdf5',
                    'sketch_val_features.hdf5']

        test_data = ['art_painting_features.hdf5',
                     'cartoon_features.hdf5',
                     'photo_features.hdf5',
                     'sketch_features.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 b_unfold_label=True)
            self.batImageGenVals.append(batImageGenVal)

    def load_state_dict(self, state_dict=''):

        if state_dict:
            try:
                tmp = torch.load(state_dict)
                pretrained_dict = tmp['state']
            except:
                pretrained_dict = model_zoo.load_url(state_dict)

            model_dict = self.network.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.network.load_state_dict(model_dict)

    def heldout_test(self, flags):

        # load the best model in the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)

        # test
        batImageGenTest = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path, stage='test',
                                              b_unfold_label=False)
        test_images = batImageGenTest.images

        threshold = 100
        n_slices_test = len(test_images) / threshold
        indices_test = []
        for per_slice in range(n_slices_test - 1):
            indices_test.append(len(test_images) * (per_slice + 1) / n_slices_test)
        test_image_splits = np.split(test_images, indices_or_sections=indices_test)

        # Verify the splits are correct
        test_image_splits_2_whole = np.concatenate(test_image_splits)
        assert np.all(test_images == test_image_splits_2_whole)

        # split the test data into splits and test them one by one
        predictions = []
        self.network.eval()
        for test_image_split in test_image_splits:
            images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
            outputs, end_points = self.network(images_test)

            pred = end_points['Predictions']
            pred = pred.cpu().data.numpy()
            predictions.append(pred)

        # concatenate the test predictions first
        predictions = np.concatenate(predictions)

        # accuracy
        accuracy = accuracy_score(y_true=batImageGenTest.labels,
                                  y_pred=np.argmax(predictions, -1))

        flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
        write_log(accuracy, flags_log)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = sgd(parameters=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = crossentropyloss()

    def train(self, flags):
        self.network.train()

        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            self.scheduler.step(epoch=ite)

            total_loss = 0.0
            for index in range(len(self.batImageGenTrains)):
                images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch()

                inputs, labels = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda()

                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)
                total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite, 'loss:', total_loss.cpu().data.numpy()[0], 'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(total_loss.cpu().data.numpy()[0]),
                flags_log)

            del total_loss, outputs

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(self.batImageGenVals, flags, ite)

    def test_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.network.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = len(images_test) / threshold
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(images_test) * (per_slice + 1) / n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                outputs, end_points = self.network(images_test)

                predictions = end_points['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            outputs, end_points = self.network(images_test)

            predictions = end_points['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        # switch on the network train mode after test
        self.network.train()

        return accuracy


class ModelMLDG(ModelBaseline):
    def __init__(self, flags):

        ModelBaseline.__init__(self, flags)

    def train(self, flags):
        self.network.train()

        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            self.scheduler.step(epoch=ite)

            # select the validation domain for meta val
            index_val = np.random.choice(a=np.arange(0, len(self.batImageGenTrains)), size=1)[0]
            batImageMetaVal = self.batImageGenTrains[index_val]

            meta_train_loss = 0.0
            # get the inputs and labels from the data reader
            for index in range(len(self.batImageGenTrains)):

                if index == index_val:
                    continue

                images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch()

                inputs_train, labels_train = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs_train, labels_train = Variable(inputs_train, requires_grad=False).cuda(), \
                                             Variable(labels_train, requires_grad=False).long().cuda()

                # forward with the adapted parameters
                outputs_train, _ = self.network(x=inputs_train)

                # loss
                loss = self.loss_fn(outputs_train, labels_train)
                meta_train_loss += loss

            image_val, labels_val = batImageMetaVal.get_images_labels_batch()
            inputs_val, labels_val = torch.from_numpy(
                np.array(image_val, dtype=np.float32)), torch.from_numpy(
                np.array(labels_val, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs_val, labels_val = Variable(inputs_val, requires_grad=False).cuda(), \
                                     Variable(labels_val, requires_grad=False).long().cuda()

            # forward with the adapted parameters
            outputs_val, _ = self.network(x=inputs_val,
                                          meta_loss=meta_train_loss,
                                          meta_step_size=flags.meta_step_size,
                                          stop_gradient=flags.stop_gradient)

            meta_val_loss = self.loss_fn(outputs_val, labels_val)

            total_loss = meta_train_loss + meta_val_loss * flags.meta_val_beta

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite,
                'meta_train_loss:', meta_train_loss.cpu().data.numpy()[0],
                'meta_val_loss:', meta_val_loss.cpu().data.numpy()[0],
                'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(meta_train_loss.cpu().data.numpy()[0]) + '\t' + str(meta_val_loss.cpu().data.numpy()[0]),
                flags_log)

            del total_loss, outputs_val, outputs_train

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(self.batImageGenVals, flags, ite)
