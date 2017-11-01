import glob
import os
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
import random

from datetime import datetime
# from torchvision.transforms import ToTensor
#from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.color import rgb2gray
from tqdm import trange, tqdm
import pickle

class PendulumData(object):
    def __init__(self, root, split):
        if split not in ['train', 'test', 'all']:
            raise ValueError

        dir = os.path.join(root, split)
        filenames = glob.glob(os.path.join(dir, '*.png'))

        if split == 'all':
            filenames = glob.glob(os.path.join(root, 'train/*.png'))
            filenames.extend(glob.glob(os.path.join(root, 'test/*.png')))

        filenames = sorted(
            filenames, key=lambda x: int(os.path.basename(x).split('.')[0]))

        images = []

        for f in filenames:
            img = plt.imread(f)
            img[img != 1] = 0
            images.append(resize(rgb2gray(img), [48, 48], mode='constant'))

        self.images = np.array(images, dtype=np.float32)
        self.images = self.images.reshape([len(images), 48, 48, 1])

        action_filename = os.path.join(root, 'actions.txt')

        with open(action_filename) as infile:
            actions = np.array([float(l) for l in infile.readlines()])

        self.actions = actions[:len(self.images)].astype(np.float32)
        self.actions = self.actions.reshape(len(actions), 1)

    def __len__(self):
        return len(self.actions) - 1

    def __getitem__(self, index):
        return self.images[index], self.actions[index], self.images[index]

class GymPendulumDatasetV2(object):
    width = 48 * 2
    height = 48
    action_dim = 1

    def __init__(self, dir):
        self.dir = dir
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()
        self._index_in_epoch = 0
        self._epochs_completed = False
        self._num_examples = len(self)
        self.processed = self._processed

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        temp = np.squeeze(np.reshape(np.array((img.convert('L').
                           resize((GymPendulumDatasetV2.width,
                                   GymPendulumDatasetV2.height)))), (-1, GymPendulumDatasetV2.width * GymPendulumDatasetV2.height)).astype(np.float32), axis=0)
        # temp[temp != 255.] = 0.
        return temp

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing data'):
                before = Image.open(os.path.join(self.dir, sample['before']))
                after = Image.open(os.path.join(self.dir, sample['after']))

                processed.append((self._process_image(before),
                                  np.array(sample['control'], dtype=np.float32),
                                  self._process_image(after),
                                  np.array(sample['before_state']),
                                  np.array(sample['after_state'])))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)

    @staticmethod
    def _render_state_fully_observed(env, state):
        before1 = state
        before2 = env.step_from_state(state, np.array([0]))
        return map(env.render_state, [before1[0], before2[0]])

    @classmethod
    def sample(cls, sample_size, output_dir, step_size=1,
               apply_control=True, num_shards=10):
        env = gym.make('Pendulum-v0').env
        assert sample_size % num_shards == 0

        samples = []

        if not path.exists(output_dir):
            os.makedirs(output_dir)

        for i in trange(sample_size):
            th = np.random.uniform(0, 2 * np.pi)
            thdot = np.random.uniform(-10, 10)

            state = np.array([th, thdot])
            u0 = np.array([0])

            initial_state = state
            before1, before2 = GymPendulumDatasetV2._render_state_fully_observed(env, state)

            # apply the same control over a few timesteps
            if apply_control:
                u = np.random.uniform(-2, 2, size=(1,))
            else:
                u = np.zeros((1,))

            # state = env.step_from_state(state, u0)
            for _ in range(step_size):
                state = env.step_from_state(state, u)

            after_state = state
            after1, after2 = GymPendulumDatasetV2._render_state_fully_observed(env, state)

            before = np.hstack((before1, before2))
            after = np.hstack((after1, after2))

            shard_no = i // (sample_size // num_shards)

            shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no, num_shards))

            if not path.exists(path.join(output_dir, shard_path)):
                os.makedirs(path.join(output_dir, shard_path))

            before_file = path.join(shard_path, 'before-{:05d}.png'.format(i))
            plt.imsave(path.join(output_dir, before_file), before)

            after_file = path.join(shard_path, 'after-{:05d}.png'.format(i))
            plt.imsave(path.join(output_dir, after_file), after)

            samples.append({
                'before_state': initial_state.tolist(),
                'after_state': after_state.tolist(),
                'before': before_file,
                'after': after_file,
                'control': u.tolist(),
            })

        with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
            json.dump(
                {
                    'metadata': {
                        'num_samples': sample_size,
                        'step_size': step_size,
                        'apply_control': apply_control,
                        'time_created': str(datetime.now()),
                        'version': 1
                    },
                    'samples': samples
                }, outfile, indent=2)

        env.viewer.close()

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            random.shuffle(self.processed)
        if start + batch_size > self._num_examples:
            # Finished epoch
            # print("[WARN]: All datas have been iterated!!!!")
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            processed_data_rest_part = self.processed[start:self._num_examples]

            if shuffle:
                random.shuffle(self.processed)
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            processed_data_new_part = self.processed[start:self._index_in_epoch]
            return processed_data_rest_part + processed_data_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.processed[start:end]


if __name__=="__main__":
    GymPendulumDatasetV2.sample(1000, 'data/pendulum_markov_train')
    # GymPendulumDatasetV2.sample(100, 'data/pendulum_markov_test')
    dataset = GymPendulumDatasetV2('data/pendulum_markov_train')
    perm0 = np.arange(dataset._num_examples)
    before_image = [np.reshape(before_img, (-1, 4608)) for ind, (before_img,control_signal, after_image, before_states, after_states) in enumerate(dataset)]
    # print ((np.array(before_image, np.float64))).shape
    # print np.array(before_image, np.float64)[0,:,23*48 + 24]
    print before_image[0].shape
    after_image = [np.reshape(after_image, (-1, 4608)) for
                 ind, (before_img, control_signal, after_image, before_states, after_states) in enumerate(dataset)]
    print ((np.array(after_image, np.float64))).shape
    control_signal = [np.reshape(control_signal, (-1, dataset.action_dim)) for ind, (before_img,control_signal, after_img, before_states, after_states) in enumerate(dataset)]
    print (np.squeeze(np.array(control_signal, np.float64), axis=1)).shape
    # print (dataset.next_batch(100)[0][0])
    # print len(dataset.next_batch(10))
    # random.shuffle(dataset.processed)
    # print len(dataset.processed[1:3] + (dataset.processed[2:4]))

    # dataset_before =  (dataset[0][0])
    # dataset_after = (dataset[0][2])
    # print (dataset_before)
    # before_image = [before_image_single[before_image_single != 255] = 0 for ind,(before_image_single) in enumerate(before_image)]
    # for i in range(len(before_image)):
    #     before_image[i][before_image[i] != 255] = 0.
    print before_image[60]
    plt.imshow(np.reshape(before_image[60], (48, 2 * 48)), cmap="gray")
    plt.show()
    plt.imshow(np.reshape(after_image[0], (48,  2 * 48)), cmap="gray")
    plt.show()
    # a = np.array([[1, 2], [3, 4], [10, 11]])
    # a = np.reshape(a, ())
    # print(dataset[0][3])
    # print(dataset[0][4])