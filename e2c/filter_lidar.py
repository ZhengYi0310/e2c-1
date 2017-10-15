from e2c import * 
from configs import * 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim
from tensorboardX import SummaryWriter

index = 12345
dataset = hkl.load('../../prednet/kitti_data/X_train_1d_512.hkl')
dataset = dataset.astype('float32') / 80.
print dataset.shape
# ds has shape num_samples x 60 x 512 x 1
avg = dataset.mean(axis=0).mean(axis=1)
std = dataset.std(axis=0).mean(axis=1)
ps = [0.0015, 0.001, 0.0005, 1]
#dilater = np.arange(1., 1.5, 1/60.)
tt = 0.01
for index in np.random.choice(dataset.shape[0], 10):
    sample = dataset[index]
    for p in ps:
        t = 0 if p == 1 else tt
        test = np.array(dataset[index], copy=True)
        for i in range(test.shape[0]-1):
            mu, sigma = avg[i], std[i]
            for j in range(test.shape[1]):
                pt = sample[i][j]
                #* dilater[i]
                # mu, sigma = avg[i], theta * std[i]
                if np.abs(sample[i][j] - sample[i+1][j]) > p or (pt > mu - t*sigma and pt < mu + t*sigma)  : 
                    test[i][j] = 0.
        save_sample('../clouds/fake_{}_{}.hkl'.format(index, p), test, is_numpy=True)
    

