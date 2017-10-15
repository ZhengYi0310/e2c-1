from e2c import * 
from configs import LidarEncoder 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim
from collections import OrderedDict
import pdb

# hallucination length
hallu_len = 10

# maximize reproducibility
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# name of models I want to test for
models = OrderedDict()
models['e2c_bn_1e-05'] = [25, 49, 99, 149, 199]
models['e2c_lsgan']    = [25, 49, 99, 149, 199, 241]

# load models on GPU
all_models = OrderedDict()
for name in models.keys():
    for epoch in models[name]:
        model = E2C(512*60, 100, 2, config='lidar').cuda()
        full_name = name + str(epoch)
        path = '../models/' + full_name + '.pth'
        model.load_state_dict(torch.load(path))
        all_models[full_name] = model

# reduce overhead
for model in all_models.values():
    for p in model.parameters():
        p.requires_grad = False


model = E2C(512*60, 100, 2, config='lidar').cuda()
netD  = LidarEncoder(-1, -1, nz=1).cuda()

dataset = hkl.load('../data/triplets_5_test.hkl')
test_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=1,
                                           shuffle=False)

for name, model in all_models.items():
    print name
    # model.test()
    iters = 0
    for batch_idx, data in enumerate(test_loader):     
            iters += 1
            x_t, u_t, x_tp1 = data
            # put on GPU
            x_t   = Variable(x_t.cuda())
            u_t   = Variable(u_t.cuda())
            x_tp1 = Variable(x_tp1.cuda())

            if (iters % hallu_len) == 1 : # stop hallucinating
                mu, logvar   = model.encode(x_t)
                z, Qz        = model.reparam(mu, logvar * 0.)      
                z_c, Qz_c   = z, Qz
                
            else : # use previous prediction as ground truth
                z, Qz       = z_tp1, Qz_tp1
                z_c, Qz_c   = z_tp1_c, Qz_tp1_c

            # corrupt the next action 
            u_t_c = u_t
            # 3x speed
            u_t_c[:, 0] = u_t_c[:, 0] * 3
            # change turning angle
            u_t_c[:, 1] = - u_t_c[:, 1]

            # predict next
            z_tp1, Qz_tp1     = model.predict_latent(z, Qz, u_t)
            z_tp1_c, Qz_tp1_c = model.predict_latent(z_c, Qz_c, u_t_c)

            x_pred   = model.decode(z_tp1)
            x_pred_c = model.decode(z_tp1_c)

            save_sample('../test_clouds/%s_%s_real.hkl'      % (iters, name), x_tp1[0])
            save_sample('../test_clouds/%s_%s_recon.hkl'     % (iters, name), x_pred[0])
            save_sample('../test_clouds/%s_%s_corrupt.hkl'   % (iters, name), x_pred_c[0])



