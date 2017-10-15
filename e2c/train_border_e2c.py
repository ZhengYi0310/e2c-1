from e2c import * 
from configs import * 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim
from tensorboardX import SummaryWriter


# KL ANNEAL
theta = 0               # theta start
theta_speed = 0.0001     # increase per epoch
theta_max = .01          # theta end 

print_every = 25        # mini batches
save_every = 5          # epochs
dump_every = 2          # epochs
batch_size = 32
epochs = 200

model_name = 'e2c_borders_og_30_' + '_' + str(theta_speed)
writer = SummaryWriter(log_dir = 'runs/' + model_name)

model = E2C(512*60, 30, 2, config='lidar').cuda()
print model.encoder
print model.decoder

dataset = hkl.load('../data/triplets_borders_5_train.hkl')
train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)
theta_speed /= len(train_loader)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
writes = 0
for epoch in range(epochs) : 
    print 'epoch : ', epoch
    model.train()
    recons, kls, klds, total_loss, iters = [0] * 5
    writer.add_scalar('data/theta', theta, epoch)
    for batch_idx, data in enumerate(train_loader):
        iters += 1
        optimizer.zero_grad()
        x_t, u_t, x_tp1, border_x_t, border_x_tp1 = data
        
        # put on GPU
        x_t   = Variable(x_t.cuda())
        u_t   = Variable(u_t.cuda())
        x_tp1 = Variable(x_tp1.cuda())
        border_x_t = Variable(border_x_t.cuda())
        border_x_tp1 = Variable(border_x_tp1.cuda())

        # run forward pass
        model(x_t, u_t, x_tp1)
        
        # calculate loss
        recon_x_t, recon_x_tp1, kld_element, kld, kl = (
                                          compute_loss(model.x_dec, 
                                          model.x_next_pred_dec, 
                                          x_t, 
                                          x_tp1, 
                                          model.Qz, 
                                          model.Qz_next_pred, 
                                          model.Qz_next, 
                                          use_mask=False))

        
        mask = (border_x_t != 0.).float()
        recon_x_t   = torch.mean(mask * (model.x_dec - x_t) ** 2)
        mask = (border_x_tp1 != 0.).float()
        recon_x_tp1 = torch.mean(mask * (model.x_next_pred_dec - x_tp1) ** 2)
        recon_loss = recon_x_t + recon_x_tp1
        recons += recon_loss.cpu().data[0]
        klds += kld.cpu().data[0]
        kls += kl.cpu().data[0]
        
        alpha = theta
        loss = (recon_loss + theta * kl.mean() + alpha * kld.mean()).mean()
        total_loss += loss.cpu().data[0]
        loss.backward()
        optimizer.step()

        # update theta
        theta = min(theta_max, theta + theta_speed)
        # beta  = min(beta_max, beta + beta_speed)

        if (batch_idx + 1) % print_every == 0 :  
            writer.add_scalar('data/klds',  (klds   / iters), writes)
            writer.add_scalar('data/kls' ,  (kls    / iters), writes)
            writer.add_scalar('data/recon', (recons / iters), writes)
            activations = kld_element.mul(-.5).data.mean(dim=0).cpu().numpy()
            writer.add_histogram('activations', activations, writes)
            recons, kls, klds, total_loss, iters = [0] * 5
            writes += 1

    # end of epoch here
    if (epoch + 1) % dump_every == 0 : 
        save_sample('../clouds/x_dec_%s_%s.hkl'      % (model_name, epoch), model.x_dec[0])
        save_sample('../clouds/x_next_dec_%s_%s.hkl' % (model_name, epoch), model.x_next_dec[0])
        save_sample('../clouds/x_%s_%s.hkl'          % (model_name, epoch), x_t[0])
        save_sample('../clouds/x_next_%s_%s.hkl'     % (model_name, epoch), x_tp1[0])
        print 'dumped samples'

    if (epoch + 1) % save_every == 0 : 
        torch.save(model.state_dict(), '../models/' + model_name + str(epoch) + '.pth')
        print 'model saved'   


