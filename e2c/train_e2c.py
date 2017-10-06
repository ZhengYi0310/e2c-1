from e2c import * 
from configs import * 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim


LAMBDA = 1 # 100
print_every = 100
save_every = 2
batch_size = 16
ext = '_e2c_lidar_{}_'.format(LAMBDA)
epochs = 100

model = E2C(512*60, 100, 2, config='lidar').cuda()
print model.encoder
print model.decoder

dataset = hkl.load('../data/triplets_5_train.hkl')
train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

# print dataset key values to validate preprocessing
ff = np.array([ f for (f,u,n) in dataset])
print ff.shape[0], ff.min(), ff.max(), ff.mean()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs) : 
    model.train()
    recons, kls, klds = [0] * 3
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x_t, u_t, x_tp1 = data
        
        # put on GPU
        x_t   = Variable(x_t.cuda())
        u_t   = Variable(u_t.cuda())
        x_tp1 = Variable(x_tp1.cuda())

        # run forward pass
        model(x_t, u_t, x_tp1, epoch=epoch)
        
        # calculate loss
        recon_x_t, recon_x_tp1, kld_element, kld, kl = (
                                          compute_loss(model.x_dec, 
                                          model.x_next_pred_dec, 
                                          x_t, 
                                          x_tp1, 
                                          model.Qz, 
                                          model.Qz_next_pred, 
                                          model.Qz_next))

        recon_loss = recon_x_t + recon_x_tp1
        recons += recon_loss.cpu().data[0]
        klds += kld.cpu().data[0]
        kls += kl.cpu().data[0]
        
        # bound_loss = recon_loss + kld
        # loss = bound_loss + LAMBDA * kl
        # loss =  recon_loss + alpha * kld.mean()
        
        alpha = 5e-6 if epoch > -1 else 0
        loss = recon_loss + LAMBDA * kl.mean() + alpha * kld.mean()
        loss.mean().backward()
        optimizer.step()

        if (batch_idx + 1) % print_every == 0 :  
            print epoch
            print('recon loss : %s   '   % (recons / print_every))
            print('klds loss  : %s   '   % (klds   / print_every))
            print('kls loss   : %s   '   % (kls    / print_every))
            monitor_units(kld_element.mul(-.5))

            save_sample('../clouds/x_dec_%s.hkl'      % epoch, model.x_dec[0])
            save_sample('../clouds/x_next_dec_%s.hkl' % epoch, model.x_next_dec[0])
            save_sample('../clouds/x_%s.hkl'          % epoch, x_t[0])
            save_sample('../clouds/x_next_%s.hkl'     % epoch, x_tp1[0])
            recons, kls, klds = [0] * 3

    if (epoch + 1) % save_every == 0 : 
        torch.save(model.state_dict(), '../models/' + ext + str(epoch) + '.pth')
        print 'model saved'   


