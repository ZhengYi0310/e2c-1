from e2c import * 
from configs import LidarEncoder 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim


LAMBDA = 1e-4 # 100
lambda_adv = .01
print_every = 50
save_every = 2
dump_every = 200
batch_size = 10
mse_pretrain = -1

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

ext = '_e2c_wgan_{}_{}_'.format(LAMBDA, lambda_adv)
epochs = 100

model = E2C(512*60, 100, 2, config='lidar').cuda()
netD  = LidarEncoder(-1, -1, nz=1).cuda()

print model.encoder
print model.decoder
print netD

dataset = hkl.load('../data/triplets_5_train.hkl')
train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

# print dataset key values to validate preprocessing
print(len(train_loader))
ff = np.array([ f for (f,u,n) in dataset])
print ff.shape[0], ff.min(), ff.max(), ff.mean()

one = torch.FloatTensor([1]).cuda()
mone = one * -1

optimizerG = optim.Adam(model.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters() , lr=1e-4)

for epoch in range(epochs) : 
    model.train()
    data_iter = iter(train_loader)
    gen_iters, disc_iters, iters = 0, 0, 0
    recons, kls, klds, fake_d, fake_g1, fake_g2, real_d = [0] * 7
    while iters < len(train_loader) - 6:
        j = 0

        """ update Discriminator Network """
        while j < 5  and iters < len(train_loader):
            disc_iters += 1; j += 1; iters += 1
            
            # overhead
            optimizerD.zero_grad()
            set_grad(netD, True)
            set_grad(model, False)
            
            # fetch data
            x_t, _, x_tp1 = next(data_iter)

            # to make sure we go through all available samples
            x = x_t if np.random.random() < .5 else x_tp1
            
            # put on GPU
            x = Variable(x.cuda())
             
            model(x, x_t_only=True)
            fake_out = netD(model.x_dec).mean()
            fake_d += fake_out.cpu().data[0]
            fake_out.backward(one)
            grad_penalty = 10 * calc_gradient_penalty(netD,
                                                      x.data, 
                                                      model.x_dec.data)

            real_out = netD(x).mean()
            real_d += real_out.cpu().data[0]
            real_out.backward(mone)
            optimizerD.step()

        
        """ update Generator Network """
        gen_iters += 1; iters += 1
        
        # overhead
        optimizerG.zero_grad()
        set_grad(netD, False)
        set_grad(model, True)

        # fetch data
        x_t, u_t, x_tp1 = next(data_iter)

        # put on GPU
        x_t   = Variable(x_t.cuda())
        u_t   = Variable(u_t.cuda())
        x_tp1 = Variable(x_tp1.cuda())

        # run forward pass
        model(x_t, u_t, x_tp1)
        
        # calculate loss for E2C
        recon_x_t, recon_x_tp1, kld_element, kld, kl = (
                                          compute_loss(model.x_dec, 
                                          model.x_next_pred_dec, 
                                          x_t, 
                                          x_tp1, 
                                          model.Qz, 
                                          model.Qz_next_pred, 
                                          model.Qz_next))

        # adversarial loss
        fake_out_x_t = netD(model.x_dec).mean()
        fake_g1 += fake_out_x_t.cpu().data[0]
        fake_out_x_t = lambda_adv * fake_out_x_t
        fake_out_x_t.backward(mone, retain_graph=True)
        
        recon_loss = recon_x_t + recon_x_tp1
        recons += recon_loss.cpu().data[0]
        klds += kld.cpu().data[0]
        kls += kl.cpu().data[0]
        
        alpha = 1e-4
        loss = recon_loss + LAMBDA * kl.mean() + alpha * kld.mean()
        loss.mean().backward()
        optimizerG.step()

        if (iters + 1) % print_every <= 1 :  
            print epoch
            print('recon loss : %s   '   % (2*recons  / print_every))
            print('klds loss  : %s   '   % (2*klds    / print_every))
            # print('kls loss   : %s   '   % (2*kls     / print_every))'''
            print('fake_d     : %s   '   % (fake_d  / gen_iters))
            print('fake_g1    : %s   '   % (fake_g1 / gen_iters))
            # print('fake_g2    : %s   '   % (fake_g2 / gen_iters))
            print('real_d     : %s   '   % (real_d  / disc_iters))
            recons, kls, klds, fake_d, fake_g1, fake_g2, real_d = [0] * 7
            gen_iters, disc_iters = 0, 0
            monitor_units(kld_element.mul(-.5))

        if (iters + 1) % dump_every <= 1 : 
            save_sample('../clouds/x_dec_%s.hkl'      % epoch, model.x_dec[0])
            save_sample('../clouds/x_next_dec_%s.hkl' % epoch, model.x_next_dec[0])
            save_sample('../clouds/x_%s.hkl'          % epoch, x_t[0])
            save_sample('../clouds/x_next_%s.hkl'     % epoch, x_tp1[0])
            print 'dumped samples'

    if (epoch + 1) % save_every == 0 : 
        torch.save(model.state_dict(), '../models/' + ext + str(epoch) + '.pth')
        print 'model saved'   


