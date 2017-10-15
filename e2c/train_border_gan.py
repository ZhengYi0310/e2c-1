from e2c import * 
from configs import LidarEncoder 
from torch.autograd import Variable
from utils import * 
import torch.optim as optim
from tensorboardX import SummaryWriter

# KL ANNEAL
theta = 0             # theta start
theta_speed = 0.00001 # increase per epoch
theta_max = .001      # theta end 

print_every = 25      # mini batches
save_every = 5        # epochs
dump_every = 1        # epochs
batch_size = 32
epochs = 500
lambda_adv = .001
lambda_mse = 1. - lambda_adv
GEN_ITERS = 2         # only when lambda_adv very low

model_name = 'e2c_lsgan_strutural_no_hid_{}_{}'.format(theta_speed, lambda_adv) 
writer = SummaryWriter(log_dir = 'runs/' + model_name)

# maximize reproducibility
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

model = E2C(512*60, 100, 2, config='lidar').cuda()
netD  = LidarEncoder(-1, -1, nz=1).cuda()

#model.decoder.load_state_dict(torch.load(
#    '../../Lidar/models/modelG_lsgan_40.pth'))
# netD.load_state_dict(torch.load(
#     '../../Lidar/models/modelD_lsgan_100.pth'))

print model.encoder
print model.decoder
print netD

dataset = hkl.load('../data/triplets_borders_5_train.hkl')
train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)
theta_speed /= len(train_loader)

optimizerG = optim.Adam(model.parameters(), lr=1e-3) #1e-4
optimizerD = optim.RMSprop(netD.parameters() , lr=1e-4)


writes = 0
for epoch in range(epochs) : 
    print 'epoch : ', epoch
    model.train()
    data_iter = iter(train_loader)
    gen_iters, disc_iters, iters = 0, 0, 0
    (recons, kls, klds, fake_d, fake_g1, fake_g2, 
     real_d, prior_g, prior_d, total_loss, hid_g) = [0] * 11
    writer.add_scalar('data/theta', theta, epoch)

    while iters < len(train_loader) - 2:
        disc_iters += 1
        """ update Discriminator Network """
        # overhead
        optimizerD.zero_grad()
        set_grad(netD, True)
        set_grad(model, False)
        
        # fetch data
        x_t, _, x_tp1, _, _ = next(data_iter)
        iters += 1

        # to make sure we go through all available samples
        x = x_t # if np.random.random() < .5 else x_tp1
        
        # put on GPU
        x = Variable(x.cuda())
         
        model(x, x_t_only=True)
        fake_out, real_hid = netD(model.x_dec, return_hidden=True)
        fake_d += fake_out.mean().cpu().data[0]
        
        # draw sample from prior
        noise = Variable(torch.cuda.FloatTensor(batch_size, 100, 1, 1).cuda())
        prior = model.decode(noise)
        prior_out = netD(prior)
        prior_d += prior_out.mean().cpu().data[0]
        loss_fake = torch.mean((fake_out - 0.) ** 2 + (prior_out - 0.) ** 2)

        real_out = netD(x)
        real_d += real_out.mean().cpu().data[0]
        loss_real = torch.mean((real_out - 1.) ** 2)

        loss = (loss_real + loss_fake) / 2.
        loss.backward()
        optimizerD.step()
        
        for _ in range(GEN_ITERS): 
            """ update Generator Network """
            optimizerG.zero_grad()
            gen_iters += 1
            set_grad(netD, False)
            set_grad(model, True)

            x_t, u_t, x_tp1, border_x_t, border_x_tp1 = next(data_iter)
            iters += 1

            # put on GPU
            x_t   = Variable(x_t.cuda())
            u_t   = Variable(u_t.cuda())
            x_tp1 = Variable(x_tp1.cuda())
            border_x_t = Variable(border_x_t.cuda())
            border_x_tp1 = Variable(border_x_tp1.cuda())

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

            # structural loss
            mask = (border_x_t != 0.).float()
            recon_x_t   = torch.mean(mask * (model.x_dec - x_t) ** 2)
            mask = (border_x_tp1 != 0.).float()
            recon_x_tp1 = torch.mean(mask * (model.x_next_pred_dec - x_tp1) ** 2)
            
                                              
            # draw sample from prior
            noise = Variable(torch.cuda.FloatTensor(batch_size, 100, 1, 1).cuda())
            prior = model.decode(noise)
            prior_out = netD(prior)

            # adversarial loss
            _,            real_hid  = netD(x_t, return_hidden=True)
            fake_out_x_t, fake_hid  = netD(model.x_dec, return_hidden=True)
            fake_out_x_tp1          = netD(model.x_next_dec)
            loss_adv = torch.mean((fake_out_x_t   - 1.) ** 2 + 
                                  (fake_out_x_tp1 - 1.) ** 2) 
                                  
            loss_prior = torch.mean((prior_out- 1.) ** 2)
            loss_hid   = torch.mean((real_hid - fake_hid) ** 2)
            
            prior_g += prior_out.mean().cpu().data[0]
            hid_g   += loss_hid.cpu().data[0]
            fake_g1 += fake_out_x_t.mean().cpu().data[0]
            fake_g2 += fake_out_x_tp1.mean().cpu().data[0]

            alpha = theta
            recon_loss = recon_x_t + recon_x_tp1
            loss = (50 * recon_loss + # 50 before
                    theta * kl.mean() + 
                    alpha * kld.mean() + 
                    lambda_adv * (loss_adv)).mean()# + #)# + loss_prior) + 
                    #lambda_mse * loss_hid).mean()

            total_loss += loss.cpu().data[0]
            loss.backward()
            optimizerG.step()

            recons += recon_loss.cpu().data[0]
            klds += kld.cpu().data[0]
            kls += kl.cpu().data[0]
            
            # update theta 
            theta = min(theta_max, theta + theta_speed)

        if (iters + 1) % print_every <= 1 :  
            '''
            print 'data/kldsi'    , (klds    / gen_iters)
            print 'data/kls'      , (kls     / gen_iters)
            print 'data/recon'    , (recons  / gen_iters)
            print 'data/hid_g'    , (hid_g   / gen_iters)
            print 'gan/fake disc' , (fake_d  / disc_iters)
            print 'gan/fake g1'   , (fake_g1 / gen_iters)
            print 'gan/fake g2'   , (fake_g2 / gen_iters)
            print 'gan/prior g'   , (prior_g / gen_iters)
            print 'gan/prior d'   , (prior_d / gen_iters)
            print 'gan/real disc' , (real_d  / disc_iters)
            '''

            writer.add_scalar('data/klds'     , (klds    / gen_iters), writes)
            writer.add_scalar('data/kls'      , (kls     / gen_iters), writes)
            writer.add_scalar('data/recon'    , (recons  / gen_iters), writes)
            writer.add_scalar('gan/fake disc' , (fake_d  / disc_iters), writes)
            writer.add_scalar('gan/fake g1'   , (fake_g1 / gen_iters), writes)
            writer.add_scalar('gan/fake g2'   , (fake_g2 / gen_iters), writes)
            writer.add_scalar('gan/prior g'   , (prior_g / gen_iters), writes)
            writer.add_scalar('gan/prior d'   , (prior_d / gen_iters), writes)
            writer.add_scalar('gan/real disc' , (real_d  / disc_iters), writes)
            writer.add_scalar('gan/hid g'     , (hid_g   / gen_iters), writes)
            
            print 'wrote to tensorboard'

            activations = kld_element.mul(-.5).data.mean(dim=0).cpu().numpy()
            writer.add_histogram('activations', activations, writes)

            (recons, kls, klds, fake_d, fake_g1, fake_g2, 
             real_d, prior_g, prior_d, total_loss, hid_g) = [0] * 11
            gen_iters, disc_iters = 0, 0
            writes += 1

    # end of epoch
    if (epoch + 1) % dump_every == 0 : 
        save_sample('../clouds/x_dec_%s_%s.hkl'      % (model_name, epoch), model.x_dec[0])
        save_sample('../clouds/x_next_dec_%s_%s.hkl' % (model_name, epoch), model.x_next_dec[0])
        save_sample('../clouds/x_%s_%s.hkl'          % (model_name, epoch), x_t[0])
        save_sample('../clouds/x_next_%s_%s.hkl'     % (model_name, epoch), x_tp1[0])
        print 'dumped samples'
        
    if (epoch + 1) % save_every == 0 : 
        torch.save(model.state_dict(), '../models/' + model_name + str(epoch) + '.pth')
        print 'model saved'   


