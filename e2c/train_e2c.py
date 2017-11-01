from e2c import *
from configs import * 
from torch.autograd import Variable
from datasets import *
# from utils import *
import torch.optim as optim
# from tf.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


# KL ANNEAL
theta = 0               # theta start
theta_speed = 0.0001     # increase per epoch
theta_max = .01          # theta end 

print_every = 25        # mini batches
save_every = 5          # epochs
dump_every = 2          # epochs
batch_size = 100
epochs = 5

model_name = 'e2c_anneal_10_' + '_' + str(theta_speed)
# writer = SummaryWriter(log_dir = 'runs/' + model_name)

model = E2C(48 * 48 * 2, 3, 1, config='pendulum').cuda()
print model.encoder
print model.decoder

# dataset = hkl.load('../data/triplets_5_train.hkl')
pendulum_dataset = GymPendulumDatasetV2('data/pendulum_markov_train')
train_loader = torch.utils.data.DataLoader(pendulum_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# theta_speed /= len(train_loader)

optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.1, 0.1))
# writes = 0
recon_x_t_np = None
x_t_np = None
precit_x_tp1_np = None
x_t_next_np = None
z_mean_numpy = np.zeros((len(train_loader) * batch_size, 3), dtype=np.float32)
z_mean_tp1_numpy = np.zeros((len(train_loader) * batch_size, 3), dtype=np.float32)
state_numpy_train = np.zeros((len(train_loader) * batch_size, 2), dtype=np.float32)
state_tp1_numpy_train = np.zeros((len(train_loader) * batch_size, 2), dtype=np.float32)
print z_mean_numpy.shape

for epoch in range(epochs) :
    print 'epoch : ', epoch
    model.train()
    recons, kls, klds, total_loss, iters = [0] * 5
    batch_loss = None
    # writer.add_scalar('data/theta', theta, epoch)
    for batch_idx, data in enumerate(train_loader):
        iters += 1
        optimizer.zero_grad()
        x_t, u_t, x_tp1, state_t, state_tp1 = data

        # put on GPU
        x_t   = Variable((x_t / 255.).cuda())
        u_t   = Variable(u_t.cuda())
        x_tp1 = Variable((x_tp1 / 255.).cuda())
        state_t = Variable((state_t).cuda())
        state_tp1 = Variable((state_tp1).cuda())

        # run forward pass
        model(x_t, action=u_t, x_next=x_tp1)

        # calculate loss
        recon_x_t, recon_x_tp1, kld_element, kld, kl = (
                                          compute_loss(model.x_dec,
                                          model.x_next_pred_dec,
                                          x_t,
                                          x_tp1,
                                          model.Qz,
                                          model.Qz_next_pred,
                                          model.Qz_next))
        if epoch == epochs - 1 and batch_idx == 0:
            # recon_x_t_np = (recon_x_t.data).cpu().numpy()
            recon_x_t_np = model.embedding_to_sample(model.latent_embedding(x_t))
            recon_x_t_np = (recon_x_t_np.data).cpu().numpy()
            print recon_x_t_np
            precit_x_tp1_np = model.embedding_to_sample(model.latent_embedding(model.predict(x_t, u_t)))
            precit_x_tp1_np = (precit_x_tp1_np.data).cpu().numpy()
            x_t_next_np = (x_tp1.data).cpu().numpy()
            x_t_np = (x_t.data).cpu().numpy()
        if epoch == epochs - 1:
            z_mean_numpy[batch_idx * batch_size:batch_idx * batch_size + batch_size, :] = (model.latent_embedding(x_t)).data.cpu().numpy()
            print (model.latent_embedding(x_t)).data.cpu().numpy()
            state_numpy_train[batch_idx * batch_size:batch_idx * batch_size + batch_size, :] = (state_t.data).cpu().numpy()
            state_tp1_numpy_train[batch_idx * batch_size:batch_idx * batch_size + batch_size, :] = (state_tp1.data).cpu().numpy()


        recon_loss = recon_x_t + recon_x_tp1
        recons += recon_loss.cpu().data[0]
        klds += kld.cpu().data[0]
        kls += kl.cpu().data[0]

        alpha = 0.25
        loss = (recon_loss + kl.mean() + alpha * kld.mean()).mean() #+ alpha * kld.mean()).mean()
        total_loss += loss.cpu().data[0]
        loss.backward()
        optimizer.step()

        # update theta
        theta = min(theta_max, theta + theta_speed)
        # beta  = min(beta_max, beta + beta_speed)

        # if (batch_idx + 1) % print_every == 0 :
        #     writer.add_scalar('data/klds',  (klds   / iters), writes)
        #     writer.add_scalar('data/kls' ,  (kls    / iters), writes)
        #     writer.add_scalar('data/recon', (recons / iters), writes)
        #     activations = kld_element.mul(-.5).data.mean(dim=0).cpu().numpy()
        #     writer.add_histogram('activations', activations, writes)
        #     recons, kls, klds, total_loss, iters = [0] * 5
        #     writes += 1


    # end of epoch here
    if (epoch + 1) % dump_every == 0 :
        # save_sample('../clouds/x_dec_%s_%s.hkl'      % (model_name, epoch), model.x_dec[0])
        # save_sample('../clouds/x_next_dec_%s_%s.hkl' % (model_name, epoch), model.x_next_dec[0])
        # save_sample('../clouds/x_%s_%s.hkl'          % (model_name, epoch), x_t[0])
        # save_sample('../clouds/x_next_%s_%s.hkl'     % (model_name, epoch), x_tp1[0])
        # print 'dumped samples'
        print("Epoch:", '%04d' % (epoch + 1),
              "cost=", "{:.9f}".format(total_loss))

    if (epoch + 1) % save_every == 0 :
        torch.save(model.state_dict(), 'models/' + model_name + str(epoch) + '.pth')
        print 'model saved'


# x_t, u_t, x_tp1, _, _ = pendulum_dataset[10]
# x_t = Variable(torch.from_numpy(x_t))
# u_t = Variable(torch.from_numpy(u_t))
# x_tp1 = Variable(torch.from_numpy(x_tp1))
# print(recon_x_t_np.shape)
# print(precit_x_tp1_np.shape)
# print(x_t_np.shape)
# print(x_t_next_np.shape)
# print(z_mean_numpy.shape)
# print(state_numpy_train.shape)
# print(z_mean_numpy[0:5])
# print(state_numpy_train[0:5])
#
cmrainbow = plt.get_cmap("rainbow")
fig1 = plt.figure(figsize=(12, 12))
ax1 = fig1.add_subplot(121, projection='3d')
p = ax1.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy_train[:, 0], cmap=cmrainbow)
fig1.colorbar(p)
ax1.set_title('latent embedding with state joint angle')

ax2 = fig1.add_subplot(122, projection='3d')
p = ax2.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy_train[:, 1], cmap=cmrainbow)
fig1.colorbar(p)
ax2.set_title('latent embedding with state joint velocity')

# fig4 = plt.figure(figsize=(10, 6))
# ax4 = fig1.add_subplot(111, projection='3d')
# p = ax4.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy_train[:, 1])
# fig4.colorbar(p)
# ax4.set_title('latent embedding with state joint velocity')
# plt.colorbar()
fig2 = plt.figure(figsize=(20, 30))

for i in range(3):

    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(recon_x_t_np[i * 10].reshape(48, 2 * 48), cmap="gray")
    plt.title("Reconstruct")
    plt.colorbar()
    plt.subplot(3, 2, 2*i + 2)
    plt.imshow(x_t_np[i * 10].reshape(48, 2 * 48) ,cmap="gray")
    plt.title("Training Input")
    plt.colorbar()
plt.tight_layout()
plt.show()

fig3 = plt.figure(figsize=(20, 30))
for i in range(3):

    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(precit_x_tp1_np[i * 10].reshape(48, 2 * 48), cmap="gray")
    plt.title("Prediction")
    plt.colorbar()
    plt.subplot(3, 2, 2*i + 2)
    plt.imshow(x_t_next_np[i * 10].reshape(48, 2 * 48),cmap="gray")
    plt.title("Training Input")
    plt.colorbar()
plt.tight_layout()
plt.show()

latent_dim = 3
model = E2C(48 * 48 * 2, latent_dim , 1, config='pendulum').cuda()
model_name = 'e2c_anneal_10_' + '_' + str(theta_speed) + '4'
path = 'models/' + model_name + '.pth'
model.load_state_dict(torch.load(path))

# for p in model.parameters():
#     p.requires_grad = False

# dataset = hkl.load('../data/triplets_5_train.hkl')
pendulum_dataset = GymPendulumDatasetV2('data/pendulum_markov_test')
test_loader = torch.utils.data.DataLoader(pendulum_dataset,
                                           batch_size=1,
                                           shuffle = False)

z_mean_numpy = np.zeros((len(test_loader), latent_dim), dtype=np.float32)
z_mean_tp1_numpy = np.zeros((len(test_loader), latent_dim), dtype=np.float32)
state_numpy = np.zeros((len(test_loader), 2), dtype=np.float32)
state_tp1_numpy = np.zeros((len(test_loader), 2), dtype=np.float32)
recon_x_t_np = np.zeros((len(test_loader), 48 * 48 * 2), dtype=np.float32)
precit_x_tp1_np = np.zeros((len(test_loader), 48 * 48 * 2), dtype=np.float32)
x_t_np = np.zeros((len(test_loader), 48 * 48 * 2), dtype=np.float32)
x_tp1_np = np.zeros((len(test_loader), 48 * 48 * 2), dtype=np.float32)


for batch_idx, data in enumerate(test_loader):
    model.eval()
    x_t, u_t, x_tp1, state_t, state_tp1 = data

    # put on GPU
    x_t = Variable((x_t / 255.).cuda())
    u_t = Variable(u_t.cuda())
    x_tp1 = Variable((x_tp1 / 255.).cuda())
    state_t = Variable(state_t.cuda())
    state_tp1 = Variable(state_tp1.cuda())

    z_mean = model.latent_embedding(x_t)
    z_mean_tp1 = model.latent_embedding(x_tp1)
    z_mean_numpy[batch_idx, :] = (z_mean.data).cpu().numpy()
    z_mean_tp1_numpy[batch_idx, :] = (z_mean_tp1.data).cpu().numpy()
    state_numpy[batch_idx, :] = (state_t.data).cpu().numpy()
    state_tp1_numpy[batch_idx, :] = (state_tp1.data).cpu().numpy()
    x_t_np[batch_idx, :] = (x_t.data).cpu().numpy()
    x_tp1_np[batch_idx, :] = (x_tp1.data).cpu().numpy()
    recon_x_t_np[batch_idx, :] = (model.embedding_to_sample(model.latent_embedding(x_t))).data.cpu().numpy()
    precit_x_tp1_np[batch_idx, :] = (model.embedding_to_sample(model.latent_embedding(model.predict(x_t, u_t))).data).cpu().numpy()

print z_mean_numpy.shape
print z_mean_tp1_numpy.shape
print state_numpy.shape
print state_tp1_numpy.shape
print x_t_np.shape
print x_tp1_np.shape
print recon_x_t_np.shape
print precit_x_tp1_np.shape
plt.figure(figsize=(20, 30))
for i in range(3):
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(recon_x_t_np[i].reshape(48, 2 * 48), cmap="gray")
    plt.title("Reconstruct")
    plt.colorbar()
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(x_t_np[i].reshape(48, 2 * 48), cmap="gray")
    plt.title("Training Input")
    plt.colorbar()
    plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 30))
for i in range(3):
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(precit_x_tp1_np[i].reshape(48, 2 * 48), cmap="gray")
    plt.title("Prediction")
    plt.colorbar()
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(x_tp1_np[i].reshape(48, 2 * 48), cmap="gray")
    plt.title("Training Input")
    plt.colorbar()
    plt.tight_layout()
plt.show()

cmrainbow = plt.get_cmap("rainbow")
fig5 = plt.figure(figsize=(12, 12))
ax5 = fig5.add_subplot(121, projection='3d')
p = ax5.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy[:, 0], cmap=cmrainbow)
fig5.colorbar(p)
ax5.set_title('latent embedding with state joint angle')

ax6 = fig5.add_subplot(122, projection='3d')
p = ax6.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy[:, 1], cmap=cmrainbow)
fig5.colorbar(p)
ax6.set_title('latent embedding with state joint velocity')
plt.show()