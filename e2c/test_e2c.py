from e2c import *
from configs import LidarEncoder
from torch.autograd import Variable
from utils import *
import torch.optim as optim
from collections import OrderedDict
import pdb
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from e2c import *
from datasets import *


# KL ANNEAL
theta = 0               # theta start
theta_speed = 0.0001     # increase per epoch
theta_max = .01          # theta end
latent_dim = 3
model = E2C(48 * 48 * 2, latent_dim , 1, config='pendulum').cuda()
model_name = 'e2c_anneal_10_' + '_' + str(theta_speed) + '99'
path = 'models/' + model_name + '.pth'
model.load_state_dict(torch.load(path))

# for p in model.parameters():
#     p.requires_grad = False

# dataset = hkl.load('../data/triplets_5_train.hkl')
pendulum_dataset = GymPendulumDatasetV2('data/pendulum_markov_train')
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
    recon_x_t_np[batch_idx, :] = (model.embedding_to_sample(z_mean)).data.cpu().numpy()
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
    plt.imshow(x_t_np[i].reshape(48, 2 * 48) / 255, cmap="gray")
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
    plt.imshow(x_tp1_np[i].reshape(48, 2 * 48) / 255, cmap="gray")
    plt.title("Training Input")
    plt.colorbar()
    plt.tight_layout()
plt.show()
#
# # state_numpy[:, 0] = (state_numpy[:, 0] - np.min(state_numpy[:, 0]))
# p = ax1.scatter(z_mean_numpy[:, 0], z_mean_numpy[:, 1], z_mean_numpy[:, 2], c=state_numpy[:, 0], vmin=0, vmax=2 * np.pi)
# fig.colorbar(p)
# ax1.set_title('latent embedding with state joint angle')
# # plt.colorbar()
# # ax1.legend()
#     # ax2.scatter(m.series[i].X_mean.value[:,0], m.series[i].X_mean.value[:,1], m.series[i].X_mean.value[:,2], color=c, label=i)
#     # ax2.scatter(m.X_variational_mean.value[labels==i, 0], m.X_variational_mean.value[labels==i, 1], m.X_variational_mean.value[labels==i, 2], color=c, label=i)
#     # ax2.set_title('Bayesian GPLVM')
#     # ax2.legend()
# plt.show()
# num_resolution = 50
# th = np.linspace(-3, 3, num_resolution)
# thdot = np.linspace(-10, 10, num_resolution)
# z = np.zeros((num_resolution, num_resolution))
# th_axis, thdot_axis = np.meshgrid(np.linspace(-3, 3, num_resolution), np.linspace(-10, 10, num_resolution))
# for i in range(num_resolution):
#     for j in range(num_resolution):
#         z[i, j] = i * 100 + j
#
# # z = ndimage.rotate(z, 45)
# # print z
# # z = np.array([i for i in range(0, 10000)])
# # z = z.reshape(100, 100)
# fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(111)
# ax.set_title("X vs Y AVG",fontsize=14)
# ax.set_xlabel("XAVG",fontsize=12)
# ax.set_ylabel("YAVG",fontsize=12)
# # print th_axis
# # print thdot_axis
# # display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar')
# # display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to
# #                                   ## multiply the values such that 1 become 2*pi
# #                                   ## this field is supposed to take values 1 or -1 only!!
#
# norm = colors.Normalize(0.0, 2*np.pi)
#
# # Plot the colorbar onto the polar axis
# # note - use orientation horizontal so that the gradient goes around
# # the wheel rather than centre out
# quant_steps = 2056
# # cb = matplotlib.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap('hsv',quant_steps),
# #                                    norm=norm,
# #                                    orientation='horizontal')
#
# # # # scatter with colormap mapping to z value
# # colors1 = plt.cm.Greens(np.linspace(0.5, 1, 2500))
# # colors2 = plt.cm.Blues(np.linspace(0.5, 1, 2500))
# # # colo3 = plt.cm.
# # print colors1.shape
# #
# # colors = np.vstack((colors1, colors2))
# print colors
# mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
# ax.scatter(th_axis,thdot_axis,c=z, marker = 'o', cmap=cm.get_cmap('hsv',quant_steps), norm = colors.Normalize(0.0, 2.0 * np.pi))
# plt.show()
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # H = np.array([[1, 2, 3, 4],
# #               [5, 6, 7, 8],
# #               [9, 10, 11, 12],
# #               [13, 14, 15, 16]])  # added some commas and array creation code
# #
# # fig = plt.figure(figsize=(6, 3.2))
# #
# # ax = fig.add_subplot(111)
# # ax.set_title('colorMap')
# # plt.imshow(z)
# # ax.set_aspect('equal')
# #
# # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# # cax.get_xaxis().set_visible(False)
# # cax.get_yaxis().set_visible(False)
# # cax.patch.set_alpha(0)
# # cax.set_frame_on(False)
# # plt.colorbar(orientation='vertical')
# # plt.show()
#
#
#
# # plt.show()
# # fig = plt.figure(figsize=(10, 10))
# #
# # ax = fig.add_subplot(111)
# # ax.set_title('colorMap')
# # plt.imshow(H)
# # ax.set_aspect('equal')
# #
# # # plt.plot(th, thdot, lw=0.5, alpha=0.5, cmap='rainbow')
# # # plt.show()
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # H = np.array([[1, 2, 3, 4],
# #               [5, 6, 7, 8],
# #               [9, 10, 11, 12],
# #               [13, 14, 15, 16]])  # added some commas and array creation code
# #
# # fig = plt.figure(figsize=(6, 3.2))
# #
# # ax = fig.add_subplot(111)
# # ax.set_title('colorMap')
# # plt.imshow(H)
# # ax.set_aspect('equal')
# #
# # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# # cax.get_xaxis().set_visible(False)
# # cax.get_yaxis().set_visible(False)
# # cax.patch.set_alpha(0)
# # cax.set_frame_on(False)
# # plt.colorbar(orientation='vertical')
# plt.show()