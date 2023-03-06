import pickle
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from scipy.integrate import odeint
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"

# example code for active learning

# folder name you are saving to; functions as a prefix for all file names
folder = 'example_folder_ChimeraKuramoto'

# this class if for establishing the NN architecture


class autoencoder(nn.Module):
    def __init__(self):

        super(autoencoder, self).__init__()

        # how many latent dims?
        self.latent_dimnum = 2
        # how many filters?
        self.filter_num = 2

        # dilation factor to make input and output match dimensions
        self.factor = (bin_num+2)//3

        # the encoder is a 2D convolutional neural network
        # fed through a ReLU nonlinearity
        # and then flattened to make a single vector
        self.encoder_base = nn.Sequential(
            nn.Conv2d(1, self.filter_num, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.Flatten(start_dim=1)
        )

        # this vector is the input for two linear transformations
        # one transformation gives a latent dim-dimensional mean
        # the other gives a ld-dimensional (log)-variance
        self.mu_head = nn.Linear(
            self.filter_num*self.factor**2, self.latent_dimnum)
        self.logvar_head = nn.Linear(
            self.filter_num*self.factor**2, self.latent_dimnum)

        # the decoder receives a vector sampled from the latent distribution
        # expands it through a linear transformation
        # and undoes the flattening and convolution
        # before a sigmoid nonlinearity - because we want the output to be between 0/1
        self.decoder_base = nn.Sequential(
            nn.Linear(self.latent_dimnum, self.filter_num*self.factor**2),
            nn.Unflatten(dim=1, unflattened_size=(
                self.filter_num, self.factor, self.factor)),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.filter_num, 1, 3,
                               stride=3, padding=1),  # b, 8, 15, 15
            nn.Sigmoid()
        )

    # this is the full encoder
    def encoder(self, x):
        x = self.encoder_base(x)
        return self.mu_head(x), self.logvar_head(x)

    # repar allows us to sample from the latent dim
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # full decoder
    def decoder(self, z):
        return self.decoder_base(z)

    # full chain for sampling from latent space
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar

# this function constructs the training loss of a variational autoencoder


def training_loss(recon_x, mu, logvar, x):
    # BCE penalizes mismatch between input/output
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KLD tries to make the latent space distribution gaussian
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + 1*KLD)/x.shape[0]

# This function trains the model based on the subset of data it's given


def training_round(model, training_loss, optimizer, subset, num_epochs):

    total_loss = 0
    for epoch in range(num_epochs):

        img = torch.from_numpy(subset[:, None, :, :].astype(np.float32))
        img = Variable(img).cuda()
        # ===================forward=====================
        output, mu, logvar = model(img.cuda())
        loss = training_loss(output, mu, logvar, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        if epoch + 1 == num_epochs:
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch+1, num_epochs, loss))

# three functions for visualizing; mostly deprecated
# the first is for visualizing the "state" i.e. the representation of the dynamics
# that we hand to the autoencoder


def visualize_element(state):

    plt.figure()
    plt.imshow(state, cmap='Greys', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

# this visualizes the output of the autoencoder given a state


def visualize_output(state, model):

    output = model(prepare_array_for_network(state).cuda())
    plt.figure()
    plt.imshow(output[0].cpu().detach().numpy()[0]
               [0], cmap='Greys', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

# visualizes state and output for ease of comparison


def compare_io(state, model):

    visualize_element(state)
    visualize_output(state, model)

# this function is for doing a full rollout of a parameter
# it accepts a parameter, and returns a timeseries and the state construction of that series


def rollout_parameter(param, dt, timesteps, N, initial_angles, natfreqs):

    # this function computes the time derivative of all the oscillators
    def kuramoto_derivative(angles_vec, t):

        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        dxdt = natfreqs + (weight_mat * np.sin(angles_j -
                           angles_i - alpha)).sum(axis=0)
        return dxdt

    # constructing by hand a more convenient representation of the Kuramoto parameters
    # taken from the raw parameters that are passed
    alpha = param[-1]
    p = np.ones((N_type, N_type))
    p[0, 0] = 1. - param[0]
    p[1, 1] = 1. - param[0]
    p[0, 1] = param[0]
    p[1, 0] = param[0]
    weight_mat = np.zeros((N, N))
    for i in np.arange(len(param)):
        for j in np.arange(len(param)):
            weight_mat[(N//N_type)*i:, (N//N_type)*j:] = p[i, j]/(N//N_type)

    # here, we actually perform the integration of the kuramoto model
    t = np.linspace(0, dt*timesteps, timesteps)
    timeseries = odeint(kuramoto_derivative, initial_angles, t)

    # downsampling to begin state construction - note that we had to choose
    # s_gap, a frequency of sampling
    downsample_series = timeseries[-samples*s_gap::s_gap] % (2*np.pi)

    # function that processes line by line the down-sampled timeseries
    def f(x):

        # find the average theta
        avg_theta = np.arctan2(np.sum(np.sin(x)), np.sum(np.cos(x)))
        # recenter around the average theta and take the sine
        y = np.sin((x-avg_theta) % (2*np.pi))
        # construct a histogram of sine values
        h = np.histogram(y, bins=bin_num, range=(-1.1, 1.1))[0]/N
        return h

    state = np.apply_along_axis(f, 1, downsample_series)

    # return both the full dynamics and the constructed state
    return timeseries, state

# initialize the whole process with randomly sampled parameters


def initialize_history(model, Ninit, dt, timesteps, N):

    h = []
    a = []

    initial_parameters = np.random.uniform(lb, ub, size=((Ninit, param_size)))
    for p_orig in initial_parameters:
        p = mutate_parameter(p_orig, mu_step)
        timeseries, state = rollout_parameter(
            p, dt, timesteps, N, initial_angles, natfreqs)
        gpu_state = prepare_array_for_network(state).cuda()
        representation = model.encoder(gpu_state)[0]
        output = model.decoder(representation)
        h.append([p, state, representation.cpu().detach().numpy()[0],
                  criterion(output, gpu_state).data])  # .cpu().detach().numpy()])

    return h, a

# this function performs "novelty sampling" by finding parameters
# which the autoencoder has the greatest error on
# mostly unused


def sample_parameter(history):

    errors = np.array(np.array(history)[:, 3], dtype=np.float32)
    param_index = np.argmax(np.array(errors))
    param = history[param_index][0]

    return param, param_index

# sampling algorithm which uniformly samples latent space values


def sample_parameter_uniform(history):

    reps = []
    for item in history:

        reps.append(item[2].astype(float))
    reps = np.array(reps)

    goal = []
    for i in np.arange(len(reps.T)):

        upper = np.max(reps[:, i])
        lower = np.min(reps[:, i])
        size = upper-lower
        center = (upper+lower)*.5

        sample = np.random.uniform(center-size/2, center+size/2)
        goal.append(sample)
    goal = np.array(goal)

    distance_to_goal = np.linalg.norm(reps-goal, axis=1)
    closest_entry = np.argmin(distance_to_goal)
    param = history[closest_entry][0]

    return param, closest_entry

# once a parameter is chose, we mutate it
# if the mutation exceeds the parameter bounds, we resample


def mutate_parameter(p, mu_step):

    new_p = p + mu_step*np.random.uniform(-1., 1., size=param_size)

    while np.any(new_p < lb) or np.any(new_p > ub):

        new_p = p + mu_step*np.random.uniform(-1., 1., size=param_size)

    return new_p

# selects training subset, weighting more recent regions of exploration


def select_training_data(history, batch_size):

    n_data = len(history)

    inds = np.random.choice(n_data-batch_size//2, batch_size//2, replace=False)
    inds = np.concatenate(
        (inds, n_data-batch_size//2+np.arange(batch_size//2)))

    samples = []
    for i in inds:

        samples.append(history[i][1])

    return np.array(samples)

# converting between pytorch variables and numpy variables


def prepare_array_for_network(img):

    return torch.from_numpy(img[None, None, :, :].astype(np.float32))


# defining hyperparameters for the training
# how many training epochs when we update the latent space
num_epochs = 2000
# how many states are passed for training
batch_size = 200
# learning rate for training
learning_rate = 1e-3
# total number of samples that are going to be aquired (in addition to the initial ones)
num_rounds = 1400
# how often we update the latent space (measured in terms of samples)
train_interval = batch_size//2

# these are simulation hyperparameters
dt = .05
# how much many timesteps we simulate for (total time is dt*timesteps)
timesteps = 15000
# how many oscillators?
N = 16*2
# how many bins in the state construction histograms
bin_num = 7
# how many families of oscillators
N_type = 2
# dimensionality of parameter space
param_size = 1+1
# upper and lower bounds on parameter space
ub = np.array([1., np.pi/2])
lb = 0*np.ones(param_size)
# how many timesteps do we sample when constructing the state from the timeseries?
samples = bin_num
# gap between timeseries sampling
s_gap = int(timesteps/32/samples)
# how big are mutation steps?
mu_step = (ub-lb)/10

# defining and initializing the learning structures
# initializes the NN
model = autoencoder().cuda()
# criterion is only relevant for novelty detection - ignore for now
criterion = nn.L1Loss()
# what optimization algorithm will be used to train the NN?
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

# initializing GLOBAL VARIABLES for the simulation
initial_angles = 2 * np.pi * np.random.random(size=N)
natfreqs = np.random.normal(size=N, scale=.1)*0
np.save(folder+'_initial_angles.npy', initial_angles)
np.save(folder+'_natfreqs.npy', natfreqs)

# initialize history!
history, _ = initialize_history(model, batch_size, dt, timesteps, N)
# also we want to record which points were sampled from, pre-mutation
parent_record = []

# Ok, here's the actual algorithm in action now
for i in np.arange(num_rounds):

    # this does the sampling
    p_orig, parent = sample_parameter_uniform(history)
    parent_record.append(parent)
    p = mutate_parameter(p_orig, mu_step)
    timeseries, state = rollout_parameter(
        p, dt, timesteps, N, initial_angles, natfreqs)
    gpu_state = prepare_array_for_network(state).cuda()
    rep = model.encoder(gpu_state)[0]
    output = model.decoder(rep)
    # everything we do gets recorded into the history
    # p is the parameter we sampled
    # state is the state we construction from the timeseries
    # rep is the latent space representation that the state got mapped to
    # criterion is important for novelty detection - you can ignore this for now
    history.append([p, state, rep.cpu().detach().numpy()[0],
                    criterion(output, gpu_state).data])
    print(i)

    # this actually does the training
    if i % train_interval == 0:

        data_subset = select_training_data(history, batch_size)
        training_round(model, training_loss, optimizer,
                       data_subset, num_epochs)

        # note that we recompute every entry in the history
        # whenever we retrain the network
        for item in history:
            state = item[1]
            gpu_state = prepare_array_for_network(state).cuda()
            rep = model.encoder(gpu_state)[0]
            output = model.decoder(rep)
            item[2] = rep.cpu().detach().numpy()[0]
            # .cpu().detach().numpy()
            item[3] = criterion(output, gpu_state).data

    # we need to save things in case something goes awry
    if i % 200 == 0:  # saving time

        torch.save(model.state_dict(), folder+'_weights.pt')
        with open(folder+'_history.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# OK that's it!! simulation done. now time for some preliminary analysis + visualization

# let's extract everything out from the history and make numpy arrays of it - easier that way
latent_space = []
collection = []
param = []
for h in history:
    param.append(h[0])
    collection.append(h[1])
    latent_space.append(h[2])
param = np.array(param)
latent_space = np.array(latent_space)

# another hyperparameter - maximum number of phases we'll allow ourselves to identify
phase_num_ub = 6

# construct cmap for nice colors during plotting
flatui = ["#5eccab", "#c0c0c0", "#8c9fb7", "#984464", "#796880",  "#00678a"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

# perform agglomerative clustering to identify which places in
# latent space are close to each other
agg = AgglomerativeClustering(linkage='ward', n_clusters=phase_num_ub)
A = agg.fit_predict(latent_space)

# also do PCA on latent space - not really necessary when latent dim = 2
# but it does help us figure out if one axis is doing all the work
pca = PCA()
X = pca.fit_transform(latent_space)

# visualize latent space
plt.figure()
plt.scatter(latent_space[:, np.argmax(np.abs(pca.components_[0]))],
            latent_space[:, np.argmax(np.abs(pca.components_[1]))], c=A, cmap=my_cmap, alpha=.6)
plt.colorbar(ticks=np.arange(phase_num_ub))
plt.ylim([-.1, .1])
plt.xlabel('latent dimension 1')
plt.ylabel('latent dimension 2')
plt.savefig(folder+'_latent_space.png', format='png', bbox_inches="tight")
plt.savefig(folder+'_latent_space.pdf', format='pdf', bbox_inches="tight")
plt.close()

# visualize parameter space - we can do this because param_dim = 2
# can't always do this
plt.figure()
plt.scatter(np.pi/2-param[:, 1], 1-2*param[:, 0], alpha=.6, c=A, cmap=my_cmap)
plt.colorbar(ticks=np.arange(phase_num_ub))
plt.xlabel('phase offset')
plt.ylabel('cross-coupling strength')
plt.savefig(folder+'_param_space.png', format='png', bbox_inches="tight")
plt.savefig(folder+'_param_space.pdf', format='pdf', bbox_inches="tight")
plt.close()

# zoom in on the area that theoretically should be interesting
plt.figure()
plt.scatter(np.pi/2-param[:, 1], 1-2*param[:, 0], alpha=.6, c=A, cmap=my_cmap)
plt.xlabel('phase offset')
plt.ylabel('cross-coupling strength')
plt.xlim([0, .25])
plt.ylim([0, .5])
plt.colorbar(ticks=np.arange(phase_num_ub))
plt.savefig(folder+'_param_space_zoom.png', format='png', bbox_inches="tight")
plt.savefig(folder+'_param_space_zoom.pdf', format='pdf', bbox_inches="tight")
plt.close()

# Last thing to do - check to see what the dynamics of the phases actually look like
for i in np.arange(phase_num_ub):

    # to do this, we find all latent space entires which correspond to a given phase
    phase = np.where(A == i)[0]
    phase_points = []
    for phase_p in phase:
        phase_points.append(history[phase_p][2])
    phase_points = np.array(phase_points)

    # since we clustered in latent space, it makes sense to look at the
    # entry which is closest to the median of the cluster
    phase_mean = np.median(phase_points, axis=0)
    dist_to_mean = np.linalg.norm(phase_points-phase_mean, axis=1)
    closest_entry = np.argmin(dist_to_mean)
    phase_rep = phase[closest_entry]

    # we then visualize the output of the parameters at that "representative"
    # parameter value in more detail
    hidden_answer, state = rollout_parameter(
        history[phase_rep][0], dt, timesteps, N, initial_angles, natfreqs)
    # we compute the phase coherence of all clusters
    # just among family A
    norm_A = np.abs(np.mean(np.exp(1j*hidden_answer[:, 0:N//2]), axis=1))
    # just among family B
    norm_B = np.abs(np.mean(np.exp(1j*hidden_answer[:, N//2:]), axis=1))
    # and then among all the clusters
    norm_tot = np.abs(np.mean(np.exp(1j*hidden_answer), axis=1))
    # then we plot!
    plt.figure()
    plt.plot(norm_A, label='group A')
    plt.plot(norm_B, label='group B')
    plt.plot(norm_tot, label='all')
    plt.ylim([-0.1, 1.1])
    plt.xlabel('timesteps')
    plt.ylabel('oscillator coherence')
    plt.legend()
    plt.savefig(folder+'_phase_rep_norm'+str(i)+'.png',
                format='png', bbox_inches="tight")
    plt.savefig(folder+'_phase_rep_norm'+str(i)+'.pdf',
                format='pdf', bbox_inches="tight")
    plt.close()
