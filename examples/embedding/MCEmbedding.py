# %% [markdown]
# ## HTorch hyperbolic embedding for the WordNet Mammals

# %%
import timeit
import torch, HTorch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
from HTorch import MCHParameter, MCHTensor, MCTensor
from hype.graph import load_edge_list, eval_reconstruction
from HTorch.layers import HEmbedding
from HTorch.optimizers import RiemannianSGD, RiemannianAdam
import sys, os, random
import json
import torch.multiprocessing as mp
from hype.graph_dataset import BatchedDataset

# %%
class MCHEmbedding(torch.nn.Embedding):
    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, nc=1, **kwargs):
        super(MCHEmbedding, self).__init__(*args, **kwargs)
        self.weight = MCHParameter(
            self.weight, manifold=manifold, curvature=curvature, nc=nc)
        self.weight.init_weights()
    
    def forward(self, input):
        # import pdb; pdb.set_trace()
        tmp = super().forward(input)
        # tmp = self.weight[input]
        output = tmp.as_subclass(MCHTensor)
        output.res = tmp.res
        output._nc = tmp._nc
        output.manifold = self.weight.manifold
        output.curvature = self.weight.curvature
        return output

# overwrite torch.nn.functional.embedding
# %%
# model defined using HTorch
class MCEnergyFunction(torch.nn.Module):
    def __init__(self, size, dim, sparse=False, manifold='PoincareBall', curvature=-1.0, nc=1, **kwargs):
        super().__init__()
        # initialize layer, weights are automatically initialized around origin
        self.lt = MCHEmbedding(size, dim, sparse=sparse, manifold=manifold, curvature=curvature, nc=nc) 
        # self.lt = HEmbedding(size, dim, sparse=sparse, manifold=manifold, curvature=curvature) 
        self.nobjects = size

    def forward(self, inputs):
        e = self.lt(inputs)
        with torch.no_grad():
            e.proj_()
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        return o.Hdist(s).squeeze(-1)

    def loss(self, inp, target, **kwargs):
        return F.cross_entropy(inp.neg(), target)

# %%
# set meta-parameters, float precision etc.
os.environ["NUMEXPR_MAX_THREADS"] = '8'

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# note d16, d32 may produce infs and NaNs due to imprecision
d16 = torch.float16; d32 = torch.float32; d64 = torch.float64
cpu = torch.device("cpu"); gpu = torch.device(type='cuda', index=0)
device = cpu
opt_dtype = d64

if opt_dtype == d16:
    dtype = "d16"
    torch.set_default_tensor_type('torch.HalfTensor')
elif opt_dtype == d32:
    dtype = "d32"
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    dtype = "d64"
    torch.set_default_tensor_type('torch.DoubleTensor')
    
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ### Hyperparameters for PyTorch Poincare Halfspace model: (now use opt_epochs = 20)

# %%
## parameters; these are global in the notebook!
opt_maxnorm = 500000; opt_debug = False;
opt_dim = 2;
opt_negs = 50;  opt_eval_each = 20;
opt_sparse = True; opt_ndproc = 1;  opt_burnin = 20;
opt_dampening = 0.75; opt_neg_multiplier = 1.0; 
opt_burnin_multiplier = 0.01; 
###########################################################
opt_epochs = 2000; opt_batchsize = 32; 
opt_lr = 2.0;  opt_dscale = 0.3
#opt_manifold = "PoincareBall"
# opt_manifold = "Lorentz"
opt_manifold = "HalfSpace"
opt_curvature = -1.0
opt_task = 'mammals'
#######################################
nc = 2;
#######################################
FILE_NAME = "_".join([opt_task, 'lr', str(opt_lr), 'batch', str(opt_batchsize),
                      str(opt_epochs), "torch", dtype, str(opt_dscale)])

# %% [markdown]
# ### Initializing logging and data loading

# %%
log_level = logging.DEBUG if opt_debug else logging.INFO
log = logging.getLogger('Embedding')
logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
log.info('Using edge list dataloader')
idx, objects, weights = load_edge_list("wordnet/mammal_closure.csv", False) 
#idx, objects, weights = load_edge_list("/home/jl3789/Hyperbolic_Library/applications/poincare_embedding/wordnet/mammal_closure.csv", False) 
torch.autograd.anomaly_mode.set_detect_anomaly(True)
# %% [markdown]
# ### Initializing model

# %%
def init_model(manifold, curvature, dim, idx, objects, weights, nc=1, sparse=True):
    model_name = '%s_dim%d'
    mname = model_name % (manifold, dim)
    data = BatchedDataset(idx, objects, weights, opt_negs, opt_batchsize,
        opt_ndproc, opt_burnin > 0, opt_dampening)
    model = MCEnergyFunction(len(data.objects), opt_dim, sparse=sparse, manifold=manifold, curvature=curvature, nc=nc)
    data.objects = objects
    return model, data, mname

def adj_matrix(data):
    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}
    return adj

# %% [markdown]
# ### Training

# %%
def data_loader_lr(data, epoch, progress = False):
    data.burnin = False 
    lr = opt_lr
    if epoch < opt_burnin:
        data.burnin = True
        lr = opt_lr * train._lr_multiplier
    loader_iter = tqdm(data) if progress else data
    return loader_iter, lr

# %%
def train(device, model, data, optimizer, progress=False):
    epoch_loss = torch.Tensor(len(data))
    LOSS = np.zeros(opt_epochs)
    for epoch in range(opt_epochs):
        epoch_loss.fill_(0)
        t_start = timeit.default_timer()
        torch.autograd.anomaly_mode.set_detect_anomaly(True)
        # handling burnin, get loader_iter and learning rate
        loader_iter, lr = data_loader_lr(data, epoch, progress=progress)
        for i_batch, (inputs, targets) in enumerate(loader_iter):
            elapsed = timeit.default_timer() - t_start
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs) * opt_dscale
            # loss = torch.norm(preds)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            print(f'iter {i_batch}, loss {loss.item()}')
            # optimizer.step()
            optimizer.step(lr=lr)
            epoch_loss[i_batch] = loss.cpu().item()
            with torch.no_grad():
                loss = model.loss(preds.to(d64), targets, size_average=True)
                epoch_loss[i_batch] = loss.cpu().item()

        LOSS[epoch] = torch.mean(epoch_loss).to(d64).item()
        # since only one thread is used:
        log.info('json_stats: {' f'"epoch": {epoch}, '
                 f'"elapsed": {elapsed}, ' f'"loss": {LOSS[epoch]}, ' '}')
    return LOSS


# %% [markdown]
# # Training embedding

# %%
# setup model
seed_everything(1)
model, data, model_name = init_model(opt_manifold, opt_curvature, opt_dim, idx, objects, weights, sparse=opt_sparse, nc=nc)
data.neg_multiplier = opt_neg_multiplier
train._lr_multiplier = opt_burnin_multiplier
model = model.to(device)
print('the total dimension', model.lt.weight.data.size(-1))
print(">>>>>> # Tensor# | dtype is:", model.lt.weight.dtype, "| device is:", model.lt.weight.device)
# setup optimizer, both works, though a small lr should be used for RiemannianAdam (which is not tuned yet)
#optimizer = RiemannianAdam(model.parameters(), lr=opt_lr)
optimizer = RiemannianSGD(model.parameters(), lr=opt_lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=opt_lr)
# get adjacency matrix
adj = adj_matrix(data)
# begin training
start_time = timeit.default_timer()
loss = train(device, model, data, optimizer, progress=False)
train_time = timeit.default_timer() - start_time
print("Total training time is:", train_time)

# %% [markdown]
# # Evaluate embedding

# %%
class RES():
    """for logging results"""
    def __init__(self, loss, eval_res, weight):
        self.loss = torch.tensor(loss, dtype=torch.float64, 
                                 device=cpu)
        self.eval_res = torch.tensor(eval_res, dtype=torch.float64, 
                                     device=cpu)
        self.weight = weight

# %%
model_weight = model.lt.weight.clone()
# eval meanrank, maprank in the original model
meanrank, maprank = eval_reconstruction(adj, model_weight, workers=opt_ndproc)
if opt_manifold != "PoincareBall":
    # change to PoincareBall to derive the sqnorms metric
    model_weight = model_weight.to_other_manifold("PoincareBall")
sqnorms = torch.sqrt(torch.sum(torch.pow(model_weight, 2), dim=-1))
sqnorm_min = sqnorms.min().item()
sqnorm_avg = sqnorms.mean().item()
sqnorm_max = sqnorms.max().item()
eval_res = [meanrank, maprank, sqnorm_min, sqnorm_avg, sqnorm_max, train_time]
RESULTS = RES(loss, eval_res, model_weight)
# torch.save(RESULTS, "./results_weights/"+FILE_NAME+"_seed1"+ ".pt")
log.info(
    'json_stats final test: \n{'
    f'"sqnorm_min": {round(sqnorm_min,6)}, '
    f'"sqnorm_avg": {round(sqnorm_avg,6)}, '
    f'"sqnorm_max": {round(sqnorm_max,6)}, \n'
    f'"mean_rank": {round(meanrank,6)}, '
    f'"map": {round(maprank,6)}, '
    '}'
)

# %%



