import torch
import torch.nn.functional as F
from HTorch.HTensor import HParameter, HTensor
from HTorch.MCHTensor import MCHParameter, MCHTensor

class HEmbedding(torch.nn.Embedding):
    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        super(HEmbedding, self).__init__(*args, **kwargs)
        self.weight = HParameter(self.weight, manifold=manifold, curvature=curvature)
        self.weight.init_weights()
    
    def forward(self, input):
        output = super().forward(input).as_subclass(HTensor)
        output.manifold = self.weight.manifold
        output.curvature = self.weight.curvature
        return output
    
class MCHEmbedding(torch.nn.Embedding):
    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, nc=1, **kwargs):
        super(MCHEmbedding, self).__init__(*args, **kwargs)
        self.weight = MCHParameter(self.weight, manifold=manifold, curvature=curvature, nc=nc)
        self.weight.init_weights()
    
    def forward(self, input):
        tmp = super().forward(input)
        print(tmp.shape)
        output = tmp[..., 0].as_subclass(MCHTensor)
        tmp.manifold = self.weight.manifold
        tmp.curvature = self.weight.curvature
        return tmp