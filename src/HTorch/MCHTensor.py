from __future__ import annotations
import torch, math
from HTorch.manifolds import Euclidean, PoincareBall, Lorentz, HalfSpace, Manifold, Sphere
from torch import Tensor, device, dtype, Size
from torch.nn import Parameter
import functools
from typing import Union
from HTorch.MCTensor import MCTensor

manifold_maps = {
    'Euclidean': Euclidean, 
    'PoincareBall': PoincareBall,
    'Lorentz': Lorentz, 
    'HalfSpace': HalfSpace,
    'Sphere':Sphere
}
__all__ = [
    'MCHTensor',
    'MCHParameter'
]


class MCHTensor(MCTensor):
    @staticmethod
    def __new__(cls, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        ret = super().__new__(cls, *args, **kwargs)
        if isinstance(manifold, str):
            ret.manifold: Manifold = manifold_maps[manifold]()
        elif isinstance(manifold, Manifold):
            ret.manifold: Manifold = manifold
        else:
            raise NotImplemented
        ret.curvature = curvature
        return ret

    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(manifold, str):
            self.manifold: Manifold = manifold_maps[manifold]()
        elif isinstance(manifold, Manifold):
            self.manifold: Manifold = manifold
        else:
            raise NotImplemented
        self.curvature = curvature

    def __repr__(self):
        return "{}, manifold={}, curvature={}".format(
            super().__repr__(), self.manifold.name, self.curvature)

    def to_other_manifold(self, name: str) -> MCHTensor:
        """Convert to the same point on the other manifold."""
        assert name != self.manifold.name
        if name == 'Lorentz':
            ret = self.manifold.to_lorentz(self, abs(self.curvature))
        elif name == 'HalfSpace':
            ret = self.manifold.to_halfspace(self, abs(self.curvature))
        elif name == 'PoincareBall':
            ret = self.manifold.to_poincare(self, abs(self.curvature))
        else:
            raise NotImplemented
        ret.manifold = manifold_maps[name]()
        return ret

    def norm(self, p:int=2, dim:int=-1, keepdim=False) -> Tensor:
        """Returns p-norm as Tensor type"""
        return torch.norm(self, dim=dim, p=p, keepdim=keepdim).as_subclass(Tensor)

    def origin(self, d:int, c:Union[float,Tensor], size:Size=None, 
               device:device= None, dtype:dtype=None) -> Tensor:
        """The origin in the manifold"""
        res = self.manifold.origin(d, c, size, device, dtype)
        return MCHTensor(res, manifold=self.manifold, curvature=self.curvature, nc=self.nc)

    def Hdist(self, other: MCHTensor) -> Tensor:
        """Computes hyperbolic distance to other."""
        assert self.curvature == other.curvature, "Inputs should in models with same curvature!"
        if self.manifold.name == other.manifold.name:
            dist = self.manifold.distance(self, other, abs(self.curvature))
        else:
            #### transform to a self's manifold, combine with lazy evaulation?
            other_ = other.to_other_manifold(self.manifold.name)
            dist = self.manifold.distance(self, other_, abs(self.curvature))
        return dist.as_subclass(Tensor)

    def proj(self) -> MCHTensor:
        """Projects point p on the manifold."""
        return self.manifold.proj(self, abs(self.curvature))

    def proj_(self) -> MCHTensor:
        """Projects point p on the manifold."""
        return self.copy_(self.proj())

    def proj_tan(self, u: Tensor) -> Tensor:
        """Projects u on the tangent space of p."""
        return self.manifold.proj_tan(self, u, abs(self.curvature)).as_subclass(Tensor)

    def proj_tan0(self, u: Tensor) -> Tensor:
        """Projects u on the tangent space of the origin."""
        return self.manifold.proj_tan0(u, abs(self.curvature)).as_subclass(Tensor)

    def expmap(self, x: MCHTensor, u: Tensor) -> MCHTensor:
        """Exponential map."""
        return self.manifold.expmap(x, u, abs(self.curvature))

    def expmap0(self, u: Tensor) -> MCHTensor:  # wrap u to MCHTensor???
        """Exponential map, with x being the origin on the manifold."""
        res = self.manifold.expmap0(u, abs(self.curvature))
        return MCHTensor(res, manifold=self.manifold, curvature=self.curvature, nc=self.nc)

    def logmap(self, x: MCHTensor, y: MCHTensor) -> Tensor:
        """Logarithmic map, the inverse of exponential map."""
        return self.manifold.logmap(x, y, abs(self.curvature)).as_subclass(Tensor)

    def logmap0(self, y: MCHTensor) -> Tensor:
        """Logarithmic map, where x is the origin."""
        return self.manifold.logmap0(y, abs(self.curvature)).as_subclass(Tensor)

    def mobius_add(self, x: MCHTensor, y: MCHTensor, dim: int = -1) -> MCHTensor:
        """Performs hyperboic addition, adds points x and y."""
        return self.manifold.mobius_add(x, y, abs(self.curvature), dim=dim)

    def mobius_matvec(self, m: Tensor, x: MCHTensor) -> MCHTensor:
        """Performs hyperboic martrix-vector multiplication to m (matrix)."""
        return self.manifold.mobius_matvec(m, x, abs(self.curvature))

    def check_(self) -> Tensor:
        """Check if point on the specified manifold, project to the manifold if not."""
        check_result = self.manifold.check(
            self, abs(self.curvature)).as_subclass(Tensor)
        if not check_result:
            print('Warning: data not on the manifold, projecting ...')
            self.proj_()
        return check_result

    @staticmethod
    def find_mani_cur(args):
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                # Recursively apply the function to each element of the list
                manifold, curvature = MCHTensor.find_mani_cur(arg)
                break
            elif isinstance(arg, MCHTensor):
                manifold, curvature = arg.manifold, arg.curvature
                break
        return manifold, curvature

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        tmp = super().__torch_function__(func, types, args, kwargs)
        if type(tmp) in [MCTensor, MCHTensor] and not hasattr(tmp, 'manifold'):
            # ret = cls(tmp)
            ret = MCHTensor(tmp)
            ret._nc, ret.res = tmp.nc, tmp.res
            ret.manifold, ret.curvature = cls.find_mani_cur(args)
            return ret
        return tmp


class MCHParameter(MCHTensor, Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization with MCTensor arithmetic
    """
    @staticmethod
    def __new__(cls, data, manifold='PoincareBall', curvature=-1.0, nc=1, requires_grad=True):
        res = MCHTensor._make_subclass(cls, data, requires_grad)
        return res

    def __init__(self, x, manifold='PoincareBall', curvature=-1.0, nc=1, dtype=None, device=None):
        if isinstance(x, MCHTensor):
            self.manifold = x.manifold
            self.curvature = x.curvature
            self._nc = x._nc
            self.res = x.res.clone()
        elif isinstance(x, MCTensor):
            self.manifold = manifold_maps[manifold]()
            self.curvature = curvature
            self._nc = x._nc
            self.res = x.res.clone()
        else:
            self.manifold = manifold_maps[manifold]()
            self.curvature = curvature
            self._nc = nc
            self.res = torch.zeros(x.size() + (nc-1,),
                                   dtype=dtype, device=device)

    def init_weights(self, irange=1e-5):
        # this irange need to be controled for different floating-point precision
        self.data.copy_(self.manifold.init_weights(
            self, abs(self.curvature), irange))
        self.res.zero_()


class _MC_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                weight: MCTensor,
                padding_idx=None,
                max_norm=None,
                norm_type: float = 2,
                scale_grad_by_freq: bool = None,
                sparse: bool = False):
        ctx.sparse, ctx.weight_shape = sparse, weight.shape
        if max_norm is not None:
            torch.embedding_renorm_(weight.as_subclass(torch.Tensor).detach(), input, max_norm, norm_type)
            weight.res.data.zero_()
        if scale_grad_by_freq:
            raise NotImplemented
        if padding_idx:
            raise NotImplemented
        ctx.save_for_backward(input)
        return weight[input]

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        _grad_output = torch.sparse_coo_tensor(
            x.unsqueeze(0), grad_output, size=ctx.weight_shape)
        if not ctx.sparse:
            _grad_output = _grad_output.to_dense()
        return (None, _grad_output, ) + (None, ) * 5


mc_embedding_func = _MC_Embedding.apply
class MCHEmbedding(torch.nn.Embedding):
    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, nc=1, 
                 padding_idx=None, max_norm=None, norm_type=2, 
                 scale_grad_by_freq=False, sparse=False):
        super(MCHEmbedding, self).__init__(*args)
        self.weight = MCHParameter(self.weight, manifold=manifold, curvature=curvature, nc=nc)
        self.weight.init_weights()

        self.sparse = sparse
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

    def forward(self, input):
        tmp = mc_embedding_func(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
        output = tmp.as_subclass(MCHTensor)
        output.res = tmp.res
        output._nc = tmp._nc
        output.manifold = self.weight.manifold
        output.curvature = self.weight.curvature
        return output