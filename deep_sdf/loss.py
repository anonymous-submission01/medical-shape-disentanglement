import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.optimize
import logging
from scipy.spatial.distance import cdist
#from torch.func import jvp, vjp

def corr_leakage_penalty(x: torch.Tensor, y: torch.Tensor, target_dim: int, eps: float = 1e-6) -> torch.Tensor:
    """
    L_leak = sum_{d != target_dim} corr(x[:, d], y)^2

    Args:
        x: [B, D] latent tensor
        y: [B] (or [B,1]) binary labels {0,1}
        target_dim: the dimension that is allowed to carry label info
        eps: numerical stability

    Returns:
        scalar tensor (keeps gradients)
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)
    if target_dim < 0 or target_dim >= D:
        raise ValueError(f"target_dim {target_dim} out of range for D={D}")

    y = y.view(-1).float()
    if y.numel() != B:
        raise ValueError("y must have the same batch size as x")

    # standardize y
    y = (y - y.mean()) / (y.std().clamp_min(eps))
    y = y.view(B, 1)

    # standardize x per-dimension
    xz = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True).clamp_min(eps))

    mask = torch.ones(D, dtype=torch.bool, device=x.device)
    mask[target_dim] = False
    if mask.sum() == 0:
        return x.new_tensor(0.0)

    xr = xz[:, mask]

    # Pearson corr for standardized variables: corr = mean(x*y)
    corr = (xr * y).mean(dim=0)
    return (corr ** 2).sum()


def cross_cov_penalty(x: torch.Tensor, target_dim: int, eps: float = 1e-6) -> torch.Tensor:
    """
    L_cross = sum_j cov(x_target, x_rest_j)^2

    Args:
        x: [B, D] latent tensor
        target_dim: target dimension index
        eps: numerical stability

    Returns:
        scalar tensor (keeps gradients)
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)
    if target_dim < 0 or target_dim >= D:
        raise ValueError(f"target_dim {target_dim} out of range for D={D}")

    x0 = x - x.mean(dim=0, keepdim=True)
    xt = x0[:, target_dim:target_dim + 1]

    mask = torch.ones(D, dtype=torch.bool, device=x.device)
    mask[target_dim] = False
    if mask.sum() == 0:
        return x.new_tensor(0.0)

    xr = x0[:, mask]

    # covariance for zero-mean variables: cov = mean(x*y)
    cov = (xt * xr).mean(dim=0)
    return (cov ** 2).sum()

class CovarianceLoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        B, D = z.shape
        if B <= 1 or D <= 1:
            return z.new_tensor(0.0)

        z = z - z.mean(dim=0, keepdim=True)
        denom = float(B - 1)
        cov = (z.t() @ z) / (denom + self.eps)
        offdiag = cov - torch.diag_embed(torch.diagonal(cov))
        # Normalize by 1/(D*(D-1)) where D is latent dimension
        # D*(D-1) is the number of off-diagonal elements
        return (offdiag ** 2).sum() / (D * (D - 1))


class GMMPriorLoss(nn.Module):
    """
    Unsupervised GMM prior on deterministic latents z:

      p(z) = sum_k pi_k * N(z | mu_k, Sigma_k)
      L_gmm = -(1/B) * sum_i log p(z_i)

    Uses diagonal covariance for stability.
    """

    def __init__(
        self,
        K: int,
        latent_dim: int,
        learn_pi: bool = True,
        init_sigma: float = 0.5,
        min_sigma: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.K = int(K)
        self.D = int(latent_dim)
        self.learn_pi = bool(learn_pi)
        self.eps = float(eps)

        # mixture parameters
        self.mu = nn.Parameter(torch.randn(self.K, self.D) * 0.01)
        init_log_sigma = math.log(float(init_sigma))
        self.log_sigma = nn.Parameter(torch.full((self.K, self.D), init_log_sigma))
        self.min_sigma = float(min_sigma)

        # pi_k stored as logits -> softmax
        self.logits = nn.Parameter(torch.zeros(self.K), requires_grad=self.learn_pi)

        # logging helpers
        self.last_nll = None
        self.last_avg_entropy = None

    def _log_gaussian_diag(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns log N(z | mu_k, diag(sigma_k^2)) for all i,k:
          output: [B, K]
        """
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        B, D = z.shape
        assert D == self.D

        sigma = self.min_sigma + F.softplus(self.log_sigma)  # [K, D]
        var = sigma * sigma  # [K, D]

        z_ = z.unsqueeze(1)   # [B, 1, D]
        mu_ = self.mu.unsqueeze(0)   # [1, K, D]
        var_ = var.unsqueeze(0)      # [1, K, D]

        mahal = ((z_ - mu_) ** 2 / (var_ + self.eps)).sum(dim=2)  # [B, K]
        log_det = torch.log(var_ + self.eps).sum(dim=2)           # [1, K] -> [B, K]
        const = self.D * math.log(2.0 * math.pi)

        return -0.5 * (mahal + log_det + const)

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        """
        Soft assignments r_{ik} = p(k|z_i).
        """
        logN = self._log_gaussian_diag(z)  # [B, K]

        if self.learn_pi:
            log_pi = F.log_softmax(self.logits, dim=0)  # [K]
        else:
            log_pi = z.new_tensor([-math.log(self.K)] * self.K)

        log_num = logN + log_pi.unsqueeze(0)           # [B, K]
        log_den = torch.logsumexp(log_num, dim=1, keepdim=True)  # [B, 1]
        return torch.exp(log_num - log_den)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        B, D = z.shape
        assert D == self.D
        if B == 0:
            return z.new_tensor(0.0)

        logN = self._log_gaussian_diag(z)  # [B, K]
        if self.learn_pi:
            log_pi = F.log_softmax(self.logits, dim=0)
        else:
            log_pi = z.new_tensor([-math.log(self.K)] * self.K)

        logp = torch.logsumexp(logN + log_pi.unsqueeze(0), dim=1)  # [B]
        nll = -logp.mean()

        with torch.no_grad():
            r = self.responsibilities(z)
            entropy = -(r * torch.log(r + self.eps)).sum(dim=1).mean()
            self.last_avg_entropy = entropy
            self.last_nll = nll.detach()

        return nll


class SensitivityLoss(nn.Module):
    """
    Hinge-floor sensitivity loss on a latent dimension.

    For z in R^{B x D}, perturb z[:, target_dim] by +/- eps,
    compute delta = mean ||g(z+) - g(z-)||_2, and penalize if
    delta < eta: loss = max(0, eta - delta)^2.
    """

    def __init__(self, eps: float = 0.02, eta: float = 0.0025, target_dim: int = 0):
        super().__init__()
        self.eps = float(eps)
        self.eta = float(eta)
        self.target_dim = int(target_dim)

    def forward(self, z: torch.Tensor, decoder: nn.Module):
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            return z.new_tensor(0.0), z.new_tensor(0.0)
        if self.target_dim < 0 or self.target_dim >= z.size(1):
            raise ValueError(
                f"target_dim {self.target_dim} out of range for D={z.size(1)}"
            )

        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[:, self.target_dim] += self.eps
        z_minus[:, self.target_dim] -= self.eps

        c_plus = decoder(z_plus)
        c_minus = decoder(z_minus)
        delta = torch.norm(c_plus - c_minus, dim=1).mean()
        loss = (F.relu(self.eta - delta) / self.eta) ** 2
        return loss, delta


class RankLossZ0(nn.Module):
    """
    Pairwise hinge ranking loss on a target latent dimension.

    Enforces z[target_dim] to be larger for CN than AD by a margin.
    """

    def __init__(self, margin: float = 0.5, target_dim: int = 0, cn_label: int = 1):
        super().__init__()
        self.margin = float(margin)
        self.target_dim = int(target_dim)
        self.cn_label = int(cn_label)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            return z.new_tensor(0.0)
        if self.target_dim < 0 or self.target_dim >= z.size(1):
            raise ValueError(
                f"target_dim {self.target_dim} out of range for D={z.size(1)}"
            )

        z0 = z[:, self.target_dim]
        y = y.view(-1)
        cn = z0[y == self.cn_label]
        ad = z0[y != self.cn_label]
        if cn.numel() == 0 or ad.numel() == 0:
            return z0.new_tensor(0.0)

        diffs = cn.unsqueeze(1) - ad.unsqueeze(0)  # [nCN, nAD]
        loss = F.relu(self.margin - diffs).mean()
        return loss


class MatchStdZ0(nn.Module):
    """
    Match the std of a target latent dimension to the mean std of other dims.
    """

    def __init__(self, target_dim: int = 0, eps: float = 1e-6):
        super().__init__()
        self.target_dim = int(target_dim)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor):
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            zero = z.new_tensor(0.0)
            return zero, zero.detach(), zero.detach()
        if self.target_dim < 0 or self.target_dim >= z.size(1):
            raise ValueError(
                f"target_dim {self.target_dim} out of range for D={z.size(1)}"
            )

        z0 = z[:, self.target_dim]
        std0 = z0.std(unbiased=False).clamp_min(self.eps)

        if z.size(1) <= 1:
            return (std0 - std0).pow(2), std0.detach(), std0.detach()

        other = torch.cat([z[:, : self.target_dim], z[:, self.target_dim + 1 :]], dim=1)
        std_ref = other.std(dim=0, unbiased=False).mean().clamp_min(self.eps)

        return (std0 - std_ref).pow(2), std0.detach(), std_ref.detach()


class IsometryLoss(nn.Module):
    """
    Isometric regularization loss from "Isometric Regularization for 
    Manifolds of Functional Data" (ICLR 2025).
    
    Encourages the latent→function map to be locally distance/angle preserving
    by regularizing the latent Jacobian metric H(z) = E_x[J(x,z)^T J(x,z)].
    
    Uses Hutchinson trace estimator with JVP + VJP for efficient computation.
    """
    
    def __init__(self, num_hutchinson_probes: int = 1, eps: float = 1e-8):
        """
        Args:
            num_hutchinson_probes: Number of random probe vectors for Hutchinson estimator.
                                   1 is usually sufficient, 2 if noisy.
            eps: Small constant for numerical stability in division.
        """
        super(IsometryLoss, self).__init__()
        self.num_hutchinson_probes = num_hutchinson_probes
        self.eps = eps
    
    def forward(
        self,
        decoder: nn.Module,
        latent_codes: torch.Tensor,
        iso_points: torch.Tensor,
        latent_size: int,
    ) -> torch.Tensor:
        """
        Compute isometry loss using Algorithm 1 from the paper.
        
        Args:
            decoder: The SDF decoder network (expects input [latent, xyz] concatenated)
            latent_codes: [N, m] latent codes (already expanded per point)
            iso_points: [N, 3] xyz coordinates
            latent_size: Dimension of latent code (m)
            
        Returns:
            Scalar isometry loss (G2 / G1 ratio)
        """
        N = iso_points.shape[0]
        m = latent_size
        device = iso_points.device
        
        G1_accum = 0.0
        G2_accum = 0.0
        
        for _ in range(self.num_hutchinson_probes):
            # Sample probe vector v ~ N(0, I_m)
            # Use same probe for all points (per-batch Hutchinson)
            v = torch.randn(1, m, device=device).expand(N, m)  # [N, m]
            
            # Build input: [latent, xyz]
            inp = torch.cat([latent_codes, iso_points], dim=-1)  # [N, m+3]
            inp.requires_grad_(True)
            
            # Build tangent: [v, 0_xyz] - perturb only latent part
            tangent = torch.cat([
                v,  # Perturbation in latent (first m dims)
                torch.zeros(N, 3, device=device)  # No perturbation in xyz
            ], dim=-1)  # [N, m+3]
            
            with torch.enable_grad():
                outputs = decoder(inp)  # [N, 1]
                
                # Full gradient w.r.t. input
                G = torch.autograd.grad(
                    outputs=outputs,
                    inputs=inp,
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=True,
                    retain_graph=True
                )[0]  # [N, m+3]
                
                # JVP result
                jvp_result = (G * tangent).sum(dim=-1)  # [N]
                
                # G1: E[G^2]
                G1 = (jvp_result ** 2).mean()
                G1_accum += G1
                
                # VJP for scalar output: J^T (J v) = (J v) * J
                D_full = jvp_result.unsqueeze(-1) * G  # [N, m+3]
                
                # Get z-part (first m components since input is [z, x])
                Dz = D_full[:, :m]  # [N, m]
                
                # E_x[D_z] then ||.||^2
                Dz_mean = Dz.mean(dim=0)  # [m]
                G2 = (Dz_mean ** 2).sum()  # scalar
                G2_accum += G2
        
        G1_avg = G1_accum / self.num_hutchinson_probes
        G2_avg = G2_accum / self.num_hutchinson_probes

        # Expose for logging without changing the return signature
        self.last_g1 = G1_avg.detach()
        self.last_g2 = G2_avg.detach()
        
        return G2_avg / (G1_avg + self.eps)


class GradientMetricIsotropyLoss(nn.Module):
    """
    L = ||offdiag(H)||_F^2 + alpha * Var(diag(H))

    where:
      g_i = grad_z f(z, x_i) in R^m
      H   = (1/N) sum_i g_i g_i^T = (G^T G)/N in R^{m x m}
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-12, normalize: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.normalize = bool(normalize)

        # for logging/debug
        self.last_offdiag = None
        self.last_diag_var = None
        self.last_diag_mean = None

    def forward(
        self,
        decoder: nn.Module,
        latent_codes: torch.Tensor,  # [N, m] (expanded per point)
        iso_points: torch.Tensor,    # [N, 3]
        latent_size: int,
    ) -> torch.Tensor:
        N = iso_points.shape[0]
        m = latent_size

        assert latent_codes.shape[0] == N, "latent_codes and iso_points must have same N"
        assert latent_codes.shape[1] == m, "latent_codes dim must match latent_size"
        assert iso_points.shape[1] == 3, "iso_points must be [N,3]"

        # Build input [z, x]
        inp = torch.cat([latent_codes, iso_points], dim=-1)  # [N, m+3]
        inp.requires_grad_(True)

        # Forward
        out = decoder(inp)  # [N,1] or [N]
        if out.dim() == 2 and out.shape[1] == 1:
            out = out[:, 0]  # [N]

        # Gradient wrt input (then slice z-part)
        grad_inp = torch.autograd.grad(
            outputs=out,
            inputs=inp,
            grad_outputs=torch.ones_like(out),
            create_graph=True,   # needed so this loss trains the decoder
            retain_graph=True,
        )[0]  # [N, m+3]

        G = grad_inp[:, :m]  # [N, m] = grad_z f(z, x_i)

        # H = (G^T G)/N (metric estimate)
        H = (G.transpose(0, 1) @ G) / (float(N) + self.eps)  # [m, m]

        diag = torch.diagonal(H)  # [m]
        offdiag = H - torch.diag_embed(diag)

        off_loss = (offdiag ** 2).sum()
        diag_var = diag.var(unbiased=False)

        if self.normalize:
            # keeps magnitude somewhat stable across latent sizes
            off_loss = off_loss / (m * (m - 1) + self.eps)

        loss = off_loss + self.alpha * diag_var

        # stash for logging
        self.last_offdiag = off_loss.detach()
        self.last_diag_var = diag_var.detach()
        self.last_diag_mean = diag.mean().detach()

        return loss


def select_near_surface_points(xyz, sdf_gt, clamp_dist, num_iso_points):
    """
    Select near-surface points for isometry loss computation.
    Prioritizes points with |SDF| < clamp_dist (near surface).
    
    Args:
        xyz: [N, 3] point coordinates
        sdf_gt: [N, 1] ground truth SDF values
        clamp_dist: Truncation distance (points with |SDF| < this are near-surface)
        num_iso_points: Number of points to select
        
    Returns:
        [num_iso_points, 3] selected points
    """
    sdf_abs = sdf_gt.abs().squeeze()
    
    # Find near-surface points (|SDF| < clamp_dist)
    near_surface_mask = sdf_abs < clamp_dist
    near_surface_indices = torch.where(near_surface_mask)[0]
    
    if len(near_surface_indices) >= num_iso_points:
        # Sample from near-surface points
        perm = torch.randperm(len(near_surface_indices), device=xyz.device)[:num_iso_points]
        selected_indices = near_surface_indices[perm]
    else:
        # Use all near-surface points + some random points
        num_random = num_iso_points - len(near_surface_indices)
        far_indices = torch.where(~near_surface_mask)[0]
        
        if len(far_indices) >= num_random:
            perm = torch.randperm(len(far_indices), device=xyz.device)[:num_random]
            random_indices = far_indices[perm]
        else:
            random_indices = far_indices
        
        selected_indices = torch.cat([near_surface_indices, random_indices])
        
        # If still not enough, pad by repeating
        if len(selected_indices) < num_iso_points:
            repeat_times = (num_iso_points // len(selected_indices)) + 1
            selected_indices = selected_indices.repeat(repeat_times)[:num_iso_points]
    
    return xyz[selected_indices]


# SNNL loss modified fast
class SNNLoss(nn.Module):
    def __init__(self, T):
        super(SNNLoss, self).__init__()
        self.T = T
        self.STABILITY_EPS = 0.00001

    def forward(self, x, y):
        x = x.to('cuda')
        b = x.size(0)  # Batch size
        y = y.squeeze()

        x_expanded = x[:,0].unsqueeze(1)
          # Expand dimensions for broadcasting
        y_expanded = y.unsqueeze(0)

        same_class_mask = y_expanded == y_expanded.t()

        squared_distances = (x_expanded - x_expanded.t()) ** 2
        exp_distances = torch.exp(-(squared_distances / self.T))
        exp_distances = exp_distances * (1 - torch.eye(b, device='cuda'))
        #print(exp_distances)

        numerator = exp_distances * same_class_mask
        denominator = exp_distances
        # remaining elements
        exp_distances_all = torch.zeros_like(exp_distances, device='cuda')
        for i in range(1, x.shape[1]):
            x_expanded = x[:,i].unsqueeze(1)
            squared_distances = (x_expanded - x_expanded.t()) ** 2
            exp_distances = torch.exp(-(squared_distances / self.T))
            exp_distances = exp_distances * (1 - torch.eye(b, device='cuda'))
            exp_distances = exp_distances * same_class_mask
            exp_distances_all = exp_distances_all + exp_distances

        
        denominator1 = exp_distances_all/float(x.shape[1]-1)
        #print(denominator)

        lsn_loss = -torch.log(self.STABILITY_EPS + (numerator.sum(dim=1) / (self.STABILITY_EPS + (0.5*denominator.sum(dim=1)) + (0.5*denominator1.sum(dim=1))))).mean()

        return lsn_loss
    

class SNNLossCls(nn.Module):
    """
    Classification SNNL:
      - Forces z[:, target_dim] to align with binary labels (0/1).
      - Penalizes disease similarity in non-target dims.
      - Matches the unified-SNNL form (classification variant with same-class positives).
    """
    def __init__(self,
                 T: float = 2.0,
                 lam1: float = 1.0,
                 lam2: float = 2.0,
                 target_dim: int = 0,
                 normalize_z: bool = True,
                 use_adaptive_T: bool = True,
                 eps: float = 1e-8,
                 clamp_ratio: bool = True):
        super().__init__()
        self.T = float(T)
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.target_dim = int(target_dim)
        self.normalize_z = bool(normalize_z)
        self.use_adaptive_T = bool(use_adaptive_T)
        self.eps = float(eps)
        self.clamp_ratio = bool(clamp_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] latents
        y: [B] or [B,1] binary labels {0,1}
        """
        device, dtype = x.device, x.dtype
        B, D = x.shape
        y = y.view(-1, 1).to(device=device, dtype=torch.long)

        # optional per-dim batch standardization (stabilizes distances/temperature)
        if self.normalize_z:
            with torch.no_grad():
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - m) / s

        offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
        same = (y == y.t()) & offdiag  # positives: same class pairs

        # --- target dimension distances (numerator + denom term 1)
        zt = x[:, self.target_dim:self.target_dim+1]     # [B,1]
        d2_t = (zt - zt.t()).pow(2)                      # [B,B]

        # adaptive temperature (median of positive distances) or fixed T
        if self.use_adaptive_T and same.any():
            T_eff = d2_t[same].median().clamp_min(1e-6).detach()
        else:
            T_eff = torch.tensor(self.T, device=device, dtype=dtype)

        Kt = torch.exp(-d2_t / T_eff)
        num  = (Kt * same).sum(dim=1)            # sum over positives on target dim
        den1 = (Kt * offdiag).sum(dim=1)         # sum over all off-diagonals on target dim

        # --- other dims term: exp( mean_{d != target} ||z_d^i - z_d^j||^2 / T )
        if D > 1:
            other_idx = torch.tensor([d for d in range(D) if d != self.target_dim],
                                     device=device)
        else:
            other_idx = torch.empty(0, dtype=torch.long, device=device)

        if other_idx.numel() > 0 and same.any():
            xo = x[:, other_idx]                               # [B, D-1]
            diff = xo.unsqueeze(1) - xo.unsqueeze(0)           # [B,B,D-1]
            sq_mean = diff.pow(2).mean(dim=2)                  # [B,B]
            K_other = torch.exp(-sq_mean / T_eff)
            den2 = (K_other * same).sum(dim=1)
        else:
            den2 = torch.zeros(B, device=device, dtype=dtype)

        denom = self.lam1 * den1 + self.lam2 * den2 + self.eps
        frac = num / denom
        if self.clamp_ratio:
            frac = torch.clamp(frac, min=1e-12, max=1.0 - 1e-7)

        has_pos = same.any(dim=1)
        if has_pos.any():
            loss = -torch.log(frac[has_pos]).mean()
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        return loss

    
# SNNL loss reg modified fast
class SNNRegLoss(nn.Module):
    def __init__(self, T, threshold):
        super(SNNRegLoss, self).__init__()
        self.T = T
        self.STABILITY_EPS = 0.00001
        self.threshold = threshold

    def forward(self, x, y):
        x = x.to('cuda')
        b = x.size(0)  # Batch size
        y = y.squeeze()

        x_expanded = x[:,1].unsqueeze(1)  # Expand dimensions for broadcasting
        y_expanded = y.unsqueeze(0)

        abs_diff_matrix = torch.abs(y_expanded - y_expanded.t())
        same_class_mask = abs_diff_matrix <= self.threshold

        squared_distances = (x_expanded - x_expanded.t()) ** 2
        exp_distances = torch.exp(-(squared_distances / self.T))
        exp_distances = exp_distances * (1 - torch.eye(b, device='cuda'))
        #print(exp_distances)

        numerator = exp_distances * same_class_mask
        denominator = exp_distances
        # remaining elements
        exp_distances_all = torch.zeros_like(exp_distances, device='cuda')
        x_expanded = x[:,0].unsqueeze(1)
        squared_distances = (x_expanded - x_expanded.t()) ** 2
        exp_distances = torch.exp(-(squared_distances / self.T))
        exp_distances = exp_distances * (1 - torch.eye(b, device='cuda'))
        exp_distances = exp_distances * same_class_mask
        exp_distances_all = exp_distances_all + exp_distances
        for i in range(2, x.shape[1]):
            x_expanded = x[:,i].unsqueeze(1)
            squared_distances = (x_expanded - x_expanded.t()) ** 2
            exp_distances = torch.exp(-(squared_distances / self.T))
            exp_distances = exp_distances
            exp_distances = exp_distances * (1 - torch.eye(b, device='cuda'))
            exp_distances = exp_distances * same_class_mask
            exp_distances_all = exp_distances_all + exp_distances

        #print(denominator)
        denominator1 = exp_distances_all/float(x.shape[1]-1)

        lsn_loss = -torch.log(self.STABILITY_EPS + (numerator.sum(dim=1) / (self.STABILITY_EPS + (0.5*denominator.sum(dim=1)) + (0.5*denominator1.sum(dim=1))))).mean()

        return lsn_loss
    

class SNNRegLossExact(nn.Module):
 
    def __init__(self,
                 T=2.0, lam1=1.0, lam2=0.5,
                 threshold=0.05, target_dim=1,
                 normalize_z=True, use_adaptive_T=True,
                 pos_mode='threshold', topk_frac=0.1,
                 eps=1e-8, clamp_ratio=True):
        super().__init__()
        self.T = float(T)
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.threshold = float(threshold)
        self.target_dim = int(target_dim)
        self.normalize_z = bool(normalize_z)
        self.use_adaptive_T = bool(use_adaptive_T)
        self.pos_mode = str(pos_mode)
        self.topk_frac = float(topk_frac)
        self.eps = float(eps)
        self.clamp_ratio = bool(clamp_ratio)

    def _build_positive_mask(self, y, offdiag):
        B = y.shape[0]
        abs_dy = torch.abs(y - y.t())  # [B,B]
        if self.pos_mode == 'topk':
            # row-wise K nearest in age (exclude self)
            abs_dy = abs_dy.masked_fill(~offdiag, float('inf'))
            K = max(1, int(round(self.topk_frac * (B-1))))
            thr_i = abs_dy.kthvalue(K, dim=1).values.unsqueeze(1)  # [B,1]
            same_age = (abs_dy <= thr_i)
        else:
            # fixed band in [0,1] space
            same_age = (abs_dy <= self.threshold)
        same_age = same_age & offdiag
        return same_age

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] latent codes (dim `target_dim` is z2)
        y: [B] or [B,1] age in [0,1]
        """
        device, dtype = x.device, x.dtype
        B, D = x.shape
        assert D >= 2 and 0 <= self.target_dim < D

        # optional per-dim standardization (stabilizes T and training)
        if self.normalize_z:
            with torch.no_grad():
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - m) / s

        y = y.view(-1, 1).to(device=device, dtype=dtype)
        offdiag = ~torch.eye(B, dtype=torch.bool, device=device)

        # positives based on age
        same_age = self._build_positive_mask(y, offdiag)

        # --- z2 distances (numerator + denom term 1) ---
        z2 = x[:, self.target_dim:self.target_dim+1]           # [B,1]
        d2 = (z2 - z2.t()).pow(2)                              # [B,B]

        # adaptive T on z2 (optional)
        if self.use_adaptive_T and same_age.any():
            T_eff = d2[same_age].median().clamp_min(1e-6).detach()
        else:
            T_eff = torch.tensor(self.T, device=device, dtype=dtype)

        K2 = torch.exp(-d2 / T_eff)
        num_sum  = (K2 * same_age).sum(dim=1)                  # positives on z2
        den1_sum = (K2 * offdiag).sum(dim=1)                   # all pairs on z2

        # --- other dims term: exp( mean_{d != z2} ||z_d^i - z_d^j||^2 / T ) ---
        if D > 1:
            idx_left  = torch.arange(0, self.target_dim, device=device)
            idx_right = torch.arange(self.target_dim+1, D, device=device)
            other_idx = torch.cat([idx_left, idx_right], dim=0)
        else:
            other_idx = torch.empty(0, dtype=torch.long, device=device)

        if other_idx.numel() > 0:
            x_other = x[:, other_idx]                          # [B, D-1]
            diff = x_other.unsqueeze(1) - x_other.unsqueeze(0) # [B,B,D-1]
            sq_mean = diff.pow(2).mean(dim=2)                  # mean across other dims
            K_other = torch.exp(-sq_mean / T_eff)
            den2_sum = (K_other * same_age).sum(dim=1)
        else:
            den2_sum = torch.zeros(B, device=device, dtype=dtype)

        # combine denominators
        denom = self.lam1 * den1_sum + self.lam2 * den2_sum + self.eps

        # ratio and loss
        frac = num_sum / denom
        if self.clamp_ratio:
            frac = torch.clamp(frac, min=1e-12, max=1-1e-7)

        has_pos = same_age.any(dim=1)
        if has_pos.any():
            loss = -torch.log(frac[has_pos]).mean()
        else:
            # no positives in batch; return 0 (or resample)
            loss = torch.zeros((), device=device, dtype=dtype)

        return loss

    
# Attribute VAE loss
class AttributeLoss(nn.Module):
    def __init__(self, factor=1.0):
        super(AttributeLoss, self).__init__()
        self.factor = factor
        self.loss_fn = nn.L1Loss()

    def forward(self, latent_code, attribute):
        # compute latent distance matrix
        latent_code = latent_code.to('cuda')
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        lc_tanh = torch.tanh(lc_dist_mat * self.factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        #write logging code to check the device of lc_tanh and attribute_sign
        #logging.info('lc_tanh device: %s', lc_tanh.device)
        #logging.info('attribute_sign device: %s', attribute_sign.device)
        attribute_loss = self.loss_fn(lc_tanh, attribute_sign.float())

        return attribute_loss



# Wasserstein loss proposed by nilanjan
class WassersteinLoss(nn.Module):
    def __init__(self, delta):
        super(WassersteinLoss, self).__init__()
        self.delta = delta
        self.h_loss = torch.nn.HuberLoss(reduction='mean', delta=delta)

    def linear_assignment(self, x, u):
        dist_matrix = cdist(x, u)
        _, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
        return col_ind

    def forward(self, x):
        bsize = x.shape[0]
        dim = x.shape[1]

        u = x[torch.randperm(bsize), 0:1]
        for i in range(dim - 1):
            u = torch.cat((u, x[torch.randperm(bsize), i + 1:i + 2]), dim=1)

        with torch.no_grad():
            ind = self.linear_assignment(x.cpu().detach().numpy(), u.cpu().detach().numpy())

        loss = self.h_loss(x, u[ind])

        return loss


class ClsCorrelationLoss(nn.Module):
    def __init__(self):
        super(ClsCorrelationLoss, self).__init__()

    def forward(self, z_batch, y_batch):
        # Split z_batch and y_batch into categories
        z_1 = z_batch[y_batch.flatten() == 1.0]
        z_0 = z_batch[y_batch.flatten() == 0.0]
        n_1 = len(z_1)
        n_0 = len(z_0)
        n = n_1 + n_0

        # Calculate means for the two categories
        mean_z_1 = torch.mean(z_1[:, 0])
        mean_z_0 = torch.mean(z_0[:, 0])

        # Multiplier
        mlt = math.sqrt((n_1 * n_0) / (n**2))

        # Calculate point biserial correlation
        r_pb = (mean_z_1 - mean_z_0) / torch.std(z_batch[:, 0]) * mlt

        # Calculate correlation of other dimensions with y
        other_dim_corrs = torch.zeros_like(z_batch[:, 1])
        for i in range(1, z_batch.shape[1]):
            other_dim_corrs[i-1] = (torch.mean(z_1[:, i]) - torch.mean(z_0[:, i])) / torch.std(z_batch[:, i]) * mlt

        # Loss components
        ncc_loss = 1 - torch.abs(r_pb)  # Minimize correlation
        other_dims_loss = torch.mean(torch.abs(other_dim_corrs))  # Minimize other dimension correlations

        # Combine losses with weights
        total_loss = ncc_loss + other_dims_loss

        return total_loss
    
#Pearson correlation

class RegCorrelationLoss(nn.Module):
    def __init__(self):
        super(RegCorrelationLoss, self).__init__()

    def forward(self, z_batch, y_batch):
        # Calculate the means of x and y
        y_batch = y_batch.squeeze()
        mean_z = torch.mean(z_batch[:, 1])
        mean_y = torch.mean(y_batch)
        # Calculate the differences from the means
        diff_z = z_batch[:, 1] - mean_z
        diff_y = y_batch - mean_y
        
        # Calculate the sum of squared differences
        sum_squared_diff_z = torch.sum(diff_z ** 2)
        sum_squared_diff_y = torch.sum(diff_y ** 2)
        
        # Calculate the cross-product of differences
        cross_product = torch.sum(diff_z * diff_y)
        
        # Calculate the denominator (product of standard deviations)
        denominator = torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)
        
        # Calculate the Pearson correlation coefficient
        r_p = cross_product / denominator

        # Calculate correlation of other dimensions with y
        other_dim_corrs = torch.zeros_like(z_batch[:, 0])
        #first element
        mean_z, mean_y = torch.mean(z_batch[:, 0]), torch.mean(y_batch)
        diff_z, diff_y = z_batch[:, 0] - mean_z, y_batch - mean_y
        sum_squared_diff_z, sum_squared_diff_y = torch.sum(diff_z ** 2), torch.sum(diff_y ** 2)
        other_dim_corrs[0] = torch.sum(diff_z * diff_y) / torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)
        #remaining element
        for i in range(2, z_batch.shape[1]):
            mean_z, mean_y = torch.mean(z_batch[:, i]), torch.mean(y_batch)
            diff_z, diff_y = z_batch[:, i] - mean_z, y_batch - mean_y
            sum_squared_diff_z, sum_squared_diff_y = torch.sum(diff_z ** 2), torch.sum(diff_y ** 2)
            other_dim_corrs[i-1] = torch.sum(diff_z * diff_y) / torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)

        # Loss components
        ncc_loss = 1 - torch.abs(r_p)  # Minimize correlation
        other_dims_loss = torch.mean(torch.abs(other_dim_corrs))  # Minimize other dimension correlations

        # Combine losses with weights
        total_loss = ncc_loss + other_dims_loss

        return total_loss

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               temperature=100.,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
    
    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def pairwise_cos_distance(A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
        return 1 - prod
    
    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
            
        return torch.exp(-(distance_matrix / temp))
    
    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0], device='cuda')
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))
    
    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)
    
    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)
    
    @staticmethod
    def SNNL(x, y, temp=100, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()
    
# DIP VAE II Loss
# Add this to your loss.py file

class DIPVAEIILoss(nn.Module):
    def __init__(self, lambda_off=1.0, lambda_diag=1.0, beta=0.01):
        """
        DIP-VAE II loss with overall weighting factor
        
        Args:
            lambda_off: Weight for off-diagonal covariance penalty (reduce from 10.0)
            lambda_diag: Weight for diagonal covariance penalty (reduce from 5.0)  
            lambda_mean: Weight for mean regularization penalty
            beta: Overall weighting factor for the entire DIP loss
        """
        super(DIPVAEIILoss, self).__init__()
        self.lambda_off = lambda_off
        self.lambda_diag = lambda_diag
        self.beta = beta  # Overall scaling factor
        
    def forward(self, mu, logvar):
        B, d = mu.size()
        
        # Sample from posterior using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Mean regularization (encourage zero mean)
        z_mean = torch.mean(z, dim=0)
        
        # Center the samples
        z_centered = z - z_mean.unsqueeze(0)
        
        # Compute covariance matrix
        C_z = torch.matmul(z_centered.t(), z_centered) / max(B - 1, 1)
        
        # Off-diagonal penalty (encourage independence between dimensions)
        mask = torch.eye(d, device=C_z.device)
        loss_off = self.lambda_off * torch.sum((C_z * (1 - mask)).pow(2))
        
        # Diagonal penalty (encourage unit variance)
        loss_diag = self.lambda_diag * torch.sum((torch.diag(C_z) - 1).pow(2))
        
        # Apply overall scaling
        total_dip_loss = self.beta * (loss_off + loss_diag)
        
        return total_dip_loss
