import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_target_dims(target_dims, total_dim):
    if target_dims is None or len(target_dims) == 0:
        raise ValueError("target_dims must be a non-empty list of indices.")
    if any(d < 0 or d >= total_dim for d in target_dims):
        raise ValueError(f"target_dims out of range for D={total_dim}: {target_dims}")
    if len(set(target_dims)) != len(target_dims):
        raise ValueError(f"target_dims has duplicates: {target_dims}")


def _split_dims(total_dim, target_dims, device):
    mask = torch.ones(total_dim, dtype=torch.bool, device=device)
    mask[target_dims] = False
    other_dims = mask.nonzero(as_tuple=False).view(-1)
    return other_dims


def _pca1_scores(x: torch.Tensor) -> torch.Tensor:
    """
    Return per-sample scores along the first PCA component.
    x: [B, D]
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B == 0:
        return x.new_zeros((0,))
    x_centered = x - x.mean(dim=0, keepdim=True)
    if D == 1:
        return x_centered[:, 0]
    if B <= 1:
        return x.new_zeros((B,))
    cov = (x_centered.t() @ x_centered) / (float(B - 1))
    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(cov)
        v1 = eigvecs[:, -1].detach()
    return x_centered @ v1


def corr_leakage_penalty_group(
    x: torch.Tensor,
    y: torch.Tensor,
    target_dims,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Group leakage penalty:
      L_leak = sum_{d not in target_dims} corr(x[:, d], y)^2

    Args:
        x: [B, D] latent tensor
        y: [B] or [B,1] labels
        target_dims: list of allowed dims for label info
        eps: numerical stability
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)
    _validate_target_dims(target_dims, D)

    y = y.view(-1).float()
    if y.numel() != B:
        raise ValueError("y must have the same batch size as x")

    # standardize y
    y = (y - y.mean()) / (y.std().clamp_min(eps))
    y = y.view(B, 1)

    # standardize x per-dimension
    xz = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True).clamp_min(eps))

    other_dims = _split_dims(D, target_dims, x.device)
    if other_dims.numel() == 0:
        return x.new_tensor(0.0)

    xr = xz[:, other_dims]
    corr = (xr * y).mean(dim=0)
    return (corr ** 2).sum()


def corr_leakage_penalty_pca_subsets(
    x: torch.Tensor,
    y: torch.Tensor,
    other_subsets,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Leakage penalty using PCA1 score for each non-target subset.
    L = sum_k corr(pca1(x_subset_k), y)^2
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)

    y = y.view(-1).float()
    if y.numel() != B:
        raise ValueError("y must have the same batch size as x")

    y = (y - y.mean()) / (y.std().clamp_min(eps))
    y = y.view(B)

    total = x.new_tensor(0.0)
    for dims in other_subsets:
        if not dims:
            continue
        s = _pca1_scores(x[:, dims])
        s = (s - s.mean()) / (s.std().clamp_min(eps))
        corr = (s * y).mean()
        total = total + corr ** 2
    return total


def cross_cov_penalty_group(
    x: torch.Tensor,
    target_dims,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Group cross-covariance penalty:
      L_cross = sum_{i in target} sum_{j in rest} cov(x_i, x_j)^2
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)
    _validate_target_dims(target_dims, D)

    x0 = x - x.mean(dim=0, keepdim=True)
    xt = x0[:, target_dims]  # [B, T]

    other_dims = _split_dims(D, target_dims, x.device)
    if other_dims.numel() == 0:
        return x.new_tensor(0.0)
    xr = x0[:, other_dims]  # [B, R]

    cov = (xt.t() @ xr) / (float(B) + eps)  # [T, R]
    return (cov ** 2).sum()


def cross_cov_penalty_pca_subsets(
    x: torch.Tensor,
    target_dims,
    other_subsets,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cross-subset covariance penalty using PCA1 scores.
    L = sum_k cov(pca1(x_target), pca1(x_subset_k))^2
    """
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    B, D = x.shape
    if B <= 1 or D <= 1:
        return x.new_tensor(0.0)
    _validate_target_dims(target_dims, D)

    s_t = _pca1_scores(x[:, target_dims])
    s_t = s_t - s_t.mean()
    total = x.new_tensor(0.0)
    for dims in other_subsets:
        if not dims:
            continue
        s_o = _pca1_scores(x[:, dims])
        s_o = s_o - s_o.mean()
        cov = (s_t * s_o).mean()
        total = total + cov ** 2
    return total


class SNNLossClsGroup(nn.Module):
    """
    Grouped classification SNNL for a subset of latent dimensions.
    Uses mean squared distance across target dims, and optionally
    penalizes same-class similarity in non-target dims.
    """

    def __init__(
        self,
        T: float = 2.0,
        lam1: float = 1.0,
        lam2: float = 2.0,
        target_dims=None,
        normalize_z: bool = True,
        use_adaptive_T: bool = True,
        eps: float = 1e-8,
        clamp_ratio: bool = True,
    ):
        super().__init__()
        self.T = float(T)
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.target_dims = list(target_dims) if target_dims is not None else None
        self.normalize_z = bool(normalize_z)
        self.use_adaptive_T = bool(use_adaptive_T)
        self.eps = float(eps)
        self.clamp_ratio = bool(clamp_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] latents
        y: [B] or [B,1] binary labels {0,1}
        """
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        B, D = x.shape
        _validate_target_dims(self.target_dims, D)

        device, dtype = x.device, x.dtype
        y = y.view(-1, 1).to(device=device, dtype=torch.long)

        # optional per-dim batch standardization (stabilizes distances/temperature)
        if self.normalize_z:
            with torch.no_grad():
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - m) / s

        offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
        same = (y == y.t()) & offdiag  # positives: same class pairs

        # target subset distances
        zt = x[:, self.target_dims]                     # [B, T]
        diff_t = zt.unsqueeze(1) - zt.unsqueeze(0)      # [B,B,T]
        d2_t = diff_t.pow(2).mean(dim=2)                # [B,B]

        # adaptive temperature (median of positive distances) or fixed T
        if self.use_adaptive_T and same.any():
            T_eff = d2_t[same].median().clamp_min(1e-6).detach()
        else:
            T_eff = torch.tensor(self.T, device=device, dtype=dtype)

        Kt = torch.exp(-d2_t / T_eff)
        num = (Kt * same).sum(dim=1)
        den1 = (Kt * offdiag).sum(dim=1)

        # other dims penalty
        other_dims = _split_dims(D, self.target_dims, device)
        if other_dims.numel() > 0 and same.any():
            xo = x[:, other_dims]
            diff = xo.unsqueeze(1) - xo.unsqueeze(0)
            sq_mean = diff.pow(2).mean(dim=2)
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


class SNNRegLossExactGroup(nn.Module):
    """
    Grouped regression SNNL for continuous labels.
    Uses mean squared distance across target dims, and optionally
    penalizes same-age similarity in non-target dims.
    """

    def __init__(
        self,
        T=2.0,
        lam1=1.0,
        lam2=0.5,
        threshold=0.05,
        target_dims=None,
        normalize_z=True,
        use_adaptive_T=True,
        pos_mode="threshold",
        topk_frac=0.1,
        eps=1e-8,
        clamp_ratio=True,
    ):
        super().__init__()
        self.T = float(T)
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.threshold = float(threshold)
        self.target_dims = list(target_dims) if target_dims is not None else None
        self.normalize_z = bool(normalize_z)
        self.use_adaptive_T = bool(use_adaptive_T)
        self.pos_mode = str(pos_mode)
        self.topk_frac = float(topk_frac)
        self.eps = float(eps)
        self.clamp_ratio = bool(clamp_ratio)

    def _build_positive_mask(self, y, offdiag):
        B = y.shape[0]
        abs_dy = torch.abs(y - y.t())  # [B,B]
        if self.pos_mode == "topk":
            abs_dy = abs_dy.masked_fill(~offdiag, float("inf"))
            K = max(1, int(round(self.topk_frac * (B - 1))))
            thr_i = abs_dy.kthvalue(K, dim=1).values.unsqueeze(1)  # [B,1]
            same = abs_dy <= thr_i
        else:
            same = abs_dy <= self.threshold
        same = same & offdiag
        return same

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        B, D = x.shape
        _validate_target_dims(self.target_dims, D)

        device, dtype = x.device, x.dtype
        y = y.view(-1, 1).to(device=device, dtype=dtype)

        # optional per-dim standardization
        if self.normalize_z:
            with torch.no_grad():
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - m) / s

        offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
        same = self._build_positive_mask(y, offdiag)

        # target subset distances
        zt = x[:, self.target_dims]
        diff_t = zt.unsqueeze(1) - zt.unsqueeze(0)
        d2_t = diff_t.pow(2).mean(dim=2)

        if self.use_adaptive_T and same.any():
            T_eff = d2_t[same].median().clamp_min(1e-6).detach()
        else:
            T_eff = torch.tensor(self.T, device=device, dtype=dtype)

        Kt = torch.exp(-d2_t / T_eff)
        num_sum = (Kt * same).sum(dim=1)
        den1_sum = (Kt * offdiag).sum(dim=1)

        # other dims penalty
        other_dims = _split_dims(D, self.target_dims, device)
        if other_dims.numel() > 0:
            xo = x[:, other_dims]
            diff = xo.unsqueeze(1) - xo.unsqueeze(0)
            sq_mean = diff.pow(2).mean(dim=2)
            K_other = torch.exp(-sq_mean / T_eff)
            den2_sum = (K_other * same).sum(dim=1)
        else:
            den2_sum = torch.zeros(B, device=device, dtype=dtype)

        denom = self.lam1 * den1_sum + self.lam2 * den2_sum + self.eps
        frac = num_sum / denom
        if self.clamp_ratio:
            frac = torch.clamp(frac, min=1e-12, max=1 - 1e-7)

        has_pos = same.any(dim=1)
        if has_pos.any():
            loss = -torch.log(frac[has_pos]).mean()
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        return loss


class MatchStdGroup(nn.Module):
    """
    Match the mean std of a target subset to the mean std of other dims.
    """

    def __init__(self, target_dims, eps: float = 1e-6):
        super().__init__()
        self.target_dims = list(target_dims) if target_dims is not None else None
        self.eps = float(eps)

    def forward(self, z: torch.Tensor):
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            zero = z.new_tensor(0.0)
            return zero, zero.detach(), zero.detach()

        B, D = z.shape
        _validate_target_dims(self.target_dims, D)

        target = z[:, self.target_dims]
        std_target = target.std(dim=0, unbiased=False).mean().clamp_min(self.eps)

        other_dims = _split_dims(D, self.target_dims, z.device)
        if other_dims.numel() == 0:
            return (std_target - std_target).pow(2), std_target.detach(), std_target.detach()

        other = z[:, other_dims]
        std_ref = other.std(dim=0, unbiased=False).mean().clamp_min(self.eps)

        return (std_target - std_ref).pow(2), std_target.detach(), std_ref.detach()


class SensitivityGroupLoss(nn.Module):
    """
    Hinge-floor sensitivity loss on a subset of latent dimensions.

    Perturb z[:, target_dims] by +/- eps, compute
    delta = mean ||g(z+) - g(z-)||_2, and penalize if
    delta < eta: loss = max(0, eta - delta)^2.
    """

    def __init__(self, eps: float = 0.02, eta: float = 0.0025, target_dims=None):
        super().__init__()
        self.eps = float(eps)
        self.eta = float(eta)
        self.target_dims = list(target_dims) if target_dims is not None else None

    def forward(self, z: torch.Tensor, decoder: nn.Module):
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            return z.new_tensor(0.0), z.new_tensor(0.0)

        B, D = z.shape
        _validate_target_dims(self.target_dims, D)

        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[:, self.target_dims] += self.eps
        z_minus[:, self.target_dims] -= self.eps

        c_plus = decoder(z_plus)
        c_minus = decoder(z_minus)
        delta = torch.norm(c_plus - c_minus, dim=1).mean()
        loss = (F.relu(self.eta - delta) / self.eta) ** 2
        return loss, delta


class RankLossGroup(nn.Module):
    """
    Pairwise hinge ranking loss on a subset of latent dimensions.
    Uses the mean of target dims as the score.
    """

    def __init__(self, margin: float = 0.5, target_dims=None, cn_label: int = 1):
        super().__init__()
        self.margin = float(margin)
        self.target_dims = list(target_dims) if target_dims is not None else None
        self.cn_label = int(cn_label)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        if z.size(0) == 0:
            return z.new_tensor(0.0)

        B, D = z.shape
        _validate_target_dims(self.target_dims, D)

        zt = z[:, self.target_dims].mean(dim=1)
        y = y.view(-1)
        cn = zt[y == self.cn_label]
        ad = zt[y != self.cn_label]
        if cn.numel() == 0 or ad.numel() == 0:
            return zt.new_tensor(0.0)

        diffs = cn.unsqueeze(1) - ad.unsqueeze(0)
        return F.relu(self.margin - diffs).mean()


class CovarianceSubsetLoss(nn.Module):
    """
    Cross-subset covariance loss using full cross-covariance matrices.
    Penalizes covariance between different subsets only (no within-subset penalty).
    """

    def __init__(self, subsets, lambda_off=1.0, lambda_diag=1.0, beta=0.01):
        super().__init__()
        self.subsets = {k: list(v) for k, v in subsets.items()}
        self.lambda_off = lambda_off
        self.lambda_diag = lambda_diag
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        if mu.dim() != 2:
            mu = mu.view(mu.size(0), -1)
        if logvar.dim() != 2:
            logvar = logvar.view(logvar.size(0), -1)
        B, D = mu.shape
        if B == 0:
            return mu.new_tensor(0.0)

        # Sample from posterior
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        total = mu.new_tensor(0.0)
        items = [(k, v) for k, v in self.subsets.items() if v]
        for i in range(len(items)):
            name_i, dims_i = items[i]
            if any(d < 0 or d >= D for d in dims_i):
                raise ValueError(f"subset {name_i} dims out of range for D={D}: {dims_i}")
            zi = z[:, dims_i]
            zi = zi - zi.mean(dim=0, keepdim=True)
            for j in range(i + 1, len(items)):
                name_j, dims_j = items[j]
                if any(d < 0 or d >= D for d in dims_j):
                    raise ValueError(f"subset {name_j} dims out of range for D={D}: {dims_j}")
                zj = z[:, dims_j]
                zj = zj - zj.mean(dim=0, keepdim=True)
                cov = (zi.t() @ zj) / max(B - 1, 1)  # [Di, Dj]
                total = total + self.lambda_off * torch.sum(cov.pow(2))

        return self.beta * total
