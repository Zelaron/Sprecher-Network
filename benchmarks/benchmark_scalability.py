# benchmarks/benchmark_scalability.py
"""
Find OOM breaking point for competing architectures while proving SN scales linearly.
"""
from __future__ import annotations

import csv
import gc
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    from torch.utils.checkpoint import checkpoint  # type: ignore
except Exception:  # pragma: no cover
    checkpoint = None  # type: ignore


# =============================================================================
# Registry / factory pattern
# =============================================================================

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def format_int(n: int) -> str:
    return f"{int(n):,}"


def format_mb(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:,.1f}"


# =============================================================================
# Competitors (barebones)
# =============================================================================


@register_model("StandardMLP")
class StandardMLP(nn.Module):
    """
    Barebones MLP: just Linear stacks. (No BN, no dropout, no fancy extras.)
    """

    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    @classmethod
    def param_count(cls, input_dim: int, width: int, depth: int, output_dim: int) -> int:
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        total = 0
        for i in range(len(dims) - 1):
            total += dims[i] * dims[i + 1]  # weight
            total += dims[i + 1]  # bias
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Barebones KAN (Kolmogorov-Arnold Network)
# =============================================================================


class _KANLayer(nn.Module):
    """
    Barebones KAN layer using simple B-spline basis.
    
    KAN places learnable univariate functions on edges (not nodes).
    For each edge (i, j), we have a learnable function φ_{i,j}(x_i).
    Output: y_j = Σ_i φ_{i,j}(x_i)
    
    This is O(n_in * n_out * grid_size) parameters per layer.
    No SiLU residual, no grid updates - purely for memory scaling comparison.
    """

    def __init__(self, n_in: int, n_out: int, grid_size: int = 5) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.grid_size = int(grid_size)
        
        # B-spline control points: one set per edge (n_in, n_out, grid_size)
        # This is the O(n_in * n_out) bottleneck
        self.coef = nn.Parameter(torch.empty(self.n_in, self.n_out, self.grid_size))
        
        # Fixed grid for B-spline evaluation
        grid = torch.linspace(-1.0, 1.0, self.grid_size)
        self.register_buffer("grid", grid, persistent=False)
        
        nn.init.normal_(self.coef, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in)
        B = x.shape[0]
        
        # Simple basis: use RBF-like interpolation for each edge
        # x_expanded: (B, n_in, 1)
        x_expanded = x.unsqueeze(-1)
        
        # grid: (grid_size,) -> (1, 1, grid_size)
        grid = self.grid.view(1, 1, self.grid_size)
        
        # Compute basis values using Gaussian-like kernel (simple substitute for B-splines)
        # basis: (B, n_in, grid_size)
        basis = torch.exp(-5.0 * (x_expanded - grid) ** 2)
        
        # Normalize basis
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply coefficients: coef is (n_in, n_out, grid_size)
        # basis is (B, n_in, grid_size)
        # We want: y_j = Σ_i Σ_k basis[b,i,k] * coef[i,j,k]
        # = einsum('bik,ijk->bj')
        out = torch.einsum('bik,ijk->bj', basis, self.coef)
        
        return out


@register_model("KAN")
class KANNet(nn.Module):
    """
    Barebones Kolmogorov-Arnold Network.
    No SiLU residual, no grid updates - just the core KAN structure.
    """
    
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int, grid_size: int = 5) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        self.layers = nn.ModuleList([
            _KANLayer(dims[i], dims[i + 1], grid_size=grid_size) 
            for i in range(len(dims) - 1)
        ])

    @classmethod
    def param_count(cls, input_dim: int, width: int, depth: int, output_dim: int, grid_size: int = 5) -> int:
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        total = 0
        for i in range(len(dims) - 1):
            n_in, n_out = dims[i], dims[i + 1]
            total += n_in * n_out * grid_size  # coef
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Other competitors
# =============================================================================


class _GSKANLayer(nn.Module):
    """
    GS-KAN layer proxy:
      y_q = sum_p lambda_{p,q} * psi(x_p + epsilon_q)
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        # O(N^2) weight matrix λ
        self.Lambda = nn.Parameter(torch.empty(self.n_in, self.n_out))
        # per-output shift ε
        self.epsilon = nn.Parameter(torch.empty(self.n_out))
        # shared activation ψ (cheap spline proxy)
        self.act = nn.PReLU(num_parameters=1)

        nn.init.normal_(self.Lambda, mean=0.0, std=0.02)
        nn.init.normal_(self.epsilon, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in)
        shifted_x = x.unsqueeze(-1) + self.epsilon  # (B, n_in, n_out)
        act = self.act(shifted_x)  # (B, n_in, n_out)
        out = (act * self.Lambda).sum(dim=1)  # (B, n_out)
        return out


@register_model("GSKAN")
class GSKANNet(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        self.layers = nn.ModuleList([_GSKANLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    @classmethod
    def param_count(cls, input_dim: int, width: int, depth: int, output_dim: int) -> int:
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        total = 0
        for i in range(len(dims) - 1):
            n_in, n_out = dims[i], dims[i + 1]
            total += n_in * n_out  # Lambda matrix
            total += n_out  # epsilon
            total += 1  # PReLU weight
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _SaKANLayer(nn.Module):
    """
    SaKAN proxy:
      output = spline_path(shared) + residual_path(x @ U)

    O(N^2) bottleneck is the residual matrix U (n_in x n_out).
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        # O(N^2) residual matrix
        self.U = nn.Parameter(torch.empty(self.n_in, self.n_out))

        # Minimal spline-path projection to n_out (kept O(N))
        self.v = nn.Parameter(torch.empty(self.n_out))

        # shared activation (cheap spline proxy)
        self.act = nn.PReLU(num_parameters=1)

        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.normal_(self.v, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spline path (linear memory)
        b = self.act(x)  # (B, n_in)
        s = b.sum(dim=1, keepdim=True)  # (B, 1)
        spline_out = s * self.v.view(1, -1)  # (B, n_out)

        # Residual path (dense)
        residual_out = x @ self.U  # (B, n_out)
        return spline_out + residual_out


@register_model("SaKAN")
class SaKANNet(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        self.layers = nn.ModuleList([_SaKANLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    @classmethod
    def param_count(cls, input_dim: int, width: int, depth: int, output_dim: int) -> int:
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        total = 0
        for i in range(len(dims) - 1):
            n_in, n_out = dims[i], dims[i + 1]
            total += n_in * n_out  # U
            total += n_out  # v
            total += 1  # PReLU
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Lightweight SN (Sprecher Network) – implemented from scratch for benchmarking
# =============================================================================


class _SprecherBlockLite(nn.Module):
    """
    Lightweight Sprecher block (SN) focusing ONLY on memory scaling.

    Core formula:
      y_q = Φ( Σ_p λ_p · φ(x_p + η·q) + q )

    Parameters per block: O(n_in)
      - λ vector (n_in)
      - η scalar
      - φ, Φ as shared 1D activations (PReLU(1) proxies)

    Memory:
      - forward uses output-chunking so working set is O(B*n_in*chunk)
      - training step uses checkpointing so backward does NOT store φ(x+ηq) for all q
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        q_min: float = -1.0,
        q_max: float = 1.0,
        chunk_bytes: int = 64 * 1024 * 1024,  # conservative per-chunk working-set
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.chunk_bytes = int(chunk_bytes)
        self.use_checkpoint = bool(use_checkpoint) and (checkpoint is not None)

        # O(N) params
        self.lambdas = nn.Parameter(torch.empty(self.n_in))
        self.eta = nn.Parameter(torch.tensor(0.0))

        # shared spline proxies
        self.phi = nn.PReLU(num_parameters=1)
        self.Phi = nn.PReLU(num_parameters=1)

        # fixed q-grid (buffer)
        q = torch.linspace(float(q_min), float(q_max), steps=self.n_out)
        self.register_buffer("q_values", q, persistent=False)

        nn.init.normal_(self.lambdas, mean=0.0, std=0.02)

    def _choose_chunk_size(self, batch_size: int, dtype: torch.dtype) -> int:
        # Aim: (B * n_in * chunk) * sizeof(dtype) <= chunk_bytes
        bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        denom = max(1, int(batch_size) * self.n_in * bytes_per_elem)
        chunk = max(1, int(self.chunk_bytes // denom))
        return int(min(self.n_out, max(1, chunk)))

    def _chunk_forward(self, x: torch.Tensor, q_chunk: torch.Tensor) -> torch.Tensor:
        shifted = x.unsqueeze(-1) + (self.eta * q_chunk).view(1, 1, -1)  # (B,n_in,C)
        phi_out = self.phi(shifted)
        s = (phi_out * self.lambdas.view(1, -1, 1)).sum(dim=1)  # (B,C)
        s = s + q_chunk.view(1, -1)  # (B,C)
        return self.Phi(s)  # (B,C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        dtype = x.dtype
        device = x.device

        q_vals = self.q_values.to(device=device, dtype=dtype)
        chunk = self._choose_chunk_size(B, dtype)

        out = torch.empty((B, self.n_out), device=device, dtype=dtype)

        for start in range(0, self.n_out, chunk):
            end = min(self.n_out, start + chunk)
            q_chunk = q_vals[start:end]

            if self.training and self.use_checkpoint:
                # Pass params explicitly so checkpoint sees requires_grad inputs.
                def fn(x_in: torch.Tensor, lambdas: torch.Tensor, eta: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
                    shifted = x_in.unsqueeze(-1) + (eta * q_in).view(1, 1, -1)
                    phi_out = self.phi(shifted)
                    s = (phi_out * lambdas.view(1, -1, 1)).sum(dim=1)
                    s = s + q_in.view(1, -1)
                    return self.Phi(s)

                try:
                    y_chunk = checkpoint(fn, x, self.lambdas, self.eta, q_chunk, use_reentrant=False)
                except TypeError:
                    y_chunk = checkpoint(fn, x, self.lambdas, self.eta, q_chunk)
            else:
                y_chunk = self._chunk_forward(x, q_chunk)

            out[:, start:end] = y_chunk

        return out


@register_model("SN")
class SprecherNetLite(nn.Module):
    """
    Lightweight SN stack for the benchmark:
      - no lateral mixing
      - no residuals
      - no batch norm
      - no domain updates (fixed by design)
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        *,
        chunk_bytes: int = 64 * 1024 * 1024,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        self.blocks = nn.ModuleList(
            [
                _SprecherBlockLite(
                    dims[i],
                    dims[i + 1],
                    chunk_bytes=chunk_bytes,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(len(dims) - 1)
            ]
        )

    @classmethod
    def param_count(cls, input_dim: int, width: int, depth: int, output_dim: int) -> int:
        dims = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]
        total = 0
        for i in range(len(dims) - 1):
            n_in = dims[i]
            total += n_in + 3  # lambdas(n_in) + eta(1) + phi(1) + Phi(1)
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# =============================================================================
# Benchmark harness
# =============================================================================


@dataclass
class ResultRow:
    width: int
    model: str
    params: int
    peak_mb: Optional[float]
    status: str


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_system_ram_gb() -> Optional[float]:
    try:
        if psutil is not None:
            return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        pass
    return None


def _rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _device_allocated_bytes(device: torch.device) -> Optional[int]:
    try:
        if device.type == "cuda":
            return int(torch.cuda.memory_allocated(device=device))
        if device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            return int(torch.mps.current_allocated_memory())
    except Exception:
        return None
    return None


def _device_sync(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        elif device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def empty_device_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def aggressive_cleanup(device: torch.device, sleep_time: float = 0.5) -> None:
    """
    Perform aggressive memory cleanup to avoid fragmentation issues.
    
    MPS (Apple Silicon) in particular doesn't release memory as cleanly as CUDA,
    so we need multiple GC passes and a brief delay.
    """
    # Multiple GC passes
    for _ in range(3):
        gc.collect()
    
    # Sync device to ensure all operations are complete
    _device_sync(device)
    
    # Clear device cache
    empty_device_cache(device)
    
    # Another sync after cache clear
    _device_sync(device)
    
    # Brief sleep to let the system fully reclaim memory
    if sleep_time > 0:
        time.sleep(sleep_time)
    
    # Final GC pass
    gc.collect()


def is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(
        m in msg
        for m in [
            "out of memory",
            "cuda out of memory",
            "mps backend out of memory",
            "insufficient memory",
            "cannot allocate memory",
            "failed to allocate",
        ]
    )


def print_ascii_table(rows: List[ResultRow]) -> None:
    headers = ["Width", "Model", "Params", "Peak Memory (MB)", "Status"]
    data: List[List[str]] = []
    for r in rows:
        data.append([str(r.width), r.model, format_int(r.params), format_mb(r.peak_mb), r.status])

    cols = list(zip(headers, *data))
    col_widths = [max(len(str(cell)) for cell in col) for col in cols]

    def hline(left: str, mid: str, right: str, fill: str = "-") -> str:
        parts = [fill * (w + 2) for w in col_widths]
        return left + mid.join(parts) + right

    def row_line(items: List[str]) -> str:
        cells = [f" {items[i].ljust(col_widths[i])} " for i in range(len(items))]
        return "|" + "|".join(cells) + "|"

    print()
    print(hline("+", "+", "+", "-"))
    print(row_line(headers))
    print(hline("+", "+", "+", "="))
    for d in data:
        print(row_line(d))
    print(hline("+", "+", "+", "-"))


# =============================================================================
# Survivor training
# =============================================================================


def high_dim_function(x: torch.Tensor) -> torch.Tensor:
    """A high-dimensional function from the KAN paper (adapted for 64D).

    f(x_1, ..., x_d) = exp( (1/d) * sum_{i=1}^{d} sin^2(π * x_i / 2) )

    This is a true high-dimensional function where all input dimensions
    contribute meaningfully to the output.

    Args:
        x: (B, d) tensor with values typically in [0, 1].
    Returns:
        (B, 1) tensor.
    """
    # sin^2(π * x / 2) for each dimension
    sin_sq = torch.sin(math.pi * x / 2.0) ** 2  # (B, d)
    # Mean over dimensions
    mean_sin_sq = sin_sq.mean(dim=1, keepdim=True)  # (B, 1)
    # Exponential
    return torch.exp(mean_sin_sq)  # (B, 1)


def make_high_dim_batch(
    *,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed training batch for the high-dimensional function.

    Generates inputs uniformly in [0, 1]^d and computes targets using
    the high-dimensional function from the KAN paper.
    """
    # Use a fixed seed for reproducibility within the benchmark
    generator = torch.Generator(device='cpu').manual_seed(42)
    x = torch.rand(int(batch_size), int(input_dim), generator=generator)
    x = x.to(device=device, dtype=dtype)

    y = high_dim_function(x)
    if int(output_dim) != 1:
        y = y.expand(int(batch_size), int(output_dim)).contiguous()
    return x, y


@dataclass
class TrainRow:
    model: str
    width: int
    depth: int
    params: int
    epochs: int
    final_loss: Optional[float]
    best_loss: Optional[float]
    status: str


def print_training_table(rows: List[TrainRow]) -> None:
    headers = ["Model", "Width", "Depth", "Params", "Epochs", "Final Loss", "Best Loss", "Status"]
    data: List[List[str]] = []
    for r in rows:
        data.append(
            [
                r.model,
                str(r.width),
                str(r.depth),
                format_int(r.params),
                str(r.epochs),
                "—" if r.final_loss is None else f"{r.final_loss:.4e}",
                "—" if r.best_loss is None else f"{r.best_loss:.4e}",
                r.status,
            ]
        )

    cols = list(zip(headers, *data))
    col_widths = [max(len(str(cell)) for cell in col) for col in cols]

    def hline(left: str, mid: str, right: str, fill: str = "-") -> str:
        parts = [fill * (w + 2) for w in col_widths]
        return left + mid.join(parts) + right

    def row_line(items: List[str]) -> str:
        cells = [f" {items[i].ljust(col_widths[i])} " for i in range(len(items))]
        return "|" + "|".join(cells) + "|"

    print()
    print(hline("+", "+", "+", "-"))
    print(row_line(headers))
    print(hline("+", "+", "+", "="))
    for d in data:
        print(row_line(d))
    print(hline("+", "+", "+", "-"))


def train_survivors(
    *,
    survivors: List[str],
    width: int,
    depth: int,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int = 400,
    lr: float = 1e-3,
) -> List[TrainRow]:
    if not survivors:
        return []

    print("\n=== Training Survivors ===")
    print(f"Target function: f(x) = exp( (1/{input_dim}) * Σ sin²(π·x_i/2) ), x ∈ [0,1]^{input_dim}")
    print(f"Train config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Survivors at width={width}: {survivors}\n")

    x_train, y_train = make_high_dim_batch(
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        dtype=dtype,
    )

    rows: List[TrainRow] = []
    for model_name in survivors:
        cls = MODEL_REGISTRY[model_name]
        params = int(getattr(cls, "param_count")(input_dim, width, depth, output_dim))

        # Aggressive cleanup before each training run
        aggressive_cleanup(device, sleep_time=0.5)

        status = "OK"
        final_loss: Optional[float] = None
        best_loss: Optional[float] = None

        print(f"--- Training {model_name} (w={width}, d={depth}, params={format_int(params)}) ---")

        try:
            model = cls(input_dim=input_dim, width=width, depth=depth, output_dim=output_dim)  # type: ignore[arg-type]
            model.to(device=device, dtype=dtype)
            model.train()

            opt = torch.optim.Adam(model.parameters(), lr=lr)

            best = float("inf")
            last = float("nan")

            for epoch in range(int(epochs)):
                opt.zero_grad(set_to_none=True)
                pred = model(x_train)
                loss = F.mse_loss(pred, y_train)
                loss.backward()
                opt.step()

                loss_val = float(loss.item())
                last = loss_val
                if loss_val < best:
                    best = loss_val

                # Print each epoch on its own line
                print(f"  Epoch {epoch + 1:3d}/{epochs}: loss={loss_val:.4e}, best={best:.4e}")

            final_loss = float(last)
            best_loss = float(best)

        except RuntimeError as e:
            if is_oom_error(e):
                status = "OOM"
                print(f"  OOM error during training!")
            else:
                status = f"ERROR({type(e).__name__})"
                print(f"  Error: {e}")
        except Exception as e:
            status = f"ERROR({type(e).__name__})"
            print(f"  Error: {e}")
        finally:
            # Free memory as much as possible
            try:
                del model  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                del opt  # type: ignore[possibly-undefined]
            except Exception:
                pass
            aggressive_cleanup(device, sleep_time=0.1)

        rows.append(
            TrainRow(
                model=model_name,
                width=width,
                depth=depth,
                params=params,
                epochs=int(epochs),
                final_loss=final_loss,
                best_loss=best_loss,
                status=status,
            )
        )

    print_training_table(rows)
    return rows


def run_single_model_test(
    *,
    model_name: str,
    width: int,
    input_dim: int,
    depth: int,
    output_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ResultRow:
    """Run memory test for a single model at a given width."""
    cls = MODEL_REGISTRY[model_name]
    params = int(getattr(cls, "param_count")(input_dim, width, depth, output_dim))

    # Aggressive cleanup before this model
    aggressive_cleanup(device, sleep_time=0.5)

    # Get fresh baseline after cleanup
    _device_sync(device)
    base_dev = _device_allocated_bytes(device) or 0
    base_rss = _rss_bytes() or 0
    peak_dev = base_dev
    peak_rss = base_rss

    def sample_peak() -> None:
        nonlocal peak_dev, peak_rss
        _device_sync(device)
        dv = _device_allocated_bytes(device)
        rs = _rss_bytes()
        if dv is not None:
            peak_dev = max(peak_dev, dv)
        if rs is not None:
            peak_rss = max(peak_rss, rs)

    status = "OK"
    peak_mb: Optional[float] = None

    try:
        model = cls(input_dim=input_dim, width=width, depth=depth, output_dim=output_dim)  # type: ignore[arg-type]
        model.to(device=device, dtype=dtype)

        x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
        y = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

        sample_peak()

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)

        pred = model(x)
        sample_peak()
        loss = F.mse_loss(pred, y)
        sample_peak()
        loss.backward()
        sample_peak()
        opt.step()
        sample_peak()

        # Only compute peak_mb for successful runs
        dev_delta = max(0, peak_dev - base_dev)
        rss_delta = max(0, peak_rss - base_rss)
        peak_bytes = max(dev_delta, rss_delta)
        peak_mb = float(peak_bytes) / (1024**2)

    except RuntimeError as e:
        if is_oom_error(e):
            status = "OOM"
            peak_mb = None
        else:
            status = f"ERROR({type(e).__name__})"
            peak_mb = None

    except Exception as e:
        status = f"ERROR({type(e).__name__})"
        peak_mb = None

    finally:
        # Aggressive cleanup after this model
        try:
            del model  # type: ignore[possibly-undefined]
        except Exception:
            pass
        try:
            del opt  # type: ignore[possibly-undefined]
        except Exception:
            pass
        try:
            del x, y, pred, loss  # type: ignore[possibly-undefined]
        except Exception:
            pass
        aggressive_cleanup(device, sleep_time=0.3)

    return ResultRow(width=width, model=model_name, params=params, peak_mb=peak_mb, status=status)


def run_width_test(
    *,
    width: int,
    model_names: List[str],
    oomed_models: set,
    input_dim: int,
    depth: int,
    output_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[ResultRow]:
    """Run memory test for all models at a given width."""
    results: List[ResultRow] = []

    for model_name in model_names:
        cls = MODEL_REGISTRY[model_name]
        params = int(getattr(cls, "param_count")(input_dim, width, depth, output_dim))

        # Skip future widths once a model has OOM'd once
        if model_name in oomed_models:
            row = ResultRow(width=width, model=model_name, params=params, peak_mb=None, status="SKIP(after OOM)")
            results.append(row)
            print(
                f"width={width:<8} model={model_name:<12} "
                f"params={format_int(params):>14}  peak_mb={'—':>10}  status={row.status}"
            )
            continue

        # Run the test for this single model (with aggressive cleanup)
        row = run_single_model_test(
            model_name=model_name,
            width=width,
            input_dim=input_dim,
            depth=depth,
            output_dim=output_dim,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        
        if row.status == "OOM":
            oomed_models.add(model_name)
        
        results.append(row)
        print(
            f"width={width:<8} model={model_name:<12} "
            f"params={format_int(row.params):>14}  peak_mb={format_mb(row.peak_mb):>10}  status={row.status}"
        )

    return results


def main() -> None:
    torch.manual_seed(0)

    depth = 3
    batch_size = 32
    input_dim = 64
    output_dim = 1
    start_width = 512

    device = get_device()
    dtype = torch.float32
    sys_ram = get_system_ram_gb()
    sys_ram_str = f"{sys_ram:.2f} GB" if sys_ram is not None else "unknown"

    model_names = list(MODEL_REGISTRY.keys())

    print("\n=== Memory Battle Royale ===")
    print(f"Device: {device} | dtype: {dtype} | System RAM: {sys_ram_str}")
    print(f"Config: batch_size={batch_size}, input_dim={input_dim}, depth={depth}, output_dim={output_dim}")
    print(f"Strategy: Start at width={start_width}, double until only one survivor")
    print(f"Models: {model_names}\n")

    all_results: List[ResultRow] = []
    oomed_models: set[str] = set()
    
    width = start_width
    final_width = start_width
    winner: Optional[str] = None
    
    # Track last round's results for tiebreaker
    last_round_results: Dict[str, ResultRow] = {}

    while True:
        # Run test at current width
        round_results = run_width_test(
            width=width,
            model_names=model_names,
            oomed_models=oomed_models,
            input_dim=input_dim,
            depth=depth,
            output_dim=output_dim,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        all_results.extend(round_results)

        # Count survivors (models that completed OK at this width)
        survivors = [r.model for r in round_results if r.status == "OK"]
        num_survivors = len(survivors)

        print(f"\n>>> Width {width}: {num_survivors} survivor(s): {survivors}\n")

        if num_survivors == 0:
            # ALL models OOM'd at this width
            # Winner is the one with lowest peak memory from the previous round
            if last_round_results:
                # Find model with lowest peak_mb from last round (among those that were OK)
                valid_last = [(m, r) for m, r in last_round_results.items() 
                              if r.status == "OK" and r.peak_mb is not None]
                if valid_last:
                    winner = min(valid_last, key=lambda x: x[1].peak_mb)[0]  # type: ignore
                    final_width = width // 2
                else:
                    winner = None
                    final_width = width // 2
            print(f"All models OOM'd at width={width}.")
            print(f"Winner (lowest peak memory at width={final_width}): {winner}")
            break
        
        elif num_survivors == 1:
            # Exactly one survivor - we have a winner!
            winner = survivors[0]
            final_width = width
            print(f"Single survivor found: {winner} at width={width}")
            break
        
        else:
            # Multiple survivors - save results and continue doubling
            last_round_results = {r.model: r for r in round_results}
            width *= 2
            final_width = width // 2

    print_ascii_table(all_results)

    out_csv = "benchmark_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["width", "model", "params", "peak_mb", "status"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "width": r.width,
                    "model": r.model,
                    "params": r.params,
                    "peak_mb": ("" if r.peak_mb is None else f"{r.peak_mb:.6f}"),
                    "status": r.status,
                }
            )

    print(f"\nSaved CSV: {out_csv}")

    # ---------------------------------------------------------------------
    # Train survivor(s) at the final width where they were the last survivor
    # ---------------------------------------------------------------------
    if winner:
        # Aggressive cleanup before training
        aggressive_cleanup(device, sleep_time=1.0)

        _ = train_survivors(
            survivors=[winner],
            width=final_width,
            depth=depth,
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            dtype=dtype,
            epochs=400,
            lr=1e-3,
        )
    else:
        print(f"\nNo clear winner; skipping training.")


if __name__ == "__main__":
    main()
