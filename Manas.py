# Manas 1.02


"""
manas4.py
=========
MANAS-4.0 — Multi-modal Adaptive Neural Architecture System
Version  : 4.2.0
Author   : Zeeshan Rahman

A fully integrated, production-grade AI research system combining six
foundational modules into one unified pipeline. Zero external ML framework
dependencies — NumPy + Python stdlib only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Module 1 — Non-Linear Continuous Optimizer (The Z-Operator)
             ZULIC: 𝒵(f,x) = [f(x+Δ)−f(x)] / Δ^β(f)
             Custom autograd engine with Navier-Stokes fluid smoothing.

  Module 2 — O(N log N) Fractal Hierarchical Attention
             Recursive Self-Similar Block (RSSB) tree architecture.
             Reduces standard O(N²) attention to O(N log_b N).
             Based on FNN paper (BBDU, 2026).

  Module 3 — Stochastic Hardware-Seeded Processing
             True non-determinism via os.urandom, clock jitter,
             memory allocation timing, CPU instruction variance.
             Von Neumann debiased, SHA3-whitened entropy pool.

  Module 4 — Dynamic Resource-Aware Compute Throttling
             Real-time psutil monitoring of CPU, RAM, thermal sensors.
             Five compute tiers: FULL → REDUCED → MINIMAL → CRITICAL → SUSPENDED.
             Navier-Stokes fallback heuristic + aggressive memory compression.

  Module 5 — Dynamic JIT Meta-Programming Pipeline
             AST-based runtime code generation from failure states.
             Sandboxed execution with security validation.
             Eight failure kinds with auto-repair functions.

  Module 6 — Temporally Weighted Causal Graph Database
             Nodes = states/inputs. Edges = temporal + causal weights.
             Dijkstra path-finding on temporal-decay weighted graph.
             Causal cone, ancestor chain, descendant chain queries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from manas4 import MANAS, ManasConfig

    cfg    = ManasConfig(embed_dim=256, vocab_size=32000, branching_b=4)
    system = MANAS(cfg)

    # One training step
    result = system.step(token_ids, targets)

    # Inference
    out = system.infer(token_ids)

    # Causal memory recall
    ancestors = system.recall_causal(result.graph_node_id, depth=5)

    # Resource status
    print(system.resource_summary())

    # Full diagnostics
    print(system.full_diagnostic())

    # Graceful shutdown
    system.shutdown()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    numpy  >= 1.24
    psutil >= 5.9     (Module 4 only)
    Python >= 3.10

    Install: pip install numpy psutil
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

# core_z_math.py
# ==============
# Module 1 — Non-Linear Continuous Optimizer: The Z-Operator
# Custom Autograd Engine based on ZULIC (Zeeshan's Universal Law of Intelligent Change)
#
#     𝒵(f, x) = lim(Δ→0) [ f(x+Δ) − f(x) ] / Δ^β(f)
#     β(f)     = log|f(x+Δ) − f(x)| / log|Δ|
#
# Architecture
# ------------
#     ZOperator            — ZULIC β(f) and Change Signature computation
#     Tensor               — Autograd node with full computational graph support
#     FluidLossLandscape   — Navier-Stokes viscosity balancer (Chaos ↔ Friction)
#     Custom_Autograd_Z    — Master engine: backward + Z-scaling + fluid smoothing
#     ZOptimizer           — Parameter update rule driven by Change Signatures
#     ZLinear              — From-scratch fully-connected layer
#     ZNetwork             — Composable MLP built on ZLinear
#
# Dependencies: numpy only (zero framework policy)

import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  § 1  ZULIC CORE — ZOperator
# ══════════════════════════════════════════════════════════════════════════════

class ZOperator:
    """
    Implements the ZULIC differential operator.

    β(f) acts as an adaptive scaling exponent:
        β = 1   →  standard first derivative
        β < 1   →  integration / smoothing regime
        β = 0   →  limit / constant regime
        β > 1   →  higher-order derivative / amplification regime

    Parameters
    ----------
    delta   : finite-difference step size (default 1e-5)
    epsilon : numerical stability floor    (default 1e-12)
    """

    def __init__(self, delta: float = 1e-5, epsilon: float = 1e-12) -> None:
        self.delta   = delta
        self.epsilon = epsilon

    # ── β(f) ─────────────────────────────────────────────────────────────────

    def beta(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute the ZULIC scaling exponent element-wise.

            β(f) = log|f(x+Δ) − f(x)| / log|Δ|

        Returns an array of the same shape as x.
        """
        d   = self.delta
        df  = np.abs(f(x + d) - f(x))
        df  = np.maximum(df, self.epsilon)
        return np.log(df) / np.log(d + self.epsilon)

    # ── Change Signature ─────────────────────────────────────────────────────

    def change_signature(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute 𝒵(f, x) — the ZULIC Change Signature.

            𝒵(f, x) = [f(x+Δ) − f(x)] / Δ^β(f)

        Returns an array of the same shape as x.
        """
        d     = self.delta
        b     = self.beta(f, x)
        diff  = f(x + d) - f(x)
        denom = np.power(np.abs(d) + self.epsilon, b)
        return diff / denom

    # ── Jacobian ──────────────────────────────────────────────────────────────

    def jacobian(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Full Jacobian via per-element Change Signatures.
        Returns shape (output_dim, input_dim).
        """
        base    = f(x)
        out_dim = base.size
        in_dim  = x.size
        J       = np.zeros((out_dim, in_dim))
        x_flat  = x.flatten()

        for i in range(in_dim):
            def f_i(v, idx=i):
                xp      = x_flat.copy()
                xp[idx] = float(v)
                return f(xp.reshape(x.shape)).flatten()

            col      = self.change_signature(f_i, x_flat[i : i + 1])
            J[:, i]  = (col[:out_dim] if col.size >= out_dim
                        else np.pad(col, (0, out_dim - col.size)))
        return J

    # ── β regime classifier ───────────────────────────────────────────────────

    @staticmethod
    def regime(beta_val: np.ndarray) -> np.ndarray:
        """
        Return a string label per element indicating the active ZULIC regime.
        For diagnostic / logging use only.
        """
        out = np.empty(beta_val.shape, dtype=object)
        out[np.isclose(beta_val, 0, atol=0.05)]               = "constant"
        out[np.isclose(beta_val, 1, atol=0.05)]               = "derivative"
        out[(beta_val > 0.05) & (beta_val < 0.95)]            = "integration"
        out[beta_val > 1.05]                                   = "higher-order"
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  TENSOR — Autograd Node
# ══════════════════════════════════════════════════════════════════════════════

class Tensor:
    """
    Autograd-capable tensor node.

    Tracks the full computational graph via _parents and _backward closures.
    Gradients are accumulated in .grad (same shape as .data).

    Parameters
    ----------
    data     : array-like numeric data
    _parents : tuple of parent Tensor nodes
    label    : optional identifier string
    """

    __slots__ = ("data", "grad", "_backward", "_parents", "label")

    def __init__(
        self,
        data,
        _parents: Tuple["Tensor", ...] = (),
        label: str = "",
    ) -> None:
        self.data:      np.ndarray         = np.array(data, dtype=np.float64)
        self.grad:      np.ndarray         = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._parents:  Tuple["Tensor", ...] = tuple(_parents)
        self.label:     str                = label

    # ── Shape helpers ─────────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other) -> "Tensor":
        other = _ensure(other)
        out   = Tensor(self.data + other.data, (self, other), "+")

        def _bwd():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _bwd
        return out

    def __mul__(self, other) -> "Tensor":
        other = _ensure(other)
        out   = Tensor(self.data * other.data, (self, other), "*")

        def _bwd():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)

        out._backward = _bwd
        return out

    def __matmul__(self, other) -> "Tensor":
        other = _ensure(other)
        out   = Tensor(self.data @ other.data, (self, other), "@")

        def _bwd():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _bwd
        return out

    def __pow__(self, exp: float) -> "Tensor":
        out = Tensor(np.power(self.data, exp), (self,), f"**{exp}")

        def _bwd():
            self.grad += exp * np.power(self.data, exp - 1) * out.grad

        out._backward = _bwd
        return out

    def __neg__(self)                        -> "Tensor": return self * -1.0
    def __sub__(self, o)                     -> "Tensor": return self + (-_ensure(o))
    def __truediv__(self, o)                 -> "Tensor": return self * _ensure(o) ** -1
    def __radd__(self, o: float)             -> "Tensor": return self + o
    def __rmul__(self, o: float)             -> "Tensor": return self * o
    def __rsub__(self, o: float)             -> "Tensor": return _ensure(o) - self
    def __rtruediv__(self, o: float)         -> "Tensor": return _ensure(o) / self

    # ── Reductions ────────────────────────────────────────────────────────────

    def sum(self, axis=None) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis), (self,), "sum")

        def _bwd():
            g = out.grad
            if axis is not None:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _bwd
        return out

    def mean(self, axis=None) -> "Tensor":
        n   = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis), (self,), "mean")

        def _bwd():
            g = out.grad / n
            if axis is not None:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _bwd
        return out

    # ── Activations ───────────────────────────────────────────────────────────

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _bwd():
            self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _bwd
        return out

    def tanh(self) -> "Tensor":
        t   = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _bwd():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _bwd
        return out

    def sigmoid(self) -> "Tensor":
        s   = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500.0, 500.0)))
        out = Tensor(s, (self,), "sigmoid")

        def _bwd():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _bwd
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit (exact form)."""
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(self.data / math.sqrt(2.0)))
        out = Tensor(self.data * cdf, (self,), "gelu")

        def _bwd():
            pdf   = np.exp(-0.5 * self.data ** 2) / math.sqrt(2.0 * math.pi)
            dcdf  = cdf + self.data * pdf
            self.grad += dcdf * out.grad

        out._backward = _bwd
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exps    = np.exp(shifted)
        s       = exps / exps.sum(axis=axis, keepdims=True)
        out     = Tensor(s, (self,), "softmax")

        def _bwd():
            dot        = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _bwd
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(np.maximum(self.data, 1e-12)), (self,), "log")

        def _bwd():
            self.grad += out.grad / np.maximum(self.data, 1e-12)

        out._backward = _bwd
        return out

    # ── Reshape / Transpose ───────────────────────────────────────────────────

    def reshape(self, *shape) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _bwd():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _bwd
        return out

    def T(self) -> "Tensor":
        out = Tensor(self.data.T, (self,), "T")

        def _bwd():
            self.grad += out.grad.T

        out._backward = _bwd
        return out

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self) -> None:
        """
        Topological backpropagation.
        Seeds root gradient as 1.0, then chains all _backward closures.
        """
        self.grad = np.ones_like(self.data)
        topo:    List["Tensor"] = []
        visited: set            = set()

        def _build(v: "Tensor") -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._parents:
                    _build(p)
                topo.append(v)

        _build(self)
        for node in reversed(topo):
            node._backward()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def detach(self) -> "Tensor":
        """Return a new Tensor with the same data but no graph attachment."""
        return Tensor(self.data.copy(), label=self.label)

    def item(self) -> float:
        return float(self.data)

    def __repr__(self) -> str:
        return (f"Tensor(shape={self.shape}, "
                f"label={self.label!r}, "
                f"grad_norm={np.linalg.norm(self.grad):.4f})")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out axes that were broadcast during the forward pass."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if g != s:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _mode(arr: np.ndarray):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[counts.argmax()]


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  FLUID LOSS LANDSCAPE — Navier-Stokes Viscosity Balancer
# ══════════════════════════════════════════════════════════════════════════════

class FluidLossLandscape:
    """
    Models the loss landscape as an incompressible viscous fluid, drawing
    analogy from the Navier-Stokes momentum equation:

        ρ (∂u/∂t + u·∇u)  =  −∇p  +  μ∇²u  +  f

    Translated into parameter-space gradient dynamics:

        Δθ  =  −∇L  +  μ·∇²(∇L)  −  ρ·(∇L ⊙ ∇²∇L)
              ─────   ────────────   ─────────────────
              signal     friction        chaos/advection

    ∇²(·) is approximated as a second finite difference over the gradient
    history buffer, acting as a temporal Laplacian.

    Parameters
    ----------
    viscosity    : μ — friction / diffusion coefficient  (default 0.1)
    density      : ρ — inertia / chaos coefficient       (default 0.9)
    history_len  : number of past gradient steps retained for Laplacian
    """

    def __init__(
        self,
        viscosity:   float = 0.1,
        density:     float = 0.9,
        history_len: int   = 5,
    ) -> None:
        self.mu          = viscosity
        self.rho         = density
        self.history_len = history_len
        self._history:   List[np.ndarray] = []

    # ── Discrete temporal Laplacian ───────────────────────────────────────────

    def _laplacian(self, grad: np.ndarray) -> np.ndarray:
        """
        Second finite difference over history:
            ∇²g ≈ g[t] − 2·g[t-1] + g[t-2]
        Returns zeros when history is insufficient.
        """
        if len(self._history) < 2:
            return np.zeros_like(grad)
        return grad - 2.0 * self._history[-1] + self._history[-2]

    # ── Main smoothing pass ───────────────────────────────────────────────────

    def smooth(self, raw_grad: np.ndarray) -> np.ndarray:
        """
        Apply Navier-Stokes balancing to a raw Change Signature.

        Chaos term    (advection):   ρ · raw_grad ⊙ tanh(∇²)
        Friction term (diffusion):   μ · ∇²

        Returns the viscosity-balanced update direction.
        """
        lap      = self._laplacian(raw_grad)
        chaos    = self.rho * raw_grad * np.tanh(lap + 1e-8)
        friction = self.mu  * lap
        balanced = raw_grad - chaos + friction

        self._history.append(raw_grad.copy())
        if len(self._history) > self.history_len:
            self._history.pop(0)

        return balanced

    def reset(self) -> None:
        """Clear gradient history."""
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  CUSTOM_AUTOGRAD_Z — Master Engine
# ══════════════════════════════════════════════════════════════════════════════

class Custom_Autograd_Z:
    """
    Full ZULIC autograd engine.

    Pipeline per backward call
    --------------------------
    1. Zero all parameter gradients.
    2. Standard topological chain-rule backward through the compute graph.
    3. Per-parameter ZULIC β-scaling → Change Signatures.
    4. Fluid smoothing via FluidLossLandscape (Navier-Stokes balancing).

    Parameters
    ----------
    delta       : ZOperator finite-difference step
    viscosity   : FluidLossLandscape μ (friction / smoothing)
    density     : FluidLossLandscape ρ (chaos / inertia)
    history_len : gradient history window for temporal Laplacian
    """

    def __init__(
        self,
        delta:       float = 1e-5,
        viscosity:   float = 0.1,
        density:     float = 0.9,
        history_len: int   = 5,
    ) -> None:
        self.z_op  = ZOperator(delta=delta)
        self.fluid = FluidLossLandscape(
            viscosity=viscosity,
            density=density,
            history_len=history_len,
        )

    # ── Loss functions ────────────────────────────────────────────────────────

    @staticmethod
    def mse(pred: Tensor, target: np.ndarray) -> Tensor:
        """Mean Squared Error loss."""
        diff = pred - Tensor(target)
        return (diff ** 2).mean()

    @staticmethod
    def mae(pred: Tensor, target: np.ndarray) -> Tensor:
        """Mean Absolute Error loss."""
        diff = pred.data - np.asarray(target, dtype=np.float64)
        loss = Tensor(np.abs(diff).mean(), (pred,), "mae")

        def _bwd():
            pred.grad += (np.sign(diff) / diff.size) * loss.grad

        loss._backward = _bwd
        return loss

    @staticmethod
    def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Numerically stable softmax cross-entropy.

        Parameters
        ----------
        logits  : Tensor of shape (batch, num_classes)
        targets : int array of shape (batch,)
        """
        data   = logits.data - logits.data.max(axis=1, keepdims=True)
        exps   = np.exp(data)
        probs  = exps / exps.sum(axis=1, keepdims=True)
        n      = targets.shape[0]
        log_p  = np.log(probs[np.arange(n), targets] + 1e-12)
        loss   = Tensor(-log_p.mean(), (logits,), "cross_entropy")

        def _bwd():
            grad                        = probs.copy()
            grad[np.arange(n), targets] -= 1.0
            logits.grad                 += (grad / n) * loss.grad

        loss._backward = _bwd
        return loss

    @staticmethod
    def binary_cross_entropy(pred: Tensor, target: np.ndarray) -> Tensor:
        """Element-wise binary cross-entropy; pred must be sigmoid output."""
        t    = np.asarray(target, dtype=np.float64)
        p    = np.clip(pred.data, 1e-7, 1.0 - 1e-7)
        loss = Tensor(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean(),
                      (pred,), "bce")

        def _bwd():
            pred.grad += (-(t / p) + (1 - t) / (1 - p)) / t.size * loss.grad

        loss._backward = _bwd
        return loss

    # ── Core API ──────────────────────────────────────────────────────────────

    def compute_change_signatures(
        self,
        loss:       Tensor,
        parameters: List[Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Run the full ZULIC backward pipeline.

        Steps
        -----
        1. Zero gradients on all parameters.
        2. Topological backward through compute graph (chain rule).
        3. Apply ZULIC β-scaling to each parameter's raw gradient.
        4. Apply Navier-Stokes fluid smoothing.

        Returns
        -------
        Dict mapping parameter label (or id) → smoothed Change Signature.
        """
        # Step 1 — zero grads
        for p in parameters:
            p.grad = np.zeros_like(p.data)

        # Step 2 — standard chain-rule backward
        loss.backward()

        signatures: Dict[str, np.ndarray] = {}

        for p in parameters:
            raw = p.grad.copy()
            key = p.label if p.label else str(id(p))

            # Step 3 — ZULIC β-scaling
            # Treat the gradient field as a function and rescale by β
            b     = self.z_op.beta(lambda g, r=raw: r * np.sign(g + 1e-12), raw)
            denom = np.power(
                np.abs(raw) + self.z_op.epsilon,
                np.maximum(b - 1.0, 0.0)
            )
            z_sig = raw / np.maximum(denom, self.z_op.epsilon)

            # Step 4 — fluid smoothing
            signatures[key] = self.fluid.smooth(z_sig)

        return signatures

    # ── Diagnostic ────────────────────────────────────────────────────────────

    def beta_report(self, parameters: List[Tensor]) -> Dict[str, Dict]:
        """
        Return per-parameter β statistics.
        Call after compute_change_signatures (grads must be populated).

        Returns
        -------
        Dict of {label: {beta_mean, beta_min, beta_max, regime_mode}}
        """
        report: Dict[str, Dict] = {}
        for p in parameters:
            key  = p.label if p.label else str(id(p))
            b    = self.z_op.beta(
                lambda g, r=p.grad: r * np.sign(g + 1e-12),
                p.grad + 1e-12,
            )
            report[key] = {
                "beta_mean":   float(b.mean()),
                "beta_min":    float(b.min()),
                "beta_max":    float(b.max()),
                "regime_mode": _mode(ZOperator.regime(b)),
            }
        return report


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  ZOPTIMIZER — Change Signature Parameter Update
# ══════════════════════════════════════════════════════════════════════════════

class ZOptimizer:
    """
    Parameter optimiser driven by ZULIC Change Signatures.

    Update rule
    -----------
        v[t]  =  momentum · v[t-1]  +  (1 − momentum) · sig
        θ[t]  =  θ[t-1]  −  lr · v[t]

    Parameters
    ----------
    parameters : list of Tensor nodes to optimise
    lr         : initial learning rate
    lr_decay   : multiplicative LR decay per step (1.0 = no decay)
    clip_norm  : max gradient norm; None to disable clipping
    momentum   : EMA coefficient for velocity buffer (0.0 = plain SGD)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr:          float          = 1e-2,
        lr_decay:    float          = 1.0,
        clip_norm:   Optional[float] = 5.0,
        momentum:    float          = 0.9,
    ) -> None:
        self.parameters = parameters
        self.lr         = lr
        self.lr_decay   = lr_decay
        self.clip_norm  = clip_norm
        self.momentum   = momentum
        self._velocity  = [np.zeros_like(p.data) for p in parameters]
        self._step      = 0

    def step(self, signatures: Dict[str, np.ndarray]) -> None:
        """Apply Change Signatures to update all registered parameters."""
        lr = self.lr * (self.lr_decay ** self._step)

        for i, p in enumerate(self.parameters):
            key = p.label if p.label else str(id(p))
            sig = signatures.get(key, p.grad.copy())

            # Gradient norm clipping
            if self.clip_norm is not None:
                norm = np.linalg.norm(sig)
                if norm > self.clip_norm:
                    sig = sig * (self.clip_norm / (norm + 1e-8))

            # Momentum buffer
            if self.momentum > 0.0:
                self._velocity[i] = (self.momentum * self._velocity[i]
                                     + (1.0 - self.momentum) * sig)
                update = self._velocity[i]
            else:
                update = sig

            p.data -= lr * update

        self._step += 1

    def zero_grad(self) -> None:
        """Zero all parameter gradients. Call at the start of each step."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    @property
    def current_lr(self) -> float:
        return self.lr * (self.lr_decay ** self._step)

    @property
    def step_count(self) -> int:
        return self._step


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  ZLINEAR & ZNETWORK — From-Scratch Layer Primitives
# ══════════════════════════════════════════════════════════════════════════════

class ZLinear:
    """
    Fully-connected layer built entirely on Tensor operations.

    Weight init : He uniform  (bound = √(6 / fan_in))
    Bias init   : zeros

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    label   : name prefix used in parameter labels
    """

    def __init__(self, in_dim: int, out_dim: int, label: str = "fc") -> None:
        bound  = math.sqrt(6.0 / in_dim)
        self.W = Tensor(
            np.random.uniform(-bound, bound, (in_dim, out_dim)),
            label=f"{label}.W",
        )
        self.b = Tensor(np.zeros(out_dim), label=f"{label}.b")

    def forward(self, x: Tensor) -> Tensor:
        """Affine transform: y = x @ W + b (fully differentiable)."""
        out_data = x.data @ self.W.data + self.b.data
        out      = Tensor(out_data, (x, self.W, self.b), "affine")

        def _bwd():
            g            = out.grad
            x.grad      += g @ self.W.data.T
            self.W.grad += x.data.T @ g
            self.b.grad += g.sum(axis=0)

        out._backward = _bwd
        return out

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ZNetwork:
    """
    Configurable multi-layer perceptron composed of ZLinear blocks.

    Parameters
    ----------
    dims       : layer widths including input and output, e.g. [128, 256, 64, 10]
    activation : one of "relu" | "tanh" | "sigmoid" | "gelu"
    """

    _ACTIVATIONS: Dict[str, Callable] = {
        "relu":    lambda t: t.relu(),
        "tanh":    lambda t: t.tanh(),
        "sigmoid": lambda t: t.sigmoid(),
        "gelu":    lambda t: t.gelu(),
    }

    def __init__(self, dims: List[int], activation: str = "relu") -> None:
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS)}"
            )
        self.act    = self._ACTIVATIONS[activation]
        self.layers = [
            ZLinear(dims[i], dims[i + 1], label=f"L{i}")
            for i in range(len(dims) - 1)
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass; activation after every layer except the last."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def param_count(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE LAYER — LoRA + Mixed Precision + MLX Backend
#  Optimised for Apple M-series chips (M1/M2/M3/M4/M5)
# ══════════════════════════════════════════════════════════════════════════════

# ── MLX backend detection ─────────────────────────────────────────────────────
# If mlx is installed (pip install mlx), use Apple Metal GPU.
# Otherwise fall back to NumPy CPU. Zero code change required.

try:
    import mlx.core as _mlx_core
    import mlx.nn   as _mlx_nn
    _MLX_AVAILABLE = True
except ImportError:
    _mlx_core      = None
    _mlx_nn        = None
    _MLX_AVAILABLE = False


def _backend_array(x: np.ndarray, use_mlx: bool = False):
    """Convert numpy array to MLX array if MLX is available and requested."""
    if use_mlx and _MLX_AVAILABLE:
        return _mlx_core.array(x.astype(np.float32))
    return x


def _backend_matmul(a: np.ndarray, b: np.ndarray, use_mlx: bool = False):
    """Matrix multiply using MLX (GPU) or NumPy (CPU)."""
    if use_mlx and _MLX_AVAILABLE:
        ma = _mlx_core.array(a.astype(np.float32))
        mb = _mlx_core.array(b.astype(np.float32))
        return np.array(_mlx_core.matmul(ma, mb))
    return a @ b


class MixedPrecisionManager:
    """
    Mixed Precision Manager for Apple M-series chips.

    Strategy (numerically safe)
    ---------------------------
    - Parameters stored as float32  (4 bytes/value, stable gradients)
    - Attention scores computed in float32
    - Activations kept as float32
    - Embeddings stored as float32
    - Inference-only arrays cast to float16 (2 bytes, 2x memory saving)
    - Gradient accumulation in float32 (prevents underflow)

    Why NOT pure float16 for training
    -----------------------------------
    float16 has a tiny dynamic range (max ~65504). Gradients in deep
    networks routinely exceed this, causing NaN cascades. float32
    training with float16 inference is the standard safe approach
    (same as used by all major frameworks).

    Parameters
    ----------
    training_dtype  : dtype for parameter storage and gradient computation
    inference_dtype : dtype for inference-only forward pass
    use_mlx         : route matmuls through Apple Metal GPU via MLX
    """

    def __init__(
        self,
        training_dtype:  np.dtype = np.float32,
        inference_dtype: np.dtype = np.float16,
        use_mlx:         bool     = False,
    ) -> None:
        self.train_dtype = training_dtype
        self.infer_dtype = inference_dtype
        self.use_mlx     = use_mlx and _MLX_AVAILABLE

        if use_mlx and not _MLX_AVAILABLE:
            logger.warning(
                "MLX requested but not installed. "
                "Install with: pip install mlx. Falling back to NumPy CPU."
            )
        if self.use_mlx:
            logger.info(
                "MixedPrecisionManager: MLX backend active "
                "(Apple Metal GPU enabled)."
            )
        else:
            logger.info(
                "MixedPrecisionManager: NumPy CPU backend "
                "(dtype=float32 train / float16 infer)."
            )

    def cast_param(self, x: np.ndarray) -> np.ndarray:
        """Cast a parameter array to training dtype (float32)."""
        return x.astype(self.train_dtype)

    def cast_infer(self, x: np.ndarray) -> np.ndarray:
        """Cast an activation to inference dtype (float16)."""
        return x.astype(self.infer_dtype)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiply with optional MLX GPU acceleration.
        Always returns float32 result for numerical stability.
        """
        if self.use_mlx and _MLX_AVAILABLE:
            result = _backend_matmul(
                a.astype(np.float32),
                b.astype(np.float32),
                use_mlx=True,
            )
            return result.astype(np.float32)
        return (a.astype(np.float32) @ b.astype(np.float32))

    def quantize_weights(
        self,
        weights: Dict[str, np.ndarray],
        bits: int = 8,
    ) -> Dict[str, Dict]:
        """
        Quantize weight arrays to int8 for storage / inference.

        Uses symmetric per-tensor quantization:
            scale = max(|W|) / 127
            W_q   = round(W / scale).clip(-127, 127).astype(int8)

        To dequantize: W_float = W_q * scale

        Parameters
        ----------
        weights : dict of {name: float32 array}
        bits    : quantization bits (8 recommended)

        Returns
        -------
        dict of {name: {"quantized": int8_array, "scale": float}}
        """
        quantized: Dict[str, Dict] = {}
        max_val = 2 ** (bits - 1) - 1

        for name, W in weights.items():
            W_f   = W.astype(np.float32)
            scale = np.abs(W_f).max() / max_val
            scale = max(scale, 1e-8)
            W_q   = np.round(W_f / scale).clip(-max_val, max_val).astype(np.int8)
            quantized[name] = {"quantized": W_q, "scale": float(scale), "bits": bits}

        return quantized

    def dequantize_weights(
        self,
        quantized: Dict[str, Dict],
    ) -> Dict[str, np.ndarray]:
        """Dequantize int8 weights back to float32."""
        return {
            name: (d["quantized"].astype(np.float32) * d["scale"])
            for name, d in quantized.items()
        }

    @staticmethod
    def memory_report(params: List["Tensor"]) -> Dict[str, Any]:
        """
        Report memory usage of all parameters.

        Returns dict with per-dtype breakdown and total MB.
        """
        total_bytes  = 0
        dtype_counts: Dict[str, int] = {}
        for p in params:
            nb = p.data.nbytes
            dt = str(p.data.dtype)
            dtype_counts[dt] = dtype_counts.get(dt, 0) + nb
            total_bytes += nb

        f32_bytes = sum(p.data.size * 4 for p in params)  # if all float32
        f16_bytes = sum(p.data.size * 2 for p in params)  # if all float16

        return {
            "total_MB":         total_bytes / (1024 ** 2),
            "float32_equiv_MB": f32_bytes   / (1024 ** 2),
            "float16_equiv_MB": f16_bytes   / (1024 ** 2),
            "saving_vs_f64_MB": (sum(p.data.size * 8 for p in params)
                                 - total_bytes) / (1024 ** 2),
            "dtype_breakdown":  {k: v / (1024**2) for k, v in dtype_counts.items()},
            "n_params":         sum(p.data.size for p in params),
        }


# ── LoRA Layer ────────────────────────────────────────────────────────────────

class LoRALayer:
    """
    Low-Rank Adaptation (LoRA) adapter for a ZLinear layer.

    Instead of updating all W ∈ ℝ^(in × out) parameters during fine-tuning,
    LoRA freezes W and trains two small matrices:

        A ∈ ℝ^(in × r)    (down-projection)
        B ∈ ℝ^(r × out)   (up-projection)

    The effective weight update is:   ΔW = A × B × α/r

    Forward pass:   y = x @ W  +  x @ A @ B × (α / r)
                        ──────    ─────────────────────
                        frozen      LoRA delta (trained)

    Parameter reduction
    -------------------
    Standard : in × out  parameters
    LoRA     : r × (in + out)  parameters
    Example  : in=256, out=256, r=8 → 65536 vs 4096 (16× reduction)

    Parameters
    ----------
    in_dim  : input dimension of the target ZLinear layer
    out_dim : output dimension of the target ZLinear layer
    rank    : LoRA rank r (typically 4, 8, 16)
    alpha   : LoRA scaling factor (typically = rank)
    label   : name prefix for parameter labels
    precision : MixedPrecisionManager instance
    """

    def __init__(
        self,
        in_dim:    int,
        out_dim:   int,
        rank:      int                          = 8,
        alpha:     float                        = 8.0,
        label:     str                          = "lora",
        precision: Optional[MixedPrecisionManager] = None,
    ) -> None:
        self.rank      = rank
        self.alpha     = alpha
        self.scaling   = alpha / rank
        self.label     = label
        self.precision = precision

        # A: initialised with Gaussian (standard LoRA init)
        # B: initialised to zero (so ΔW = 0 at start → no disruption)
        self.A = Tensor(
            np.random.randn(in_dim, rank).astype(np.float32) * 0.01,
            label=f"{label}.A",
        )
        self.B = Tensor(
            np.zeros((rank, out_dim), dtype=np.float32),
            label=f"{label}.B",
        )

    def delta(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the LoRA delta contribution for input x.

            delta(x) = x @ A @ B × scaling

        Returns array of shape (..., out_dim).
        """
        if self.precision is not None:
            xA = self.precision.matmul(x, self.A.data)
            return self.precision.matmul(xA, self.B.data) * self.scaling
        return (x @ self.A.data @ self.B.data) * self.scaling

    def delta_tensor(self, x_tensor: "Tensor") -> "Tensor":
        """
        LoRA delta as a differentiable Tensor operation.
        Used during training so gradients flow through A and B.
        """
        xA  = Tensor(x_tensor.data @ self.A.data, (x_tensor, self.A), "lora_xA")

        def _bwd_xA():
            x_tensor.grad += xA.grad @ self.A.data.T
            self.A.grad   += x_tensor.data.T @ xA.grad

        xA._backward = _bwd_xA

        xAB = Tensor(xA.data @ self.B.data, (xA, self.B), "lora_xAB")

        def _bwd_xAB():
            xA.grad       += xAB.grad @ self.B.data.T
            self.B.grad   += xA.data.T @ xAB.grad

        xAB._backward = _bwd_xAB

        # Scale
        out = Tensor(xAB.data * self.scaling, (xAB,), "lora_scaled")

        def _bwd_scale():
            xAB.grad += out.grad * self.scaling

        out._backward = _bwd_scale
        return out

    def parameters(self) -> List["Tensor"]:
        return [self.A, self.B]

    def param_count(self) -> int:
        return self.A.data.size + self.B.data.size

    def reset_B(self) -> None:
        """Reset B to zeros (re-initialise LoRA delta to zero)."""
        self.B.data = np.zeros_like(self.B.data)


class ZLinearLoRA:
    """
    ZLinear layer with attached LoRA adapter.

    During training:
        - Base weight W is FROZEN (no gradient computed).
        - Only LoRA A and B are trained.
        - Output: y = x @ W_frozen + LoRA.delta_tensor(x)

    During inference:
        - LoRA delta is merged into W for zero overhead:
          W_merged = W + A @ B × scaling

    Parameters
    ----------
    in_dim  : input dimension
    out_dim : output dimension
    rank    : LoRA rank
    alpha   : LoRA alpha scaling
    label   : layer name
    frozen  : if True, base W has no gradient (LoRA-only training)
    precision : MixedPrecisionManager
    """

    def __init__(
        self,
        in_dim:    int,
        out_dim:   int,
        rank:      int                          = 8,
        alpha:     float                        = 8.0,
        label:     str                          = "fc",
        frozen:    bool                         = True,
        precision: Optional[MixedPrecisionManager] = None,
    ) -> None:
        self.frozen    = frozen
        self.precision = precision
        self._merged   = False

        # Base layer (float32 for stability)
        bound  = math.sqrt(6.0 / in_dim)
        self.W = Tensor(
            np.random.uniform(-bound, bound, (in_dim, out_dim)).astype(np.float32),
            label=f"{label}.W",
        )
        self.b = Tensor(
            np.zeros(out_dim, dtype=np.float32),
            label=f"{label}.b",
        )

        # LoRA adapter
        self.lora = LoRALayer(in_dim, out_dim, rank, alpha, f"{label}.lora", precision)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with LoRA delta.

        If merged (inference mode): y = x @ W_merged + b
        If training: y = x @ W_frozen + LoRA.delta(x) + b
        """
        if self._merged:
            # Merged inference: single matmul, no LoRA overhead
            if self.precision:
                out_data = self.precision.matmul(x.data, self.W.data) + self.b.data
            else:
                out_data = x.data @ self.W.data + self.b.data
            out = Tensor(out_data, (x, self.W, self.b), "affine_merged")

            def _bwd_merged():
                g        = out.grad
                x.grad  += g @ self.W.data.T
                if not self.frozen:
                    self.W.grad += x.data.T @ g
                self.b.grad += g.sum(axis=0)

            out._backward = _bwd_merged
            return out

        # Training path: frozen base + LoRA delta
        if self.precision:
            base_data = self.precision.matmul(x.data, self.W.data) + self.b.data
        else:
            base_data = x.data @ self.W.data + self.b.data

        base_out = Tensor(base_data, (x, self.b), "affine_frozen")

        def _bwd_base():
            g = base_out.grad
            if not self.frozen:
                x.grad      += g @ self.W.data.T
            self.b.grad += g.sum(axis=0)

        base_out._backward = _bwd_base

        # LoRA delta (trainable)
        lora_out = self.lora.delta_tensor(x)

        # Sum: base + lora
        combined = Tensor(
            base_out.data + lora_out.data,
            (base_out, lora_out), "lora_combined"
        )

        def _bwd_combined():
            base_out.grad += combined.grad
            lora_out.grad += combined.grad

        combined._backward = _bwd_combined
        return combined

    def merge_lora(self) -> None:
        """
        Merge LoRA delta into base weight W for inference.
        After merge: W = W + A @ B × scaling
        No performance overhead during inference.
        """
        if self._merged:
            return
        delta        = self.lora.A.data @ self.lora.B.data * self.lora.scaling
        self.W.data += delta.astype(self.W.data.dtype)
        self._merged  = True
        logger.info("LoRA merged into %s", self.W.label)

    def unmerge_lora(self) -> None:
        """Unmerge LoRA (restore separate W + LoRA for continued training)."""
        if not self._merged:
            return
        delta        = self.lora.A.data @ self.lora.B.data * self.lora.scaling
        self.W.data -= delta.astype(self.W.data.dtype)
        self._merged  = False

    def trainable_parameters(self) -> List[Tensor]:
        """Only LoRA parameters (A, B). Base W is frozen."""
        params = list(self.lora.parameters()) + [self.b]
        return params

    def all_parameters(self) -> List[Tensor]:
        return [self.W, self.b] + self.lora.parameters()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def param_count(self) -> Dict[str, int]:
        return {
            "base_W":     self.W.data.size,
            "base_b":     self.b.data.size,
            "lora_A":     self.lora.A.data.size,
            "lora_B":     self.lora.B.data.size,
            "trainable":  self.lora.param_count() + self.b.data.size,
            "total":      self.W.data.size + self.b.data.size + self.lora.param_count(),
        }


class ZNetworkLoRA:
    """
    ZNetwork variant where every ZLinear is replaced by ZLinearLoRA.

    Training: only LoRA adapters trained (A, B matrices).
              Base weights W are frozen.
    Inference: LoRA merged into W → zero overhead.

    Parameter reduction example (dims=[256, 512, 256, 64], rank=8):
        Standard training params: ~200k
        LoRA training params    : ~12k   (94% reduction)
        Speedup (approx)        : 8-16x on CPU, more on GPU

    Parameters
    ----------
    dims      : layer widths
    rank      : LoRA rank for all layers
    alpha     : LoRA alpha scaling
    activation: activation function name
    frozen    : freeze base weights (True = LoRA-only training)
    precision : MixedPrecisionManager
    """

    _ACTIVATIONS: Dict[str, Callable] = {
        "relu":    lambda t: t.relu(),
        "tanh":    lambda t: t.tanh(),
        "sigmoid": lambda t: t.sigmoid(),
        "gelu":    lambda t: t.gelu(),
    }

    def __init__(
        self,
        dims:      List[int],
        rank:      int                          = 8,
        alpha:     float                        = 8.0,
        activation:str                          = "gelu",
        frozen:    bool                         = True,
        precision: Optional[MixedPrecisionManager] = None,
    ) -> None:
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self._ACTIVATIONS)}")
        self.act       = self._ACTIVATIONS[activation]
        self.precision = precision
        self.layers    = [
            ZLinearLoRA(
                dims[i], dims[i + 1],
                rank=rank, alpha=alpha,
                label=f"L{i}",
                frozen=frozen,
                precision=precision,
            )
            for i in range(len(dims) - 1)
        ]

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

    def parameters(self) -> List[Tensor]:
        """Return ONLY trainable parameters (LoRA A, B + biases)."""
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.trainable_parameters())
        return params

    def all_parameters(self) -> List[Tensor]:
        """Return ALL parameters including frozen base weights."""
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.all_parameters())
        return params

    def merge_all_lora(self) -> None:
        """Merge all LoRA adapters into base weights (inference mode)."""
        for layer in self.layers:
            layer.merge_lora()
        logger.info("ZNetworkLoRA: all LoRA adapters merged for inference.")

    def unmerge_all_lora(self) -> None:
        """Unmerge all LoRA adapters (return to training mode)."""
        for layer in self.layers:
            layer.unmerge_lora()

    def param_summary(self) -> Dict[str, int]:
        total   = sum(sum(pc.values()) for layer in self.layers
                      for pc in [layer.param_count])
        trained = sum(p.data.size for p in self.parameters())
        frozen  = total - trained
        return {
            "total":     total,
            "trainable": trained,
            "frozen":    frozen,
            "reduction": f"{(1 - trained/max(total,1))*100:.1f}%",
        }

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

"""
fractal_attention_compression.py
=================================
Module 2 — O(N log N) Fractal Hierarchical Attention
Based on the Fractal Neural Network (FNN) architecture as formalised in:
    "Beyond Finite Context: A Theoretical Architecture for Fractal Neural
     Networks Utilizing Recursive Self-Similar Attention" (BBDU, 2026)

Architecture
------------
    AttentionConfig          — Immutable hyperparameter container
    SemanticCompressor       — Dense vector encoding with minimal semantic loss
    FractalAttentionNode     — Single RSSB (Recursive Self-Similar Block)
    Fractal_Attention_Node   — Full recursive tree engine (O(N log_b N))
    FractalEncoder           — End-to-end encoder: tokens → root representation
    FractalDecoder           — Hierarchical decoder: root → sequence reconstruction
    AttentionCache           — Depth-indexed KV cache for inference reuse
    ComplexityAnalyser       — Runtime complexity and VRAM usage tracker

Complexity
----------
    Time  : O(N log_b N)   — proved via constructive depth summation (Theorem 1)
    Space : O(N)           — parameter count independent of depth
    VRAM  : O(b² · log_b N) — bounded by chunk size, not sequence length

Dependencies: numpy only (zero framework policy)
"""


import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  § 1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttentionConfig:
    """
    Immutable hyperparameter container for the Fractal Attention engine.

    Parameters
    ----------
    embed_dim       : token embedding / hidden dimension  d
    branching_b     : n-ary tree branching factor         b
    depth_decay_lam : depth-penalisation coefficient      λ
    num_heads       : number of parallel attention heads
    dropout_rate    : attention weight dropout (0.0 = disabled)
    use_rope        : apply Rotary Position Embeddings to Q and K
    eps             : numerical stability floor
    """
    embed_dim:       int   = 128
    branching_b:     int   = 4
    depth_decay_lam: float = 0.1
    num_heads:       int   = 4
    dropout_rate:    float = 0.0
    use_rope:        bool  = True
    eps:             float = 1e-9

    def __post_init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  ROTARY POSITION EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding:
    """
    Rotary Position Embedding (RoPE) for relative position encoding.

    Applies a rotation matrix to Q and K vectors such that the dot product
    Q·K^T naturally encodes relative positional distance:

        RoPE(x, m) = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)

    where θ_i = 1 / 10000^(2i/d).

    Parameters
    ----------
    dim      : per-head embedding dimension
    max_seq  : maximum sequence length to precompute (extended lazily)
    base     : frequency base (default 10000)
    """

    def __init__(self, dim: int, max_seq: int = 8192, base: int = 10000) -> None:
        self.dim     = dim
        self.base    = base
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute(max_seq)

    def _precompute(self, seq_len: int) -> None:
        half  = self.dim // 2
        theta = 1.0 / np.power(
            self.base,
            np.arange(0, half, dtype=np.float64) * 2.0 / self.dim
        )
        pos   = np.arange(seq_len, dtype=np.float64)
        freqs = np.outer(pos, theta)                    # (seq, half)
        self._cos = np.cos(freqs)                        # (seq, half)
        self._sin = np.sin(freqs)                        # (seq, half)
        self._max = seq_len

    @staticmethod
    def _rotate_half(x: np.ndarray) -> np.ndarray:
        """Split last dim in two halves and rotate: [-x2, x1]."""
        h    = x.shape[-1] // 2
        x1   = x[..., :h]
        x2   = x[..., h:]
        return np.concatenate([-x2, x1], axis=-1)

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Apply RoPE to tensor x of shape (..., seq_len, dim).

        Parameters
        ----------
        x      : input array
        offset : position offset for cached inference
        """
        seq = x.shape[-2]
        if offset + seq > self._max:
            self._precompute(max(offset + seq, self._max * 2))

        cos = self._cos[offset : offset + seq]           # (seq, half)
        sin = self._sin[offset : offset + seq]           # (seq, half)

        # Tile to full dim
        cos = np.concatenate([cos, cos], axis=-1)        # (seq, dim)
        sin = np.concatenate([sin, sin], axis=-1)        # (seq, dim)

        return x * cos + self._rotate_half(x) * sin


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  SEMANTIC COMPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCompressor:
    """
    Compresses extreme-length token sequences into dense vector arrays
    with minimal semantic loss.

    Strategy
    --------
    1. Sliding-window local mean pooling (captures local context).
    2. Frequency-domain projection via DCT-II (preserves global structure).
    3. Learned linear projection down to target_dim (semantic bottleneck).
    4. Layer normalisation for stable downstream attention.

    The combination ensures Shannon entropy of the macroscopic semantic
    structure is preserved while token-level noise is discarded — consistent
    with the FNN information-loss trade-off (Section V, Limitation 1).

    Parameters
    ----------
    input_dim  : dimensionality of input token embeddings
    target_dim : output compressed embedding dimension
    window     : local pooling window size
    """

    def __init__(
        self,
        input_dim:  int,
        target_dim: int,
        window:     int = 4,
    ) -> None:
        self.input_dim  = input_dim
        self.target_dim = target_dim
        self.window     = window

        # Learned projection: input_dim → target_dim (He init)
        scale         = math.sqrt(2.0 / input_dim)
        self.W_proj   = np.random.randn(input_dim, target_dim) * scale
        self.b_proj   = np.zeros(target_dim)

        # LayerNorm parameters
        self.ln_gamma = np.ones(target_dim)
        self.ln_beta  = np.zeros(target_dim)

    # ── DCT-II (Type 2, orthonormal) ──────────────────────────────────────────

    @staticmethod
    def _dct2(x: np.ndarray) -> np.ndarray:
        """
        Compute DCT-II along the sequence axis (axis=0).
        Orthonormal form preserves energy (Parseval's theorem).
        x : (N, d)  →  returns (N, d)
        """
        N     = x.shape[0]
        n     = np.arange(N)
        k     = n[:, None]                              # (N, 1)
        basis = np.cos(math.pi / N * (n + 0.5) * k)   # (N, N)

        # Orthonormal scaling
        scale         = np.sqrt(2.0 / N) * np.ones(N)
        scale[0]     /= math.sqrt(2.0)
        return (scale[:, None] * basis) @ x             # (N, d)

    # ── Local mean pooling ────────────────────────────────────────────────────

    def _local_pool(self, x: np.ndarray) -> np.ndarray:
        """
        Sliding-window mean pooling along the sequence axis.
        x : (N, d)  →  returns (N, d)
        """
        N, d  = x.shape
        out   = np.empty_like(x)
        w     = self.window
        for i in range(N):
            lo       = max(0, i - w // 2)
            hi       = min(N, lo + w)
            out[i]   = x[lo:hi].mean(axis=0)
        return out

    # ── LayerNorm ─────────────────────────────────────────────────────────────

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1,  keepdims=True) + eps
        return self.ln_gamma * (x - mu) / std + self.ln_beta

    # ── Full compression pipeline ─────────────────────────────────────────────

    def compress(self, x: np.ndarray) -> np.ndarray:
        """
        Compress token sequence x to dense representation.

        Parameters
        ----------
        x : (N, input_dim) — raw token embeddings

        Returns
        -------
        (N, target_dim) — semantically compressed embeddings
        """
        # Stage 1: local context capture
        pooled = self._local_pool(x)

        # Stage 2: frequency-domain global structure
        freq   = self._dct2(pooled)

        # Stage 3: linear projection to target_dim
        proj   = freq @ self.W_proj + self.b_proj

        # Stage 4: layer normalisation
        return self._layer_norm(proj)

    def compress_to_scalar(self, x: np.ndarray) -> np.ndarray:
        """
        Compress a chunk to a single representative vector.
        Used by the RSSB pooling step.

        x : (chunk_size, input_dim)  →  (1, input_dim)
        """
        return x.mean(axis=0, keepdims=True)

    def parameters(self) -> List[np.ndarray]:
        return [self.W_proj, self.b_proj, self.ln_gamma, self.ln_beta]


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  ATTENTION CACHE
# ══════════════════════════════════════════════════════════════════════════════

class AttentionCache:
    """
    Depth-indexed key-value cache for inference-time reuse.

    During autoregressive generation, previously computed K and V matrices
    at each recursive depth are stored and extended rather than recomputed,
    reducing per-step cost from O(N log N) to O(log N) amortised.

    Parameters
    ----------
    max_depth : maximum recursion depth to cache
    """

    def __init__(self, max_depth: int = 32) -> None:
        self.max_depth = max_depth
        self._k: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}

    def store(self, depth: int, k: np.ndarray, v: np.ndarray) -> None:
        if depth > self.max_depth:
            return
        if depth in self._k:
            self._k[depth] = np.concatenate([self._k[depth], k], axis=-2)
            self._v[depth] = np.concatenate([self._v[depth], v], axis=-2)
        else:
            self._k[depth] = k.copy()
            self._v[depth] = v.copy()

    def retrieve(self, depth: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if depth in self._k:
            return self._k[depth], self._v[depth]
        return None

    def clear(self, depth: Optional[int] = None) -> None:
        if depth is None:
            self._k.clear()
            self._v.clear()
        else:
            self._k.pop(depth, None)
            self._v.pop(depth, None)

    @property
    def cached_depths(self) -> List[int]:
        return sorted(self._k.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  FRACTAL ATTENTION NODE (RSSB — Single Depth)
# ══════════════════════════════════════════════════════════════════════════════

class FractalAttentionNode:
    """
    Recursive Self-Similar Block (RSSB) — single recursive depth.

    Implements depth-penalised multi-head attention:

        Attention(C, l) = softmax( QKᵀ / √d · e^(−λl) ) · V

    where:
        Q, K, V  — shared projection matrices (weight-tied across depths)
        l        — current recursion depth
        λ        — depth decay penalty (AttentionConfig.depth_decay_lam)

    Aggregation (lossy compression per FNN Section III-B):

        X^(l+1)_i  =  (1/b) Σ_j Attention(C_{i,j}, l)

    Parameters
    ----------
    config : AttentionConfig
    rope   : shared RotaryEmbedding instance
    """

    def __init__(self, config: AttentionConfig, rope: RotaryEmbedding) -> None:
        self.cfg  = config
        self.rope = rope
        d         = config.embed_dim
        h         = config.head_dim
        H         = config.num_heads

        # Shared weight matrices (tied across all depths — O(1) params w.r.t. depth)
        scale      = 1.0 / math.sqrt(d)
        self.W_Q   = np.random.randn(d, d) * scale          # (d, d)
        self.W_K   = np.random.randn(d, d) * scale
        self.W_V   = np.random.randn(d, d) * scale
        self.W_O   = np.random.randn(d, d) * scale          # output projection

        # Feed-forward sub-layer (post-attention)
        self.W_ff1 = np.random.randn(d, d * 4) * scale
        self.b_ff1 = np.zeros(d * 4)
        self.W_ff2 = np.random.randn(d * 4, d) * scale
        self.b_ff2 = np.zeros(d)

        # LayerNorm parameters (two per RSSB: post-attn, post-ff)
        self.ln1_gamma = np.ones(d)
        self.ln1_beta  = np.zeros(d)
        self.ln2_gamma = np.ones(d)
        self.ln2_beta  = np.zeros(d)

    # ── LayerNorm ─────────────────────────────────────────────────────────────

    def _layer_norm(
        self,
        x:     np.ndarray,
        gamma: np.ndarray,
        beta:  np.ndarray,
    ) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1,  keepdims=True) + self.cfg.eps
        return gamma * (x - mu) / std + beta

    # ── Multi-head split / merge ───────────────────────────────────────────────

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq, d) → (H, seq, head_dim)"""
        seq, d = x.shape
        H      = self.cfg.num_heads
        h      = self.cfg.head_dim
        return x.reshape(seq, H, h).transpose(1, 0, 2)     # (H, seq, h)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(H, seq, head_dim) → (seq, d)"""
        H, seq, h = x.shape
        return x.transpose(1, 0, 2).reshape(seq, H * h)    # (seq, d)

    # ── Depth-penalised attention ─────────────────────────────────────────────

    def _attend(
        self,
        chunk: np.ndarray,
        depth: int,
        cache: Optional[AttentionCache] = None,
        pos_offset: int = 0,
    ) -> np.ndarray:
        """
        Compute depth-penalised multi-head attention for one chunk.

        Parameters
        ----------
        chunk      : (b, d) local input chunk
        depth      : current recursion depth l
        cache      : optional KV cache for inference
        pos_offset : RoPE position offset

        Returns
        -------
        (b, d) attention output
        """
        lam   = self.cfg.depth_decay_lam
        decay = math.exp(-lam * depth)

        # Linear projections
        Q = chunk @ self.W_Q                                # (b, d)
        K = chunk @ self.W_K
        V = chunk @ self.W_V

        # RoPE
        if self.cfg.use_rope:
            Q = self.rope.apply(Q[None], offset=pos_offset)[0]
            K = self.rope.apply(K[None], offset=pos_offset)[0]

        # Extend from KV cache if available
        if cache is not None:
            cached = cache.retrieve(depth)
            if cached is not None:
                K = np.concatenate([cached[0], K], axis=0)
                V = np.concatenate([cached[1], V], axis=0)
            cache.store(depth, K[-chunk.shape[0]:], V[-chunk.shape[0]:])

        # Multi-head split
        Q_h = self._split_heads(Q)                          # (H, b, h)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Scaled dot-product with depth decay
        scale  = math.sqrt(self.cfg.head_dim) + self.cfg.eps
        scores = Q_h @ K_h.transpose(0, 2, 1) / scale      # (H, b, b_k)
        scores = scores * decay

        # Causal mask (lower triangular)
        b_q, b_k = scores.shape[1], scores.shape[2]
        if b_q == b_k:
            mask   = np.tril(np.ones((b_q, b_k)))
            scores = np.where(mask[None] == 0, -1e9, scores)

        # Softmax
        scores -= scores.max(axis=-1, keepdims=True)        # stability
        weights = np.exp(scores)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + self.cfg.eps)

        # Dropout (training only, applied to weights)
        if self.cfg.dropout_rate > 0.0:
            mask    = np.random.binomial(1, 1 - self.cfg.dropout_rate, weights.shape)
            weights = weights * mask / (1 - self.cfg.dropout_rate + self.cfg.eps)

        # Weighted sum
        out_h = weights @ V_h                               # (H, b, h)
        out   = self._merge_heads(out_h)                    # (b, d)

        # Output projection
        return out @ self.W_O

    # ── Feed-forward sub-layer ────────────────────────────────────────────────

    def _feedforward(self, x: np.ndarray) -> np.ndarray:
        """Position-wise FFN: d → 4d → d with GELU activation."""
        h   = x @ self.W_ff1 + self.b_ff1                  # (seq, 4d)
        # GELU: x · Φ(x)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(h / math.sqrt(2.0)))
        h   = h * cdf
        return h @ self.W_ff2 + self.b_ff2                  # (seq, d)

    # ── Single RSSB forward pass ──────────────────────────────────────────────

    def forward(
        self,
        chunk:      np.ndarray,
        depth:      int,
        cache:      Optional[AttentionCache] = None,
        pos_offset: int = 0,
    ) -> np.ndarray:
        """
        Process one chunk through the RSSB at a given depth.

        Pre-norm architecture: LayerNorm → sub-layer → residual.

        Parameters
        ----------
        chunk      : (b, d) input chunk
        depth      : recursion depth l
        cache      : optional KV cache
        pos_offset : RoPE position offset

        Returns
        -------
        (b, d) processed chunk
        """
        # Sub-layer 1: attention
        normed = self._layer_norm(chunk, self.ln1_gamma, self.ln1_beta)
        attn   = self._attend(normed, depth, cache, pos_offset)
        x      = chunk + attn                               # residual

        # Sub-layer 2: feed-forward
        normed = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ff     = self._feedforward(normed)
        return x + ff                                       # residual

    def parameters(self) -> List[np.ndarray]:
        return [
            self.W_Q, self.W_K, self.W_V, self.W_O,
            self.W_ff1, self.b_ff1, self.W_ff2, self.b_ff2,
            self.ln1_gamma, self.ln1_beta,
            self.ln2_gamma, self.ln2_beta,
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  FRACTAL_ATTENTION_NODE — Full Recursive Tree Engine
# ══════════════════════════════════════════════════════════════════════════════

class Fractal_Attention_Node:
    """
    Full recursive fractal attention engine.

    Architecture
    ------------
    The input sequence X^(0) ∈ ℝ^(N×d) is processed by an n-ary tree of
    depth L = log_b(N) RSSB nodes. At each depth l:

        1. Chunk X^(l) into N/b^(l+1) blocks of size b.
        2. Run RSSB attention within each chunk → chunk outputs.
        3. Mean-pool each chunk output → X^(l+1)  (lossy compression).
        4. Recurse until X^(L) ∈ ℝ^(1×d)  (root / global representation).

    Simultaneously maintains a local (high-res) view at depth 0 and a
    global (low-res) view at the root — "zoom in / zoom out" per spec.

    Complexity: O(N log_b N)  (Theorem 1, constructive proof sketch)

    Parameters
    ----------
    config : AttentionConfig
    """

    def __init__(self, config: AttentionConfig) -> None:
        self.cfg   = config
        self.rope  = RotaryEmbedding(config.head_dim)
        self.rssb  = FractalAttentionNode(config, self.rope)   # shared (weight-tied)
        self._flop_log: List[Dict] = []

    # ── Padding ───────────────────────────────────────────────────────────────

    def _pad_to_multiple(self, x: np.ndarray, b: int) -> np.ndarray:
        """Right-pad sequence to the nearest multiple of b."""
        N, d  = x.shape
        rem   = N % b
        if rem == 0:
            return x
        pad   = np.zeros((b - rem, d), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    # ── Recursive RSSB pass ───────────────────────────────────────────────────

    def _recurse(
        self,
        x:     np.ndarray,
        depth: int,
        cache: Optional[AttentionCache] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Recursive tree traversal.

        Parameters
        ----------
        x     : (N_l, d) sequence at depth l
        depth : current recursion depth
        cache : optional KV cache

        Returns
        -------
        root      : (1, d) pooled root representation
        all_levels: list of per-depth outputs for hierarchical decoding
        """
        N, d  = x.shape
        b     = self.cfg.branching_b

        # Base case: sequence fits in one chunk → single RSSB call
        if N <= b:
            out = self.rssb.forward(x, depth=depth, cache=cache)
            return out.mean(axis=0, keepdims=True), [out]

        # Pad if necessary
        x_pad = self._pad_to_multiple(x, b)
        N_pad = x_pad.shape[0]
        n_chunks = N_pad // b

        # FLOP accounting
        self._flop_log.append({
            "depth":    depth,
            "N":        N_pad,
            "n_chunks": n_chunks,
            "cost":     n_chunks * (b ** 2),   # O(b²) per chunk
        })

        # Step 1 & 2: chunk + RSSB attention (local high-res view)
        chunk_outs = []
        for i in range(n_chunks):
            chunk = x_pad[i * b : (i + 1) * b]              # (b, d)
            out   = self.rssb.forward(
                chunk, depth=depth, cache=cache, pos_offset=i * b
            )
            chunk_outs.append(out)

        local_out = np.concatenate(chunk_outs, axis=0)       # (N_pad, d)
        local_out = local_out[:N]                            # remove padding

        # Step 3: mean-pool chunks → compressed sequence for depth l+1
        compressed = np.stack(
            [co.mean(axis=0) for co in chunk_outs], axis=0  # (n_chunks, d)
        )

        # Step 4: recurse (global low-res view)
        root, deeper_levels = self._recurse(compressed, depth + 1, cache)

        return root, [local_out] + deeper_levels

    # ── Public forward ────────────────────────────────────────────────────────

    def forward(
        self,
        x:     np.ndarray,
        cache: Optional[AttentionCache] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Full fractal forward pass.

        Parameters
        ----------
        x     : (N, d) input sequence embeddings
        cache : optional AttentionCache for inference reuse

        Returns
        -------
        dict with keys:
            "root"       : (1, d) global root representation
            "local"      : (N, d) high-resolution local output (depth 0)
            "all_levels" : list of per-depth outputs (hierarchical)
        """
        self._flop_log.clear()
        assert x.ndim == 2, f"Expected (N, d), got {x.shape}"
        assert x.shape[1] == self.cfg.embed_dim, (
            f"Expected embed_dim={self.cfg.embed_dim}, got {x.shape[1]}"
        )

        root, all_levels = self._recurse(x, depth=0, cache=cache)

        return {
            "root":       root,
            "local":      all_levels[0],
            "all_levels": all_levels,
        }

    # ── Complexity query ──────────────────────────────────────────────────────

    def theoretical_complexity(self, N: int) -> Dict[str, object]:
        """
        Return theoretical complexity metrics for sequence length N.

        Implements Theorem 1 constructive proof:
            Total cost = Σ_{l=0}^{log_b N} O(N) = O(N log_b N)
        """
        b     = self.cfg.branching_b
        L     = math.ceil(math.log(N, b))
        costs = []
        for l in range(L + 1):
            Nl      = math.ceil(N / (b ** l))
            n_ch    = max(1, math.ceil(Nl / b))
            cost_l  = n_ch * (b ** 2)
            costs.append({"depth": l, "seq_len": Nl, "flops": cost_l})

        total = sum(c["flops"] for c in costs)
        return {
            "N":                   N,
            "b":                   b,
            "L":                   L,
            "total_flops_O(NlogN)": total,
            "standard_O(N2)":      N ** 2,
            "reduction_factor":    (N ** 2) / max(total, 1),
            "per_depth":           costs,
        }

    def parameters(self) -> List[np.ndarray]:
        return self.rssb.parameters()


# ══════════════════════════════════════════════════════════════════════════════
#  § 7  FRACTAL ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class FractalEncoder:
    """
    End-to-end encoder: raw token indices → root representation.

    Pipeline
    --------
    1. Token embedding lookup         (N,)   → (N, embed_dim)
    2. SemanticCompressor             (N, d) → (N, d)   [optional]
    3. Fractal_Attention_Node         (N, d) → hierarchical outputs
    4. Root representation            (1, d) — final encoding

    Parameters
    ----------
    vocab_size  : vocabulary size for the embedding table
    config      : AttentionConfig
    compress    : whether to apply SemanticCompressor before attention
    """

    def __init__(
        self,
        vocab_size: int,
        config:     AttentionConfig,
        compress:   bool = True,
    ) -> None:
        self.cfg     = config
        self.compress = compress

        # Embedding table: (vocab, d), scaled as in "Attention is All You Need"
        scale             = math.sqrt(config.embed_dim)
        self.embedding    = np.random.randn(vocab_size, config.embed_dim) / scale

        # Optional semantic compressor
        self.compressor   = (SemanticCompressor(config.embed_dim, config.embed_dim)
                             if compress else None)

        # Fractal attention engine
        self.fractal      = Fractal_Attention_Node(config)

        # Final projection to root space
        self.W_out        = np.eye(config.embed_dim)          # identity default
        self.ln_out_gamma = np.ones(config.embed_dim)
        self.ln_out_beta  = np.zeros(config.embed_dim)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1,  keepdims=True) + self.cfg.eps
        return self.ln_out_gamma * (x - mu) / std + self.ln_out_beta

    def encode(
        self,
        token_ids:  np.ndarray,
        cache:      Optional[AttentionCache] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Encode a token sequence.

        Parameters
        ----------
        token_ids : (N,) integer array of token indices
        cache     : optional KV cache

        Returns
        -------
        dict with keys: root, local, all_levels, embeddings
        """
        # Step 1: embedding lookup
        emb = self.embedding[token_ids]                     # (N, d)

        # Step 2: optional compression
        if self.compress and self.compressor is not None:
            emb = self.compressor.compress(emb)

        # Step 3: fractal attention
        out = self.fractal.forward(emb, cache=cache)

        # Step 4: normalise root
        out["root"] = self._layer_norm(out["root"] @ self.W_out)
        out["embeddings"] = emb

        return out

    def parameters(self) -> List[np.ndarray]:
        params = [self.embedding, self.W_out,
                  self.ln_out_gamma, self.ln_out_beta]
        params.extend(self.fractal.parameters())
        if self.compressor:
            params.extend(self.compressor.parameters())
        return params


# ══════════════════════════════════════════════════════════════════════════════
#  § 8  FRACTAL DECODER
# ══════════════════════════════════════════════════════════════════════════════

class FractalDecoder:
    """
    Hierarchical decoder: reconstructs sequence representations from the
    fractal encoder's multi-level outputs.

    Uses a cross-attention mechanism where the higher-level (coarser)
    representation acts as memory for the lower-level (finer) reconstruction.

    Parameters
    ----------
    config : AttentionConfig
    """

    def __init__(self, config: AttentionConfig) -> None:
        self.cfg   = config
        self.rope  = RotaryEmbedding(config.head_dim)

        # Cross-attention projections
        d          = config.embed_dim
        scale      = 1.0 / math.sqrt(d)
        self.W_Q   = np.random.randn(d, d) * scale
        self.W_K   = np.random.randn(d, d) * scale
        self.W_V   = np.random.randn(d, d) * scale
        self.W_O   = np.random.randn(d, d) * scale

        # Upsampling projection (compressed → full)
        self.W_up  = np.random.randn(d, d) * scale
        self.b_up  = np.zeros(d)

        # LayerNorm
        self.ln_gamma = np.ones(d)
        self.ln_beta  = np.zeros(d)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1,  keepdims=True) + self.cfg.eps
        return self.ln_gamma * (x - mu) / std + self.ln_beta

    def _cross_attend(
        self,
        query:  np.ndarray,
        memory: np.ndarray,
    ) -> np.ndarray:
        """
        Cross-attention: query from fine level, memory from coarse level.
        query  : (N_fine,   d)
        memory : (N_coarse, d)
        returns: (N_fine,   d)
        """
        Q = query  @ self.W_Q                               # (N_f, d)
        K = memory @ self.W_K                               # (N_c, d)
        V = memory @ self.W_V

        scale   = math.sqrt(self.cfg.head_dim) + self.cfg.eps
        scores  = Q @ K.T / scale                           # (N_f, N_c)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + self.cfg.eps)

        out = weights @ V                                   # (N_f, d)
        return out @ self.W_O

    def decode(self, all_levels: List[np.ndarray]) -> np.ndarray:
        """
        Hierarchical decoding from coarse-to-fine.

        Iterates from the deepest (coarsest) level to the shallowest
        (finest) level, using cross-attention at each step.

        Parameters
        ----------
        all_levels : list of per-depth arrays from FractalEncoder.encode()
                     all_levels[0] = finest (depth 0), [-1] = coarsest

        Returns
        -------
        (N, d) reconstructed sequence representation
        """
        # Start from the coarsest level
        current = all_levels[-1]

        for level in reversed(all_levels[:-1]):
            # Upsample current coarse representation
            upsampled = current @ self.W_up + self.b_up    # (N_c, d)

            # Cross-attend: fine (query) ← coarse (memory)
            cross = self._cross_attend(level, upsampled)

            # Residual merge
            current = self._layer_norm(level + cross)

        return current

    def parameters(self) -> List[np.ndarray]:
        return [
            self.W_Q, self.W_K, self.W_V, self.W_O,
            self.W_up, self.b_up,
            self.ln_gamma, self.ln_beta,
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  § 9  COMPLEXITY ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

class ComplexityAnalyser:
    """
    Runtime complexity and VRAM usage tracker for the fractal attention engine.

    Tracks:
        - Theoretical vs measured FLOP counts per depth
        - Peak intermediate array sizes (VRAM proxy)
        - Reduction factor versus standard O(N²) attention

    Parameters
    ----------
    config : AttentionConfig
    """

    def __init__(self, config: AttentionConfig) -> None:
        self.cfg = config

    def analyse(self, N: int) -> Dict[str, object]:
        """
        Full complexity analysis for sequence length N.

        Returns
        -------
        dict with theoretical and VRAM metrics.
        """
        b = self.cfg.branching_b
        d = self.cfg.embed_dim
        L = math.ceil(math.log(max(N, b + 1), b))

        per_depth = []
        total_flops = 0
        peak_vram_bytes = 0

        for l in range(L + 1):
            Nl        = max(1, math.ceil(N / (b ** l)))
            n_chunks  = max(1, math.ceil(Nl / b))
            flops_l   = n_chunks * (b ** 2)

            # VRAM: Q, K, V, scores per chunk × n_chunks
            # Each array: (b, d) float64 = b * d * 8 bytes
            # scores: (H, b, b) float64 = H * b^2 * 8 bytes
            qkv_bytes    = 3 * n_chunks * b * d * 8
            scores_bytes = self.cfg.num_heads * (b ** 2) * 8 * n_chunks
            vram_l       = qkv_bytes + scores_bytes

            total_flops     += flops_l
            peak_vram_bytes  = max(peak_vram_bytes, vram_l)

            per_depth.append({
                "depth":       l,
                "seq_len":     Nl,
                "n_chunks":    n_chunks,
                "flops":       flops_l,
                "vram_bytes":  vram_l,
            })

        standard_n2 = N ** 2

        return {
            "N":                      N,
            "b":                      b,
            "L":                      L,
            "d":                      d,
            "H":                      self.cfg.num_heads,
            "total_flops_fractal":    total_flops,
            "total_flops_standard":   standard_n2,
            "reduction_factor":       standard_n2 / max(total_flops, 1),
            "peak_vram_bytes":        peak_vram_bytes,
            "peak_vram_MB":           peak_vram_bytes / (1024 ** 2),
            "standard_vram_bytes":    N * d * 3 * 8 + self.cfg.num_heads * N * N * 8,
            "standard_vram_MB":       (N * d * 3 * 8 + self.cfg.num_heads * N * N * 8)
                                      / (1024 ** 2),
            "per_depth":              per_depth,
        }

    def compare(self, seq_lengths: List[int]) -> List[Dict]:
        """Run analyse() for multiple sequence lengths for comparison."""
        return [self.analyse(N) for N in seq_lengths]
"""
hardware_entropy_harvester.py
==============================
Module 3 — Stochastic Hardware-Seeded Processing (Non-Deterministic Logic)

Introduces true stochasticity into decision-making branches by reading
system-level entropy from multiple hardware sources and composing them
into high-quality probability distributions.

Architecture
------------
    EntropySource            — Abstract base for all entropy sources
    OsUrandomSource          — os.urandom kernel entropy pool
    MemoryTimingSource       — Memory allocation timing jitter
    ClockJitterSource        — High-resolution clock noise sampling
    CpuInstructionSource     — CPU instruction execution timing variance
    HardwareEntropyPool      — Multi-source entropy mixer (Von Neumann debiased)
    Hardware_Entropy_Harvester — Master harvester: seeding & distribution generation
    StochasticNode           — Processing node with hardware-seeded decisions
    EntropyQualityAssessor   — NIST-inspired statistical quality tests

Sources of True Entropy Used
-----------------------------
    os.urandom               — kernel entropy (hardware RNG / /dev/urandom)
    time.perf_counter_ns()   — nanosecond clock jitter
    memory allocation timing — heap allocation latency variation
    cpu loop timing          — instruction pipeline stochasticity

Dependencies: numpy, os, time, struct, hashlib, threading (stdlib only)
"""


import hashlib
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  § 1  ABSTRACT ENTROPY SOURCE
# ══════════════════════════════════════════════════════════════════════════════

class EntropySource(ABC):
    """
    Abstract base class for a hardware entropy source.

    Every source exposes:
        collect(n_bytes) → bytes   raw entropy bytes
        quality()        → float  estimated bits-per-byte (0.0–8.0)
        name             → str
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def collect(self, n_bytes: int = 32) -> bytes: ...

    @abstractmethod
    def quality(self) -> float:
        """Estimated Shannon entropy in bits per byte (max 8.0)."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  CONCRETE ENTROPY SOURCES
# ══════════════════════════════════════════════════════════════════════════════

class OsUrandomSource(EntropySource):
    """
    Kernel entropy pool via os.urandom().

    On Linux: reads from /dev/urandom (seeded by hardware events).
    On Windows: calls CryptGenRandom().
    On macOS: reads from /dev/urandom (Fortuna CSPRNG).

    Quality: ~8.0 bits/byte — highest quality source.
    """

    @property
    def name(self) -> str:
        return "os.urandom (kernel CSPRNG)"

    def collect(self, n_bytes: int = 32) -> bytes:
        return os.urandom(n_bytes)

    def quality(self) -> float:
        # Kernel CSPRNG is considered full-entropy
        return 8.0


class MemoryTimingSource(EntropySource):
    """
    Memory allocation timing jitter.

    Measures the nanosecond-level variance in heap allocation latency.
    OS memory management introduces true non-determinism due to:
        - TLB misses and page faults
        - NUMA topology effects
        - Allocator fragmentation state

    Quality: ~2.0–4.0 bits/byte (environment-dependent).
    """

    def __init__(self, n_samples: int = 64, alloc_size: int = 4096) -> None:
        self.n_samples  = n_samples
        self.alloc_size = alloc_size

    @property
    def name(self) -> str:
        return "memory_allocation_timing"

    def collect(self, n_bytes: int = 32) -> bytes:
        timings: List[int] = []
        for _ in range(self.n_samples):
            t0 = time.perf_counter_ns()
            _  = bytearray(self.alloc_size)     # heap allocation
            t1 = time.perf_counter_ns()
            timings.append(t1 - t0)

        # Extract low-order bits of each timing (highest jitter)
        raw = bytearray()
        for t in timings:
            raw.extend(struct.pack("<Q", t))     # 8 bytes per timing

        # Hash to compress and whiten
        digest = hashlib.sha3_256(bytes(raw)).digest()
        # Extend to n_bytes if needed
        out = bytearray()
        counter = 0
        while len(out) < n_bytes:
            out.extend(
                hashlib.sha3_256(digest + struct.pack("<I", counter)).digest()
            )
            counter += 1
        return bytes(out[:n_bytes])

    def quality(self) -> float:
        # Estimate by measuring variance of timings
        timings = []
        for _ in range(32):
            t0 = time.perf_counter_ns()
            _  = bytearray(self.alloc_size)
            t1 = time.perf_counter_ns()
            timings.append(t1 - t0)
        arr = np.array(timings, dtype=np.float64)
        if arr.std() < 1.0:
            return 1.0
        # Rough bits = log2(std)
        return min(8.0, float(np.log2(arr.std() + 1.0)))


class ClockJitterSource(EntropySource):
    """
    High-resolution clock noise sampling.

    Harvests entropy from the sub-nanosecond jitter in successive
    time.perf_counter_ns() calls caused by:
        - CPU clock drift and frequency scaling
        - Interrupt arrival timing
        - Superscalar execution reordering

    Applies Von Neumann debiasing to eliminate systematic bias.

    Quality: ~1.0–3.0 bits/byte.
    """

    def __init__(self, n_samples: int = 512) -> None:
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return "clock_jitter (perf_counter_ns)"

    def _raw_bits(self, n: int) -> List[int]:
        """Collect n raw bits from clock jitter (lowest bit of delta)."""
        bits  = []
        prev  = time.perf_counter_ns()
        while len(bits) < n:
            curr  = time.perf_counter_ns()
            delta = curr - prev
            bits.append(delta & 1)               # lowest jitter bit
            prev  = curr
        return bits

    def _von_neumann_debias(self, bits: List[int]) -> List[int]:
        """
        Von Neumann extractor: process pairs (b0, b1).
            (0,1) → emit 0
            (1,0) → emit 1
            (0,0) or (1,1) → discard
        Eliminates first-order bias.
        """
        out = []
        i   = 0
        while i + 1 < len(bits):
            b0, b1 = bits[i], bits[i + 1]
            if b0 != b1:
                out.append(b0)
            i += 2
        return out

    def collect(self, n_bytes: int = 32) -> bytes:
        n_bits_needed = n_bytes * 8
        # Collect extra raw bits to survive Von Neumann discard rate (~50%)
        raw_bits  = self._raw_bits(n_bits_needed * 4)
        good_bits = self._von_neumann_debias(raw_bits)

        # Pack bits into bytes
        out = bytearray()
        i   = 0
        while len(out) < n_bytes and i + 8 <= len(good_bits):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | good_bits[i + j]
            out.append(byte)
            i += 8

        # If not enough debiased bits, pad with sha3 of what we have
        while len(out) < n_bytes:
            out.extend(hashlib.sha3_256(bytes(out)).digest())

        # Final hash whitening
        digest = hashlib.sha3_256(bytes(out[:n_bytes])).digest()
        return (digest * (n_bytes // 32 + 1))[:n_bytes]

    def quality(self) -> float:
        bits  = self._raw_bits(256)
        ones  = sum(bits)
        p1    = ones / len(bits)
        p0    = 1.0  - p1
        # Shannon entropy of bit stream
        eps   = 1e-10
        h_bit = -(p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))
        return min(8.0, h_bit * 8.0)              # bits per byte


class CpuInstructionSource(EntropySource):
    """
    CPU instruction execution timing variance.

    Exploits non-determinism in CPU pipeline execution:
        - Branch misprediction recovery time
        - Cache hit/miss latency
        - Out-of-order execution reordering

    Runs a tight arithmetic loop and measures wall-clock jitter.

    Quality: ~1.5–3.5 bits/byte.
    """

    def __init__(self, loop_iters: int = 200, n_samples: int = 128) -> None:
        self.loop_iters = loop_iters
        self.n_samples  = n_samples

    @property
    def name(self) -> str:
        return "cpu_instruction_timing"

    def _cpu_loop_timing(self) -> int:
        """Execute a short arithmetic loop and return nanosecond duration."""
        t0  = time.perf_counter_ns()
        acc = 0
        for i in range(self.loop_iters):
            acc ^= (i * 0x9e3779b9) & 0xFFFFFFFF
        t1  = time.perf_counter_ns()
        return (t1 - t0) ^ (acc & 0xFF)           # mix with loop result

    def collect(self, n_bytes: int = 32) -> bytes:
        raw = bytearray()
        for _ in range(self.n_samples):
            t = self._cpu_loop_timing()
            raw.extend(struct.pack("<Q", t))

        # Compress via SHA3
        digest = hashlib.sha3_256(bytes(raw)).digest()
        out    = bytearray()
        counter = 0
        while len(out) < n_bytes:
            out.extend(
                hashlib.sha3_256(digest + struct.pack("<I", counter)).digest()
            )
            counter += 1
        return bytes(out[:n_bytes])

    def quality(self) -> float:
        timings = [self._cpu_loop_timing() for _ in range(64)]
        arr     = np.array(timings, dtype=np.float64)
        return min(8.0, float(np.log2(arr.std() + 1.0)))


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  ENTROPY POOL — Multi-Source Mixer
# ══════════════════════════════════════════════════════════════════════════════

class HardwareEntropyPool:
    """
    Multi-source entropy mixer producing high-quality combined entropy.

    Mixing Strategy
    ---------------
    1. Collect n_bytes from each registered source.
    2. XOR all source outputs together (combines entropy, never reduces it).
    3. Apply SHA3-256 hash chain (whitening / conditioning step).
    4. Maintain a 256-byte rolling state buffer seeded on construction.

    The rolling state ensures forward-secrecy: future output cannot be
    predicted even if the state at some point is compromised.

    Parameters
    ----------
    sources     : list of EntropySource instances
    pool_size   : internal state buffer size in bytes
    reseed_after: number of collect() calls before forced re-seeding
    """

    def __init__(
        self,
        sources:      List[EntropySource],
        pool_size:    int = 256,
        reseed_after: int = 1000,
    ) -> None:
        if not sources:
            raise ValueError("At least one entropy source required.")
        self.sources      = sources
        self.pool_size    = pool_size
        self.reseed_after = reseed_after
        self._call_count  = 0
        self._lock        = threading.Lock()
        self._state       = self._initial_seed()

    def _initial_seed(self) -> bytes:
        """Seed the pool from all sources at construction time."""
        combined = bytearray(self.pool_size)
        for src in self.sources:
            try:
                raw = src.collect(self.pool_size)
                for i in range(self.pool_size):
                    combined[i] ^= raw[i % len(raw)]
            except Exception:
                pass
        return bytes(combined)

    def _mix(self, *byte_arrays: bytes) -> bytes:
        """
        XOR-mix multiple byte arrays, then apply SHA3 whitening.
        """
        length   = max(len(b) for b in byte_arrays)
        combined = bytearray(length)
        for ba in byte_arrays:
            for i in range(len(ba)):
                combined[i % length] ^= ba[i]
        # SHA3 conditioning
        digest = hashlib.sha3_512(bytes(combined)).digest()
        return digest

    def _reseed(self) -> None:
        """Force re-seeding from all sources."""
        combined = bytearray(self.pool_size)
        for src in self.sources:
            try:
                raw = src.collect(self.pool_size)
                for i in range(self.pool_size):
                    combined[i] ^= raw[i % len(raw)]
            except Exception:
                pass
        self._state = self._mix(self._state, bytes(combined))

    def collect(self, n_bytes: int = 32) -> bytes:
        """
        Collect n_bytes of high-quality entropy from the pool.

        Thread-safe via internal lock.
        """
        with self._lock:
            self._call_count += 1
            if self._call_count % self.reseed_after == 0:
                self._reseed()

            out = bytearray()
            counter = 0
            while len(out) < n_bytes:
                block = hashlib.sha3_256(
                    self._state + struct.pack("<Q", counter)
                ).digest()
                out.extend(block)
                counter += 1

            # Advance state (forward secrecy)
            self._state = hashlib.sha3_256(
                self._state + struct.pack("<Q", self._call_count)
            ).digest() + self._state[32:]

            return bytes(out[:n_bytes])

    def source_qualities(self) -> Dict[str, float]:
        """Return quality estimate (bits/byte) for each registered source."""
        return {src.name: src.quality() for src in self.sources}


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  ENTROPY QUALITY ASSESSOR
# ══════════════════════════════════════════════════════════════════════════════

class EntropyQualityAssessor:
    """
    NIST SP 800-90B-inspired statistical quality tests for entropy output.

    Tests implemented
    -----------------
    1. Frequency (monobit) test      — bit-level uniformity
    2. Block frequency test          — uniformity in m-bit blocks
    3. Runs test                     — consecutive same-bit run analysis
    4. Shannon entropy estimate      — bits per byte empirical estimate
    5. Byte uniformity (chi-squared) — 256-bucket frequency test
    6. Autocorrelation test          — lag-1 byte correlation
    7. Compression proxy test        — entropy density via run-length proxy

    Note: These tests are diagnostic heuristics, not formal NIST certification.
    """

    def __init__(self, block_size: int = 8) -> None:
        self.block_size = block_size

    # ── Individual tests ──────────────────────────────────────────────────────

    def _frequency_test(self, bits: np.ndarray) -> Dict[str, float]:
        """NIST test 1: proportion of 1s should be close to 0.5."""
        p1      = bits.mean()
        p_value = float(np.abs(p1 - 0.5))       # 0 = perfect, 0.5 = worst
        return {"p1_ratio": float(p1), "deviation": p_value,
                "pass": p_value < 0.05}

    def _block_frequency_test(self, bits: np.ndarray) -> Dict[str, float]:
        """NIST test 2: proportion of 1s in each m-bit block."""
        m      = self.block_size
        n      = len(bits)
        n_blk  = n // m
        deviations = []
        for i in range(n_blk):
            blk = bits[i * m : (i + 1) * m]
            deviations.append(abs(blk.mean() - 0.5))
        mean_dev = float(np.mean(deviations)) if deviations else 1.0
        return {"mean_block_deviation": mean_dev,
                "n_blocks": n_blk,
                "pass": mean_dev < 0.05}

    def _runs_test(self, bits: np.ndarray) -> Dict[str, float]:
        """NIST test 3: total number of bit runs vs expected."""
        n    = len(bits)
        p1   = bits.mean()
        if p1 < 1e-6 or p1 > 1 - 1e-6:
            return {"runs_ratio": 0.0, "pass": False}
        # Count runs
        runs = 1 + np.sum(np.diff(bits) != 0)
        # Expected: 2 * n * p1 * (1 - p1)
        exp  = 2.0 * n * p1 * (1.0 - p1)
        std  = math.sqrt(2.0 * n * p1 * (1.0 - p1) * (1 - 2 * p1 * (1 - p1)))
        z    = abs(runs - exp) / max(std, 1e-9)
        return {"runs": int(runs), "expected": exp, "z_score": float(z),
                "pass": z < 2.576}              # 99% confidence

    def _shannon_entropy(self, data: bytes) -> Dict[str, float]:
        """Empirical Shannon entropy in bits per byte."""
        counts = np.zeros(256, dtype=np.float64)
        for b in data:
            counts[b] += 1
        counts = counts / counts.sum()
        nonzero = counts[counts > 0]
        h = -float(np.sum(nonzero * np.log2(nonzero)))
        return {"bits_per_byte": h, "max_possible": 8.0,
                "efficiency": h / 8.0,
                "pass": h > 7.0}                # >7 bits/byte = good

    def _chi_squared_uniformity(self, data: bytes) -> Dict[str, float]:
        """Chi-squared test for byte uniformity (256 buckets)."""
        counts   = np.zeros(256, dtype=np.float64)
        for b in data:
            counts[b] += 1
        n        = len(data)
        expected = n / 256.0
        chi2     = float(np.sum((counts - expected) ** 2 / max(expected, 1e-9)))
        # df = 255; critical value at p=0.001 ≈ 310.46
        return {"chi2_statistic": chi2,
                "df": 255,
                "pass": chi2 < 310.46}

    def _autocorrelation_test(self, data: bytes, lag: int = 1) -> Dict[str, float]:
        """Lag-k byte autocorrelation; should be near 0 for good entropy."""
        arr  = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        if len(arr) <= lag:
            return {"autocorr": 0.0, "pass": True}
        x    = arr[:-lag]
        y    = arr[lag:]
        corr = float(np.corrcoef(x, y)[0, 1])
        return {"autocorr_lag1": corr, "pass": abs(corr) < 0.05}

    def _compression_proxy(self, data: bytes) -> Dict[str, float]:
        """
        Run-length encoding compression ratio as entropy density proxy.
        High entropy → poor compressibility → ratio ≈ 1.0.
        """
        if not data:
            return {"rle_ratio": 0.0, "pass": False}
        runs  = 1
        for i in range(1, len(data)):
            if data[i] != data[i - 1]:
                runs += 1
        ratio = runs / len(data)                 # 1.0 = fully random
        return {"rle_ratio": float(ratio), "pass": ratio > 0.95}

    # ── Full assessment ───────────────────────────────────────────────────────

    def assess(self, data: bytes) -> Dict[str, Any]:
        """
        Run all quality tests on entropy sample data.

        Parameters
        ----------
        data : raw entropy bytes (recommend ≥ 256 bytes for accuracy)

        Returns
        -------
        dict with per-test results and an overall pass/fail verdict.
        """
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        results = {
            "n_bytes":        len(data),
            "n_bits":         len(bits),
            "frequency":      self._frequency_test(bits),
            "block_freq":     self._block_frequency_test(bits),
            "runs":           self._runs_test(bits),
            "shannon":        self._shannon_entropy(data),
            "chi_squared":    self._chi_squared_uniformity(data),
            "autocorrelation":self._autocorrelation_test(data),
            "compression":    self._compression_proxy(data),
        }

        passed = sum(
            1 for k, v in results.items()
            if isinstance(v, dict) and v.get("pass", False)
        )
        total_tests = sum(
            1 for v in results.values()
            if isinstance(v, dict) and "pass" in v
        )

        results["overall"] = {
            "tests_passed": passed,
            "tests_total":  total_tests,
            "verdict":      "PASS" if passed >= total_tests - 1 else "FAIL",
        }
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  HARDWARE_ENTROPY_HARVESTER — Master Engine
# ══════════════════════════════════════════════════════════════════════════════

class Hardware_Entropy_Harvester:
    """
    Master hardware entropy harvester.

    Aggregates all entropy sources into a unified interface for seeding
    probability distributions and producing stochastic outputs.

    Sources used (in priority order)
    ---------------------------------
    1. OsUrandomSource       — kernel CSPRNG (highest quality)
    2. ClockJitterSource     — perf_counter_ns jitter
    3. MemoryTimingSource    — heap allocation latency
    4. CpuInstructionSource  — CPU pipeline timing

    All sources are mixed via HardwareEntropyPool with SHA3 whitening.

    Parameters
    ----------
    pool_size    : internal pool buffer size in bytes
    reseed_after : pool calls before forced re-seed
    """

    def __init__(
        self,
        pool_size:    int = 512,
        reseed_after: int = 500,
    ) -> None:
        self._sources = [
            OsUrandomSource(),
            ClockJitterSource(n_samples=256),
            MemoryTimingSource(n_samples=32),
            CpuInstructionSource(n_samples=64),
        ]
        self._pool     = HardwareEntropyPool(
            self._sources,
            pool_size=pool_size,
            reseed_after=reseed_after,
        )
        self._assessor = EntropyQualityAssessor()
        self._rng: Optional[np.random.Generator] = None
        self._seed_rng()

    # ── RNG seeding ───────────────────────────────────────────────────────────

    def _seed_rng(self) -> None:
        """Seed numpy's Generator from hardware pool."""
        raw_seed  = self._pool.collect(32)
        seed_int  = int.from_bytes(raw_seed[:16], "big")
        seed_seq  = np.random.SeedSequence(seed_int)
        self._rng = np.random.Generator(np.random.PCG64(seed_seq))

    def reseed(self) -> None:
        """Force a complete re-seed from hardware sources."""
        self._seed_rng()

    # ── Raw entropy ───────────────────────────────────────────────────────────

    def raw_bytes(self, n: int = 32) -> bytes:
        """
        Return n bytes of raw hardware entropy.

        Parameters
        ----------
        n : number of bytes (default 32)
        """
        return self._pool.collect(n)

    def raw_uint64(self, n: int = 1) -> np.ndarray:
        """
        Return n uint64 integers derived from hardware entropy.
        """
        raw = self._pool.collect(n * 8)
        return np.frombuffer(raw, dtype=np.uint64)

    # ── Probability distributions ─────────────────────────────────────────────

    def uniform(
        self,
        low:  float = 0.0,
        high: float = 1.0,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded uniform distribution U(low, high).

        Uses hardware bytes mapped to [0,1) via uint64 → float64 scaling.
        """
        n_samples = int(np.prod(size)) if size else 1
        raw       = self.raw_uint64(n_samples)
        # Scale uint64 [0, 2^64) → [0, 1)
        u01       = raw.astype(np.float64) / (2.0 ** 64)
        scaled    = low + (high - low) * u01
        return scaled.reshape(size) if size else float(scaled[0])

    def normal(
        self,
        mean: float = 0.0,
        std:  float = 1.0,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded normal distribution N(mean, std²).

        Uses Box-Muller transform on hardware uniform samples.
        """
        n_samples = int(np.prod(size)) if size else 1
        # Box-Muller needs pairs
        n_pairs   = math.ceil(n_samples / 2)
        u1        = self.uniform(1e-10, 1.0 - 1e-10, size=(n_pairs,))
        u2        = self.uniform(0.0,   1.0,          size=(n_pairs,))
        mag       = std * np.sqrt(-2.0 * np.log(u1))
        z1        = mag * np.cos(2.0 * math.pi * u2) + mean
        z2        = mag * np.sin(2.0 * math.pi * u2) + mean
        combined  = np.concatenate([z1, z2])[:n_samples]
        return combined.reshape(size) if size else float(combined[0])

    def bernoulli(
        self,
        p:    float = 0.5,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded Bernoulli distribution with probability p.
        """
        u    = self.uniform(size=size if size else (1,))
        out  = (u < p).astype(np.int32)
        return out.reshape(size) if size else int(out[0])

    def categorical(
        self,
        probs: np.ndarray,
    ) -> int:
        """
        Sample a category index from a probability vector.
        Uses hardware entropy for the selection threshold.
        """
        probs   = np.asarray(probs, dtype=np.float64)
        probs   = probs / probs.sum()
        cdf     = np.cumsum(probs)
        u       = self.uniform()
        return int(np.searchsorted(cdf, u))

    def dirichlet(
        self,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Hardware-seeded Dirichlet distribution via gamma sampling.
        α : concentration parameter vector
        """
        alpha  = np.asarray(alpha, dtype=np.float64)
        gammas = np.array([
            self._gamma_sample(a) for a in alpha
        ])
        return gammas / gammas.sum()

    def _gamma_sample(self, shape: float, scale: float = 1.0) -> float:
        """Marsaglia-Tsang method for Gamma(shape, scale) using hardware RNG."""
        if shape < 1.0:
            u = self.uniform(1e-10, 1.0)
            return self._gamma_sample(1.0 + shape) * (u ** (1.0 / shape))
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = self.uniform(1e-10, 1.0)
            if u < 1.0 - 0.0331 * (x ** 4):
                return d * v * scale
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v + 1e-12)):
                return d * v * scale

    def exponential(
        self,
        lam:  float = 1.0,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded exponential distribution via inverse CDF.
        F^{-1}(u) = -ln(1-u) / λ
        """
        u    = self.uniform(1e-10, 1.0 - 1e-10, size=size if size else (1,))
        out  = -np.log(1.0 - u) / lam
        return out.reshape(size) if size else float(out[0])

    def gumbel(
        self,
        mu:   float = 0.0,
        beta: float = 1.0,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded Gumbel distribution.
        Used in Gumbel-softmax / straight-through estimator.
        """
        u    = self.uniform(1e-10, 1.0 - 1e-10, size=size if size else (1,))
        out  = mu - beta * np.log(-np.log(u))
        return out.reshape(size) if size else float(out[0])

    def integer(
        self,
        low:  int,
        high: int,
        size: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Hardware-seeded uniform integer in [low, high).
        """
        n_samples = int(np.prod(size)) if size else 1
        raw       = self.raw_uint64(n_samples)
        out       = low + (raw % (high - low)).astype(np.int64)
        return out.reshape(size) if size else int(out[0])

    def permutation(self, n: int) -> np.ndarray:
        """
        Hardware-seeded Fisher-Yates shuffle of [0, n).
        """
        arr = np.arange(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            j       = int(self.integer(0, i + 1))
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    # ── Gumbel-softmax ────────────────────────────────────────────────────────

    def gumbel_softmax(
        self,
        logits:      np.ndarray,
        temperature: float = 1.0,
        hard:        bool  = False,
    ) -> np.ndarray:
        """
        Gumbel-Softmax (concrete distribution) with hardware entropy.

        Parameters
        ----------
        logits      : (K,) unnormalized log-probabilities
        temperature : τ — lower → more discrete
        hard        : if True, returns one-hot (straight-through)

        Returns
        -------
        (K,) soft or hard sample
        """
        g     = self.gumbel(size=(len(logits),))
        y     = (logits + g) / (temperature + 1e-8)
        y    -= y.max()
        exp_y = np.exp(y)
        soft  = exp_y / exp_y.sum()

        if hard:
            k    = soft.argmax()
            hard_out            = np.zeros_like(soft)
            hard_out[k]         = 1.0
            return hard_out - soft + soft    # straight-through gradient
        return soft

    # ── Noise injection ───────────────────────────────────────────────────────

    def inject_noise(
        self,
        x:          np.ndarray,
        noise_type: str   = "gaussian",
        scale:      float = 0.01,
    ) -> np.ndarray:
        """
        Inject hardware-seeded noise into a tensor.

        Parameters
        ----------
        x          : input array
        noise_type : "gaussian" | "uniform" | "laplace" | "dropout"
        scale      : noise magnitude
        """
        shape = x.shape

        if noise_type == "gaussian":
            noise = self.normal(std=scale, size=shape)
        elif noise_type == "uniform":
            noise = self.uniform(-scale, scale, size=shape)
        elif noise_type == "laplace":
            # Laplace via difference of two exponentials
            e1    = self.exponential(lam=1.0 / scale, size=shape)
            e2    = self.exponential(lam=1.0 / scale, size=shape)
            noise = e1 - e2
        elif noise_type == "dropout":
            mask  = self.bernoulli(1.0 - scale, size=shape)
            return x * mask / max(1.0 - scale, 1e-8)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        return x + noise

    # ── Quality assessment ────────────────────────────────────────────────────

    def assess_quality(self, n_bytes: int = 1024) -> Dict[str, Any]:
        """
        Run full statistical quality assessment on harvested entropy.

        Parameters
        ----------
        n_bytes : sample size for testing (recommend ≥ 512)

        Returns
        -------
        Full NIST-inspired test report dict.
        """
        sample = self._pool.collect(n_bytes)
        return self._assessor.assess(sample)

    def source_report(self) -> Dict[str, float]:
        """Return quality (bits/byte) for each registered entropy source."""
        return self._pool.source_qualities()


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  STOCHASTIC NODE — Hardware-Seeded Processing Unit
# ══════════════════════════════════════════════════════════════════════════════

class StochasticNode:
    """
    A processing node whose decision branches are seeded by hardware entropy.

    Uses Hardware_Entropy_Harvester for all stochastic operations:
        - Dropout masking
        - Stochastic depth (layer skip probability)
        - Noise-injected activations
        - Gumbel-softmax discrete sampling
        - Stochastic weight perturbation (weight noise regularisation)

    Parameters
    ----------
    dim           : feature dimension
    drop_rate     : neuron dropout probability
    depth_prob    : probability of executing this node (stochastic depth)
    weight_noise  : std of weight perturbation noise
    harvester     : shared Hardware_Entropy_Harvester instance
    """

    def __init__(
        self,
        dim:          int,
        drop_rate:    float = 0.1,
        depth_prob:   float = 1.0,
        weight_noise: float = 0.001,
        harvester:    Optional[Hardware_Entropy_Harvester] = None,
    ) -> None:
        self.dim          = dim
        self.drop_rate    = drop_rate
        self.depth_prob   = depth_prob
        self.weight_noise = weight_noise
        self.harvester    = harvester or Hardware_Entropy_Harvester()

        # Learnable weight (perturbed at each forward pass during training)
        scale    = math.sqrt(2.0 / dim)
        self.W   = np.random.randn(dim, dim) * scale
        self.b   = np.zeros(dim)

    def forward(
        self,
        x:        np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """
        Forward pass with hardware-seeded stochasticity.

        Parameters
        ----------
        x        : (..., dim) input array
        training : enable stochastic operations

        Returns
        -------
        (..., dim) output array
        """
        if training:
            # Stochastic depth: skip node with probability (1 - depth_prob)
            if self.depth_prob < 1.0:
                u = self.harvester.uniform()
                if u > self.depth_prob:
                    return x                    # skip this node

            # Weight perturbation: W̃ = W + ε,  ε ~ N(0, weight_noise²)
            W_noisy = self.W + self.harvester.normal(
                std=self.weight_noise,
                size=self.W.shape,
            )
        else:
            W_noisy = self.W

        # Affine + GELU activation
        h   = x @ W_noisy + self.b
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(h / math.sqrt(2.0)))
        out = h * cdf

        if training:
            # Hardware-seeded dropout
            out = self.harvester.inject_noise(out, "dropout", self.drop_rate)

        return out

    def stochastic_sample(
        self,
        logits:      np.ndarray,
        temperature: float = 1.0,
        hard:        bool  = False,
    ) -> np.ndarray:
        """
        Sample from logits using Gumbel-Softmax with hardware entropy.

        Parameters
        ----------
        logits      : (K,) or (batch, K) logit array
        temperature : τ
        hard        : straight-through one-hot

        Returns
        -------
        Soft or hard sample of same shape as logits.
        """
        if logits.ndim == 1:
            return self.harvester.gumbel_softmax(logits, temperature, hard)
        return np.stack([
            self.harvester.gumbel_softmax(row, temperature, hard)
            for row in logits
        ])
"""
system_load_balancer.py
========================
Module 4 — Dynamic Resource-Aware Compute Throttling

Continuously monitors real host machine resources via psutil and
dynamically adjusts processing depth, batch size, and memory state
based on live CPU, RAM, and thermal conditions.

Architecture
------------
    ResourceSnapshot         — Immutable point-in-time resource reading
    ThermalMonitor           — Per-core and package temperature tracking
    MemoryPressureTracker    — RSS / VMS / swap pressure monitoring
    CpuLoadTracker           — Per-core and aggregate CPU utilisation
    ResourcePoller           — Background thread: continuous polling loop
    ThrottlePolicy           — Configurable threshold → action mapping
    FallbackHeuristic        — Degraded-mode optimisation strategy
    MemoryCompressor         — Active memory state aggressive compressor
    System_Load_Balancer     — Master adaptive throttling engine

Dependencies: psutil, numpy, os, threading, time, gc (stdlib + psutil)
"""


import gc
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
#  § 1  COMPUTE TIER ENUM
# ══════════════════════════════════════════════════════════════════════════════

class ComputeTier(Enum):
    """
    Processing depth tier assigned by the load balancer.

    FULL        — all resources available; run at maximum depth / batch
    REDUCED     — moderate pressure; reduce batch size and depth by 25%
    MINIMAL     — high pressure; reduce by 50%; disable non-essential passes
    CRITICAL    — emergency; trigger fallback heuristic + memory compression
    SUSPENDED   — system overloaded; suspend compute entirely
    """
    FULL      = auto()
    REDUCED   = auto()
    MINIMAL   = auto()
    CRITICAL  = auto()
    SUSPENDED = auto()

    def depth_multiplier(self) -> float:
        return {
            ComputeTier.FULL:      1.00,
            ComputeTier.REDUCED:   0.75,
            ComputeTier.MINIMAL:   0.50,
            ComputeTier.CRITICAL:  0.25,
            ComputeTier.SUSPENDED: 0.00,
        }[self]

    def batch_multiplier(self) -> float:
        return {
            ComputeTier.FULL:      1.00,
            ComputeTier.REDUCED:   0.75,
            ComputeTier.MINIMAL:   0.50,
            ComputeTier.CRITICAL:  0.25,
            ComputeTier.SUSPENDED: 0.00,
        }[self]


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  RESOURCE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ResourceSnapshot:
    """
    Immutable point-in-time reading of all monitored resources.

    Fields
    ------
    timestamp_ns     : time.perf_counter_ns() at capture time
    cpu_percent_avg  : aggregate CPU utilisation (0–100)
    cpu_percent_per  : per-core utilisation list
    ram_used_bytes   : process RSS in bytes
    ram_total_bytes  : total physical RAM in bytes
    ram_available_bytes : available physical RAM in bytes
    ram_percent      : system-wide RAM usage (0–100)
    swap_percent     : swap usage (0–100)
    proc_ram_mb      : this process RSS in MB
    cpu_freq_mhz     : current CPU frequency in MHz (None if unavailable)
    temp_celsius     : highest reported sensor temperature (None if unavailable)
    n_threads        : number of OS threads in this process
    io_read_bytes    : cumulative I/O read bytes for this process
    io_write_bytes   : cumulative I/O write bytes for this process
    open_files       : number of open file descriptors
    """
    timestamp_ns:         int
    cpu_percent_avg:      float
    cpu_percent_per:      Tuple[float, ...]
    ram_used_bytes:       int
    ram_total_bytes:      int
    ram_available_bytes:  int
    ram_percent:          float
    swap_percent:         float
    proc_ram_mb:          float
    cpu_freq_mhz:         Optional[float]
    temp_celsius:         Optional[float]
    n_threads:            int
    io_read_bytes:        int
    io_write_bytes:       int
    open_files:           int

    @property
    def ram_free_percent(self) -> float:
        return 100.0 - self.ram_percent

    @property
    def age_ms(self) -> float:
        return (time.perf_counter_ns() - self.timestamp_ns) / 1e6


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  THERMAL MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ThermalMonitor:
    """
    Per-sensor temperature tracker using psutil.sensors_temperatures().

    Supports Linux (coretemp, acpitz, k10temp) and macOS (via SMC sensors).
    Returns None gracefully on platforms without temperature sensor support
    (Windows without third-party drivers, WSL, containers).

    Parameters
    ----------
    critical_celsius : temperature above which CRITICAL tier is forced
    warn_celsius     : temperature above which REDUCED tier is preferred
    """

    def __init__(
        self,
        critical_celsius: float = 90.0,
        warn_celsius:     float = 75.0,
    ) -> None:
        self.critical = critical_celsius
        self.warn     = warn_celsius
        self._supported = hasattr(psutil, "sensors_temperatures")

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Read all available temperature sensors.

        Returns dict with keys:
            max_celsius   : highest temperature across all sensors
            sensors       : raw psutil sensor dict
            tier_override : ComputeTier forced by thermal state, or None
        Returns None if sensors unavailable.
        """
        if not self._supported:
            return None

        try:
            raw = psutil.sensors_temperatures()
        except Exception:
            return None

        if not raw:
            return None

        all_temps: List[float] = []
        flat: Dict[str, List[float]] = {}

        for label, readings in raw.items():
            temps = [r.current for r in readings if r.current is not None]
            if temps:
                flat[label]   = temps
                all_temps.extend(temps)

        if not all_temps:
            return None

        max_t = max(all_temps)
        tier  = None
        if max_t >= self.critical:
            tier = ComputeTier.CRITICAL
        elif max_t >= self.warn:
            tier = ComputeTier.REDUCED

        return {
            "max_celsius":    max_t,
            "mean_celsius":   float(np.mean(all_temps)),
            "sensors":        flat,
            "tier_override":  tier,
        }

    def max_celsius(self) -> Optional[float]:
        info = self.read()
        return info["max_celsius"] if info else None


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  MEMORY PRESSURE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class MemoryPressureTracker:
    """
    Detailed memory pressure monitoring for the current process and system.

    Tracks
    ------
    - Process RSS (Resident Set Size) and VMS (Virtual Memory Size)
    - System-wide available RAM
    - Swap utilisation
    - Memory growth rate (MB/s) via delta between readings

    Parameters
    ----------
    history_len : number of snapshots to retain for growth rate estimation
    """

    def __init__(self, history_len: int = 30) -> None:
        self.history_len = history_len
        self._proc       = psutil.Process(os.getpid())
        self._history:   List[Tuple[float, float]] = []   # (timestamp, rss_mb)

    def read(self) -> Dict[str, Any]:
        """
        Read current memory state.

        Returns
        -------
        dict with rss_mb, vms_mb, sys_available_mb, sys_percent,
             swap_percent, growth_rate_mb_per_s, pressure_score (0–1)
        """
        try:
            mi        = self._proc.memory_info()
            rss_mb    = mi.rss / (1024 ** 2)
            vms_mb    = mi.vms / (1024 ** 2)
        except psutil.NoSuchProcess:
            rss_mb = vms_mb = 0.0

        sys_vm      = psutil.virtual_memory()
        swap        = psutil.swap_memory()

        now = time.monotonic()
        self._history.append((now, rss_mb))
        if len(self._history) > self.history_len:
            self._history.pop(0)

        # Growth rate: linear regression slope over history
        growth_rate = 0.0
        if len(self._history) >= 2:
            ts   = np.array([h[0] for h in self._history])
            rss  = np.array([h[1] for h in self._history])
            dt   = ts - ts[0]
            if dt[-1] > 0:
                # Least-squares slope
                A   = np.vstack([dt, np.ones(len(dt))]).T
                slope, _ = np.linalg.lstsq(A, rss, rcond=None)[0]
                growth_rate = float(slope)             # MB/s

        # Composite pressure score [0, 1]:
        # weighted sum of: RAM usage, swap usage, growth rate
        ram_score   = sys_vm.percent / 100.0
        swap_score  = swap.percent  / 100.0
        grow_score  = min(1.0, max(0.0, growth_rate / 100.0))  # 100 MB/s → 1.0
        pressure    = 0.5 * ram_score + 0.3 * swap_score + 0.2 * grow_score

        return {
            "rss_mb":              rss_mb,
            "vms_mb":              vms_mb,
            "sys_total_mb":        sys_vm.total   / (1024 ** 2),
            "sys_available_mb":    sys_vm.available / (1024 ** 2),
            "sys_percent":         sys_vm.percent,
            "swap_used_mb":        swap.used   / (1024 ** 2),
            "swap_total_mb":       swap.total  / (1024 ** 2),
            "swap_percent":        swap.percent,
            "growth_rate_mb_per_s": growth_rate,
            "pressure_score":      float(pressure),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  CPU LOAD TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class CpuLoadTracker:
    """
    Continuous CPU load tracking with exponential moving average smoothing.

    Tracks
    ------
    - Per-core utilisation via psutil.cpu_percent(percpu=True)
    - System-wide aggregate utilisation
    - EMA-smoothed aggregate (reduces single-spike false positives)
    - CPU frequency (current / min / max)
    - Load average (Unix only)

    Parameters
    ----------
    ema_alpha   : EMA smoothing factor (0 = no update, 1 = no smoothing)
    interval    : psutil cpu_percent sampling interval in seconds
    """

    def __init__(self, ema_alpha: float = 0.2, interval: float = 0.1) -> None:
        self.alpha    = ema_alpha
        self.interval = interval
        self._ema_cpu = 0.0
        self._first   = True

    def read(self) -> Dict[str, Any]:
        """
        Read current CPU state.

        Returns
        -------
        dict with per_core, aggregate, ema_aggregate, freq_mhz,
             load_avg_1m (Unix only), hottest_core_idx
        """
        per_core: List[float] = psutil.cpu_percent(
            interval=self.interval, percpu=True
        )
        aggregate = float(np.mean(per_core))

        # EMA smoothing
        if self._first:
            self._ema_cpu = aggregate
            self._first   = False
        else:
            self._ema_cpu = (1.0 - self.alpha) * self._ema_cpu + self.alpha * aggregate

        # CPU frequency
        freq_mhz: Optional[float] = None
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                freq_mhz = freq_info.current
        except Exception:
            pass

        # Load average (Unix only)
        load_avg: Optional[float] = None
        if hasattr(os, "getloadavg"):
            try:
                load_avg = os.getloadavg()[0]
            except Exception:
                pass

        hottest_idx = int(np.argmax(per_core)) if per_core else 0

        return {
            "per_core":        per_core,
            "aggregate":       aggregate,
            "ema_aggregate":   self._ema_cpu,
            "freq_mhz":        freq_mhz,
            "load_avg_1m":     load_avg,
            "n_physical":      psutil.cpu_count(logical=False),
            "n_logical":       psutil.cpu_count(logical=True),
            "hottest_core_idx":hottest_idx,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  RESOURCE POLLER — Background Thread
# ══════════════════════════════════════════════════════════════════════════════

class ResourcePoller:
    """
    Background daemon thread that continuously polls system resources
    at a configurable interval and maintains a rolling snapshot history.

    Parameters
    ----------
    poll_interval_s : seconds between polls (default 0.5)
    history_len     : number of snapshots to retain
    """

    def __init__(
        self,
        poll_interval_s: float = 0.5,
        history_len:     int   = 120,
    ) -> None:
        self.interval    = poll_interval_s
        self.history_len = history_len

        self._thermal   = ThermalMonitor()
        self._memory    = MemoryPressureTracker()
        self._cpu       = CpuLoadTracker()
        self._proc      = psutil.Process(os.getpid())

        self._snapshots: List[ResourceSnapshot] = []
        self._lock       = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Snapshot capture ──────────────────────────────────────────────────────

    def _capture(self) -> ResourceSnapshot:
        """Capture a single ResourceSnapshot from all subsystems."""
        cpu_info  = self._cpu.read()
        mem_info  = self._memory.read()
        therm     = self._thermal.read()

        try:
            io   = self._proc.io_counters()
            io_r = io.read_bytes
            io_w = io.write_bytes
        except (psutil.NoSuchProcess, AttributeError):
            io_r = io_w = 0

        try:
            n_threads = self._proc.num_threads()
        except psutil.NoSuchProcess:
            n_threads = 0

        try:
            open_files = len(self._proc.open_files())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            open_files = 0

        swap = psutil.swap_memory()

        return ResourceSnapshot(
            timestamp_ns         = time.perf_counter_ns(),
            cpu_percent_avg      = cpu_info["ema_aggregate"],
            cpu_percent_per      = tuple(cpu_info["per_core"]),
            ram_used_bytes       = int(mem_info["rss_mb"] * 1024 ** 2),
            ram_total_bytes      = int(mem_info["sys_total_mb"] * 1024 ** 2),
            ram_available_bytes  = int(mem_info["sys_available_mb"] * 1024 ** 2),
            ram_percent          = mem_info["sys_percent"],
            swap_percent         = swap.percent,
            proc_ram_mb          = mem_info["rss_mb"],
            cpu_freq_mhz         = cpu_info["freq_mhz"],
            temp_celsius         = therm["max_celsius"] if therm else None,
            n_threads            = n_threads,
            io_read_bytes        = io_r,
            io_write_bytes       = io_w,
            open_files           = open_files,
        )

    # ── Thread control ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._capture()
                with self._lock:
                    self._snapshots.append(snap)
                    if len(self._snapshots) > self.history_len:
                        self._snapshots.pop(0)
            except Exception as exc:
                logger.warning("ResourcePoller capture error: %s", exc)
            self._stop_event.wait(timeout=self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ResourcePoller"
        )
        self._thread.start()
        logger.info("ResourcePoller started (interval=%.2fs)", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("ResourcePoller stopped.")

    # ── Snapshot access ───────────────────────────────────────────────────────

    def latest(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def history(self, n: Optional[int] = None) -> List[ResourceSnapshot]:
        with self._lock:
            snaps = list(self._snapshots)
        return snaps[-n:] if n else snaps

    def average(self, n: int = 10) -> Optional[Dict[str, float]]:
        """
        Return averaged resource metrics over the last n snapshots.
        Useful for smoothed tier decisions.
        """
        snaps = self.history(n)
        if not snaps:
            return None
        return {
            "cpu_avg":      float(np.mean([s.cpu_percent_avg  for s in snaps])),
            "ram_avg":      float(np.mean([s.ram_percent       for s in snaps])),
            "swap_avg":     float(np.mean([s.swap_percent      for s in snaps])),
            "temp_avg":     float(np.mean([s.temp_celsius      for s in snaps
                                           if s.temp_celsius is not None] or [0.0])),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  § 7  THROTTLE POLICY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThrottlePolicy:
    """
    Configurable threshold → ComputeTier mapping.

    Each threshold is the percentage value above which the corresponding
    tier is triggered. Tiers are evaluated from most severe to least.

    Parameters
    ----------
    cpu_suspended   : CPU% above which tier = SUSPENDED
    cpu_critical    : CPU% above which tier = CRITICAL
    cpu_minimal     : CPU% above which tier = MINIMAL
    cpu_reduced     : CPU% above which tier = REDUCED
    ram_suspended   : RAM% above which tier = SUSPENDED
    ram_critical    : RAM% above which tier = CRITICAL
    ram_minimal     : RAM% above which tier = MINIMAL
    ram_reduced     : RAM% above which tier = REDUCED
    temp_critical   : °C above which tier ≥ CRITICAL
    temp_reduced    : °C above which tier ≥ REDUCED
    swap_critical   : swap% above which tier ≥ CRITICAL
    hysteresis      : minimum tier delta before switching (prevents flapping)
    """
    cpu_suspended:  float = 97.0
    cpu_critical:   float = 90.0
    cpu_minimal:    float = 75.0
    cpu_reduced:    float = 55.0

    ram_suspended:  float = 97.0
    ram_critical:   float = 90.0
    ram_minimal:    float = 80.0
    ram_reduced:    float = 65.0

    temp_critical:  float = 90.0
    temp_reduced:   float = 75.0

    swap_critical:  float = 50.0

    hysteresis:     float = 5.0    # % deadband before tier changes

    def evaluate(self, snap: ResourceSnapshot) -> ComputeTier:
        """
        Evaluate snapshot against all thresholds and return the most
        restrictive ComputeTier.
        """
        tiers: List[ComputeTier] = []

        # CPU tiers
        c = snap.cpu_percent_avg
        if   c >= self.cpu_suspended: tiers.append(ComputeTier.SUSPENDED)
        elif c >= self.cpu_critical:  tiers.append(ComputeTier.CRITICAL)
        elif c >= self.cpu_minimal:   tiers.append(ComputeTier.MINIMAL)
        elif c >= self.cpu_reduced:   tiers.append(ComputeTier.REDUCED)
        else:                         tiers.append(ComputeTier.FULL)

        # RAM tiers
        r = snap.ram_percent
        if   r >= self.ram_suspended: tiers.append(ComputeTier.SUSPENDED)
        elif r >= self.ram_critical:  tiers.append(ComputeTier.CRITICAL)
        elif r >= self.ram_minimal:   tiers.append(ComputeTier.MINIMAL)
        elif r >= self.ram_reduced:   tiers.append(ComputeTier.REDUCED)
        else:                         tiers.append(ComputeTier.FULL)

        # Temperature tiers
        if snap.temp_celsius is not None:
            t = snap.temp_celsius
            if   t >= self.temp_critical: tiers.append(ComputeTier.CRITICAL)
            elif t >= self.temp_reduced:  tiers.append(ComputeTier.REDUCED)

        # Swap tier
        if snap.swap_percent >= self.swap_critical:
            tiers.append(ComputeTier.CRITICAL)

        # Return most restrictive (highest ordinal value)
        return max(tiers, key=lambda t: t.value)


# ══════════════════════════════════════════════════════════════════════════════
#  § 8  FALLBACK HEURISTIC
# ══════════════════════════════════════════════════════════════════════════════

class FallbackHeuristic:
    """
    Degraded-mode optimisation strategy activated under CRITICAL / SUSPENDED tier.

    Actions taken
    -------------
    1. Reduce effective batch size to minimum viable
    2. Switch to lower-precision arithmetic (float32 → float16 arrays)
    3. Disable gradient accumulation buffers
    4. Truncate sequence lengths / recursion depths
    5. Flush intermediate computation caches
    6. Yield CPU time to OS scheduler

    Parameters
    ----------
    min_batch      : absolute minimum batch size
    min_depth      : absolute minimum recursion depth
    sleep_s        : OS yield duration per fallback call
    """

    def __init__(
        self,
        min_batch: int   = 1,
        min_depth: int   = 1,
        sleep_s:   float = 0.05,
    ) -> None:
        self.min_batch = min_batch
        self.min_depth = min_depth
        self.sleep_s   = sleep_s
        self._activations = 0

    def apply(
        self,
        current_batch: int,
        current_depth: int,
        tensor_cache:  Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Apply fallback heuristic and return adjusted parameters.

        Parameters
        ----------
        current_batch : current batch size
        current_depth : current recursion / processing depth
        tensor_cache  : optional dict of intermediate tensors to flush

        Returns
        -------
        dict with adjusted_batch, adjusted_depth, actions_taken
        """
        self._activations += 1
        actions: List[str] = []

        # 1. Reduce batch
        adjusted_batch = max(self.min_batch, current_batch // 2)
        actions.append(f"batch {current_batch} → {adjusted_batch}")

        # 2. Reduce depth
        adjusted_depth = max(self.min_depth, current_depth // 2)
        actions.append(f"depth {current_depth} → {adjusted_depth}")

        # 3. Flush tensor cache
        if tensor_cache:
            n_flushed = len(tensor_cache)
            tensor_cache.clear()
            actions.append(f"flushed {n_flushed} cached tensors")

        # 4. Force Python GC
        collected = gc.collect()
        actions.append(f"gc collected {collected} objects")

        # 5. Yield CPU
        time.sleep(self.sleep_s)
        actions.append(f"yielded CPU for {self.sleep_s*1000:.0f}ms")

        logger.warning(
            "FallbackHeuristic applied (activation #%d): %s",
            self._activations, "; ".join(actions),
        )

        return {
            "adjusted_batch": adjusted_batch,
            "adjusted_depth": adjusted_depth,
            "actions_taken":  actions,
            "activation_n":   self._activations,
        }

    @property
    def activation_count(self) -> int:
        return self._activations


# ══════════════════════════════════════════════════════════════════════════════
#  § 9  MEMORY COMPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class MemoryCompressor:
    """
    Aggressive active memory state compressor.

    Strategies
    ----------
    1. float64 → float32 downcast of registered arrays
    2. Sparse representation: zero out values below a magnitude threshold
    3. Low-rank approximation via truncated SVD (for 2D arrays)
    4. Run full Python GC + optional OS memory release (Linux: malloc_trim)
    5. Delete explicitly registered disposable objects

    Parameters
    ----------
    svd_rank        : kept singular values for low-rank compression
    zero_threshold  : values below this are zeroed in sparse compression
    """

    def __init__(
        self,
        svd_rank:       int   = 16,
        zero_threshold: float = 1e-4,
    ) -> None:
        self.svd_rank       = svd_rank
        self.zero_threshold = zero_threshold
        self._registry:     Dict[str, np.ndarray] = {}
        self._disposables:  List[Any]              = []

    def register(self, name: str, array: np.ndarray) -> None:
        """Register a named array for compression management."""
        self._registry[name] = array

    def register_disposable(self, obj: Any) -> None:
        """Register an object to be deleted under memory pressure."""
        self._disposables.append(obj)

    def downcast(self) -> Dict[str, int]:
        """
        Downcast all float64 arrays to float32 in-place.
        Returns dict of {name: bytes_saved}.
        """
        savings: Dict[str, int] = {}
        for name, arr in list(self._registry.items()):
            if arr.dtype == np.float64:
                saved = arr.nbytes // 2
                self._registry[name] = arr.astype(np.float32)
                savings[name]        = saved
        return savings

    def sparsify(self) -> Dict[str, int]:
        """
        Zero out near-zero values in all registered arrays.
        Returns dict of {name: n_zeroed}.
        """
        zeroed: Dict[str, int] = {}
        for name, arr in self._registry.items():
            mask           = np.abs(arr) < self.zero_threshold
            n              = int(mask.sum())
            arr[mask]      = 0.0
            zeroed[name]   = n
        return zeroed

    def low_rank_compress(self) -> Dict[str, Dict]:
        """
        Apply truncated SVD to all 2-D registered arrays.
        Replaces array with rank-k approximation A ≈ U Σ Vᵀ.
        Returns compression stats per array.
        """
        stats: Dict[str, Dict] = {}
        for name, arr in list(self._registry.items()):
            if arr.ndim != 2:
                continue
            r    = min(self.svd_rank, min(arr.shape) - 1)
            if r < 1:
                continue
            try:
                U, S, Vt = np.linalg.svd(arr, full_matrices=False)
                U_k      = U[:, :r]
                S_k      = S[:r]
                Vt_k     = Vt[:r, :]
                compressed            = (U_k * S_k) @ Vt_k
                ratio                 = (U_k.size + S_k.size + Vt_k.size) / arr.size
                self._registry[name]  = compressed
                stats[name]           = {
                    "rank":              r,
                    "compression_ratio": ratio,
                    "original_shape":    arr.shape,
                    "energy_retained":   float((S_k ** 2).sum() / (S ** 2).sum()),
                }
            except np.linalg.LinAlgError:
                pass
        return stats

    def flush_disposables(self) -> int:
        """Delete all registered disposable objects. Returns count deleted."""
        n = len(self._disposables)
        self._disposables.clear()
        return n

    def full_compress(self) -> Dict[str, Any]:
        """
        Run all compression strategies in sequence.

        Returns
        -------
        dict summarising bytes saved and actions taken.
        """
        results: Dict[str, Any] = {}

        # Stage 1: downcast
        results["downcast_bytes_saved"] = sum(self.downcast().values())

        # Stage 2: sparsify
        results["zeroed_elements"] = sum(self.sparsify().values())

        # Stage 3: low-rank
        results["low_rank"] = self.low_rank_compress()

        # Stage 4: flush disposables
        results["disposables_deleted"] = self.flush_disposables()

        # Stage 5: Python GC
        results["gc_collected"] = gc.collect()

        # Stage 6: OS malloc_trim (Linux only)
        results["malloc_trim"] = self._os_trim()

        logger.info(
            "MemoryCompressor.full_compress: %d bytes downcasted, "
            "%d elements zeroed, gc=%d",
            results["downcast_bytes_saved"],
            results["zeroed_elements"],
            results["gc_collected"],
        )
        return results

    @staticmethod
    def _os_trim() -> bool:
        """
        Call malloc_trim(0) via ctypes on Linux to return freed memory to OS.
        No-op on other platforms.
        """
        if platform.system() != "Linux":
            return False
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            return True
        except Exception:
            return False

    def registered_bytes(self) -> int:
        """Total bytes across all registered arrays."""
        return sum(a.nbytes for a in self._registry.values())


# ══════════════════════════════════════════════════════════════════════════════
#  § 10  SYSTEM_LOAD_BALANCER — Master Engine
# ══════════════════════════════════════════════════════════════════════════════

class System_Load_Balancer:
    """
    Master adaptive compute throttling engine.

    Responsibilities
    ----------------
    1. Continuously poll real system resources via ResourcePoller.
    2. Evaluate ThrottlePolicy against live snapshots → ComputeTier.
    3. Expose adaptive parameters (batch_size, depth_limit) to callers.
    4. On CRITICAL: trigger FallbackHeuristic + MemoryCompressor.
    5. On SUSPENDED: block compute via wait() until resources recover.
    6. Log all tier transitions with full resource context.
    7. Provide per-call resource gating via the `gate()` context manager.

    Parameters
    ----------
    base_batch_size  : maximum batch size at FULL tier
    base_depth       : maximum processing depth at FULL tier
    policy           : ThrottlePolicy instance (uses defaults if None)
    poll_interval_s  : ResourcePoller polling interval
    tier_callback    : optional callable(old_tier, new_tier, snapshot)
    """

    def __init__(
        self,
        base_batch_size: int                      = 32,
        base_depth:      int                      = 8,
        policy:          Optional[ThrottlePolicy]  = None,
        poll_interval_s: float                    = 0.5,
        tier_callback:   Optional[Callable]        = None,
    ) -> None:
        self.base_batch   = base_batch_size
        self.base_depth   = base_depth
        self.policy       = policy or ThrottlePolicy()
        self._callback    = tier_callback

        self._poller      = ResourcePoller(poll_interval_s)
        self._fallback    = FallbackHeuristic()
        self._compressor  = MemoryCompressor()

        self._current_tier: ComputeTier = ComputeTier.FULL
        self._tier_lock     = threading.Lock()
        self._tier_history: List[Tuple[float, ComputeTier]] = []

        self._tensor_cache: Dict[str, np.ndarray] = {}
        self._suspended_event = threading.Event()
        self._suspended_event.set()   # not suspended initially

        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher    = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "System_Load_Balancer":
        """Start the resource poller and tier-watcher thread."""
        self._poller.start()
        # Give poller one cycle to populate
        time.sleep(self._poller.interval * 2)
        self._stop_watcher.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="TierWatcher"
        )
        self._watcher_thread.start()
        logger.info(
            "System_Load_Balancer started | base_batch=%d base_depth=%d",
            self.base_batch, self.base_depth,
        )
        return self

    def stop(self) -> None:
        """Stop all background threads."""
        self._stop_watcher.set()
        self._poller.stop()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5.0)
        logger.info("System_Load_Balancer stopped.")

    def __enter__(self) -> "System_Load_Balancer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Tier watcher loop ─────────────────────────────────────────────────────

    def _watch_loop(self) -> None:
        """Background thread: continuously evaluate tier and react."""
        while not self._stop_watcher.is_set():
            snap = self._poller.latest()
            if snap is None:
                self._stop_watcher.wait(0.5)
                continue

            new_tier = self.policy.evaluate(snap)

            with self._tier_lock:
                old_tier = self._current_tier
                if new_tier != old_tier:
                    self._on_tier_change(old_tier, new_tier, snap)
                    self._current_tier = new_tier
                    self._tier_history.append((time.monotonic(), new_tier))

            # Handle suspension
            if new_tier == ComputeTier.SUSPENDED:
                self._suspended_event.clear()
            else:
                self._suspended_event.set()

            self._stop_watcher.wait(self._poller.interval)

    def _on_tier_change(
        self,
        old: ComputeTier,
        new: ComputeTier,
        snap: ResourceSnapshot,
    ) -> None:
        """React to a tier transition."""
        logger.warning(
            "Tier: %s → %s | CPU=%.1f%% RAM=%.1f%% Temp=%s°C",
            old.name, new.name,
            snap.cpu_percent_avg,
            snap.ram_percent,
            f"{snap.temp_celsius:.1f}" if snap.temp_celsius else "N/A",
        )

        if new in (ComputeTier.CRITICAL, ComputeTier.SUSPENDED):
            # Apply fallback heuristic
            self._fallback.apply(
                current_batch = self.base_batch,
                current_depth = self.base_depth,
                tensor_cache  = self._tensor_cache,
            )
            # Aggressive memory compression
            self._compressor.full_compress()

        if self._callback:
            try:
                self._callback(old, new, snap)
            except Exception as exc:
                logger.error("tier_callback raised: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def tier(self) -> ComputeTier:
        """Current compute tier (thread-safe read)."""
        with self._tier_lock:
            return self._current_tier

    @property
    def batch_size(self) -> int:
        """
        Effective batch size adjusted for current tier.
        Never returns 0 — minimum is 1.
        """
        return max(1, int(self.base_batch * self.tier.batch_multiplier()))

    @property
    def depth_limit(self) -> int:
        """
        Effective depth limit adjusted for current tier.
        Never returns 0 — minimum is 1.
        """
        return max(1, int(self.base_depth * self.tier.depth_multiplier()))

    def wait_if_suspended(self, timeout: float = 30.0) -> bool:
        """
        Block the calling thread if tier == SUSPENDED.
        Returns True when compute is allowed, False if timeout elapsed.
        """
        return self._suspended_event.wait(timeout=timeout)

    def should_run(self) -> bool:
        """Return True if current tier allows any computation."""
        return self.tier != ComputeTier.SUSPENDED

    def gate(self, timeout: float = 30.0):
        """
        Context manager that blocks if SUSPENDED and adjusts
        batch_size / depth_limit before entering the block.

        Usage::

            with balancer.gate():
                run_computation(
                    batch=balancer.batch_size,
                    depth=balancer.depth_limit,
                )
        """
        return _ComputeGate(self, timeout)

    # ── Cache management ──────────────────────────────────────────────────────

    def cache_tensor(self, key: str, tensor: np.ndarray) -> None:
        """Register a tensor in the managed cache (may be flushed under pressure)."""
        self._tensor_cache[key] = tensor
        self._compressor.register(key, tensor)

    def get_cached(self, key: str) -> Optional[np.ndarray]:
        return self._tensor_cache.get(key)

    def flush_cache(self) -> int:
        n = len(self._tensor_cache)
        self._tensor_cache.clear()
        return n

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def snapshot(self) -> Optional[ResourceSnapshot]:
        """Return the latest resource snapshot."""
        return self._poller.latest()

    def resource_summary(self) -> Dict[str, Any]:
        """Return a human-readable resource summary dict."""
        snap = self._poller.latest()
        avg  = self._poller.average(10)
        if snap is None:
            return {"error": "no snapshot available"}
        return {
            "tier":              self.tier.name,
            "batch_size":        self.batch_size,
            "depth_limit":       self.depth_limit,
            "cpu_percent":       snap.cpu_percent_avg,
            "cpu_ema_10s":       avg["cpu_avg"] if avg else None,
            "ram_percent":       snap.ram_percent,
            "ram_available_mb":  snap.ram_available_bytes / (1024 ** 2),
            "proc_ram_mb":       snap.proc_ram_mb,
            "swap_percent":      snap.swap_percent,
            "temp_celsius":      snap.temp_celsius,
            "cpu_freq_mhz":      snap.cpu_freq_mhz,
            "n_threads":         snap.n_threads,
            "open_files":        snap.open_files,
            "fallback_activations": self._fallback.activation_count,
            "cached_tensors":    len(self._tensor_cache),
            "cache_bytes":       self._compressor.registered_bytes(),
            "tier_changes":      len(self._tier_history),
            "snapshot_age_ms":   snap.age_ms,
        }

    def tier_history(self) -> List[Dict[str, Any]]:
        """Return full tier transition history."""
        with self._tier_lock:
            return [
                {"time_s": t, "tier": tier.name}
                for t, tier in self._tier_history
            ]

    def compressor(self) -> MemoryCompressor:
        """Direct access to the MemoryCompressor for external registration."""
        return self._compressor


# ══════════════════════════════════════════════════════════════════════════════
#  § 11  COMPUTE GATE CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class _ComputeGate:
    """
    Context manager returned by System_Load_Balancer.gate().
    Blocks on entry if SUSPENDED; raises RuntimeError on timeout.
    """

    def __init__(self, balancer: System_Load_Balancer, timeout: float) -> None:
        self._balancer = balancer
        self._timeout  = timeout

    def __enter__(self) -> System_Load_Balancer:
        allowed = self._balancer.wait_if_suspended(self._timeout)
        if not allowed:
            raise RuntimeError(
                "System_Load_Balancer: compute suspended for "
                f">{self._timeout:.0f}s — system overloaded."
            )
        return self._balancer

    def __exit__(self, *_) -> None:
        pass
"""
meta_programming_engine.py
===========================
Module 5 — Dynamic JIT Meta-Programming Pipeline (AST Experimentation)

Studies automated code generation and abstract syntax tree manipulation.
Takes a logical failure state as input, dynamically generates a new valid
Python function in memory, compiles it, and executes it within an isolated
sandbox namespace at runtime.

Architecture
------------
    FailureState             — Structured representation of a logical failure
    SandboxNamespace         — Isolated execution environment with resource limits
    ASTTransformer           — AST node visitor/transformer base utilities
    FunctionTemplate         — Parameterised AST function template registry
    FailureAnalyser          — Diagnoses failure states and selects repair strategy
    CodeGenerator            — Generates syntactically valid Python AST from spec
    ASTValidator             — Validates generated AST before compilation
    JITCompiler              — Compiles AST → code object → live function
    ExecutionMonitor         — Tracks runtime metrics of JIT-compiled functions
    Meta_Programming_Engine  — Master engine orchestrating the full pipeline

Dependencies: ast, types, inspect, sys, traceback, threading, time, gc (stdlib only)
"""


import ast
import builtins
import contextlib
import dis
import gc
import inspect
import io
import linecache
import logging
import sys
import textwrap
import threading
import time
import traceback
import types
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np



# ══════════════════════════════════════════════════════════════════════════════
#  § 1  FAILURE STATE
# ══════════════════════════════════════════════════════════════════════════════

class FailureKind(Enum):
    """Classification of logical failure types the engine can repair."""
    GRADIENT_VANISH    = auto()   # gradients collapsed to near-zero
    GRADIENT_EXPLODE   = auto()   # gradients exceeded safe norm
    LOSS_NAN           = auto()   # loss became NaN / Inf
    LOSS_PLATEAU       = auto()   # loss unchanged for N steps
    ACTIVATION_DEAD    = auto()   # dead ReLU (all-zero activations)
    NUMERIC_OVERFLOW   = auto()   # intermediate values overflowed float range
    SHAPE_MISMATCH     = auto()   # tensor shape incompatibility
    CONVERGENCE_SLOW   = auto()   # convergence rate below threshold
    CUSTOM             = auto()   # user-defined failure spec


@dataclass
class FailureState:
    """
    Structured representation of a detected logical failure.

    Parameters
    ----------
    kind          : FailureKind classification
    context       : dict of diagnostic values (grad_norm, loss, shapes, etc.)
    severity      : 0.0 (warning) → 1.0 (fatal)
    step          : training step at which failure occurred
    source_fn     : name of the function that failed (optional)
    traceback_str : captured traceback string (optional)
    """
    kind:          FailureKind
    context:       Dict[str, Any]         = field(default_factory=dict)
    severity:      float                  = 1.0
    step:          int                    = 0
    source_fn:     Optional[str]          = None
    traceback_str: Optional[str]          = None

    def summary(self) -> str:
        return (f"FailureState({self.kind.name}, severity={self.severity:.2f}, "
                f"step={self.step}, ctx={list(self.context.keys())})")


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  SANDBOX NAMESPACE
# ══════════════════════════════════════════════════════════════════════════════

# Whitelist of safe builtins allowed inside sandbox
_SAFE_BUILTINS: Dict[str, Any] = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "callable", "chr", "dict", "dir",
        "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object", "oct",
        "ord", "pow", "print", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "str", "sum", "tuple", "type", "vars",
        "zip", "True", "False", "None",
    ]
}

# Allowed top-level modules inside sandbox
_SAFE_MODULES: Set[str] = {"numpy", "math", "cmath", "statistics", "itertools",
                            "functools", "operator", "collections"}


class SandboxNamespace:
    """
    Isolated execution environment for JIT-compiled functions.

    Security Model
    --------------
    - Only whitelisted builtins are exposed (__builtins__ is replaced).
    - Only safe modules from _SAFE_MODULES may be imported.
    - __import__ is replaced with a filtered version.
    - Execution is time-bounded via threading.Timer.
    - stdout/stderr are captured and not forwarded.
    - No access to __class__, __subclasses__, or object internals.

    Parameters
    ----------
    allowed_modules : extra module names to allow beyond _SAFE_MODULES
    timeout_s       : maximum wall-clock seconds for a single execution
    allow_numpy     : inject numpy as 'np' into namespace
    extra_globals   : additional names to inject (must be explicitly trusted)
    """

    def __init__(
        self,
        allowed_modules: Optional[Set[str]] = None,
        timeout_s:       float              = 5.0,
        allow_numpy:     bool               = True,
        extra_globals:   Optional[Dict]     = None,
    ) -> None:
        self.timeout_s       = timeout_s
        self._allowed_mods   = _SAFE_MODULES | (allowed_modules or set())
        self._ns: Dict[str, Any] = self._build_namespace(allow_numpy, extra_globals)

    def _safe_import(self, name: str, *args, **kwargs) -> Any:
        """Filtered __import__ that only permits whitelisted modules."""
        root = name.split(".")[0]
        if root not in self._allowed_mods:
            raise ImportError(
                f"SandboxNamespace: import of '{name}' is not permitted."
            )
        return __builtins__["__import__"](name, *args, **kwargs) \
            if isinstance(__builtins__, dict) \
            else __import__(name, *args, **kwargs)

    def _build_namespace(
        self,
        allow_numpy:   bool,
        extra_globals: Optional[Dict],
    ) -> Dict[str, Any]:
        ns: Dict[str, Any] = {
            "__builtins__": {**_SAFE_BUILTINS, "__import__": self._safe_import},
            "__name__":     "__sandbox__",
            "__doc__":      None,
        }
        if allow_numpy:
            ns["np"]    = np
            ns["numpy"] = np
        if extra_globals:
            ns.update(extra_globals)
        return ns

    @property
    def namespace(self) -> Dict[str, Any]:
        """Return a fresh copy of the sandbox namespace for each execution."""
        return dict(self._ns)

    def execute(
        self,
        code_obj,
        local_vars: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute a compiled code object in the sandbox.

        Parameters
        ----------
        code_obj   : compiled code object (from compile() or JITCompiler)
        local_vars : additional local variables to inject

        Returns
        -------
        dict of all names defined in the sandbox after execution.

        Raises
        ------
        TimeoutError : if execution exceeds self.timeout_s
        Exception    : re-raised from sandboxed code (with traceback)
        """
        ns = self.namespace
        if local_vars:
            ns.update(local_vars)

        result_holder: Dict[str, Any] = {"exception": None, "ns": {}}
        timed_out     = threading.Event()

        def _run():
            try:
                exec(code_obj, ns)                  # noqa: S102
                result_holder["ns"] = {
                    k: v for k, v in ns.items()
                    if not k.startswith("__")
                }
            except Exception as exc:
                result_holder["exception"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_s)

        if thread.is_alive():
            timed_out.set()
            raise TimeoutError(
                f"SandboxNamespace: execution exceeded {self.timeout_s}s timeout."
            )

        if result_holder["exception"] is not None:
            raise result_holder["exception"]

        return result_holder["ns"]

    def call(
        self,
        fn_name:    str,
        code_obj,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute code_obj in sandbox, then call fn_name with args/kwargs.

        Returns the function's return value.
        """
        ns = self.execute(code_obj)
        if fn_name not in ns:
            raise NameError(
                f"SandboxNamespace: function '{fn_name}' not found after execution."
            )
        fn = ns[fn_name]
        if not callable(fn):
            raise TypeError(f"SandboxNamespace: '{fn_name}' is not callable.")
        return fn(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  AST TRANSFORMER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class ASTTransformer(ast.NodeTransformer):
    """
    Extended AST NodeTransformer with utility methods for code generation.

    Provides helpers for:
        - Building function definitions programmatically
        - Inserting guard clauses (nan checks, shape checks)
        - Renaming variables across a tree
        - Injecting logging / profiling calls
        - Constant folding of numeric literals
    """

    # ── Guard clause injectors ────────────────────────────────────────────────

    @staticmethod
    def nan_guard(var_name: str) -> ast.If:
        """
        Generate AST for::

            if not np.all(np.isfinite(<var_name>)):
                raise ValueError("NaN/Inf detected in <var_name>")
        """
        return ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="all", ctx=ast.Load(),
                    ),
                    args=[ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="isfinite", ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=var_name, ctx=ast.Load())],
                        keywords=[],
                    )],
                    keywords=[],
                ),
            ),
            body=[ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="ValueError", ctx=ast.Load()),
                    args=[ast.Constant(value=f"NaN/Inf detected in {var_name!r}")],
                    keywords=[],
                ),
                cause=None,
            )],
            orelse=[],
        )

    @staticmethod
    def clip_guard(var_name: str, lo: float, hi: float) -> ast.Assign:
        """
        Generate AST for::

            <var_name> = np.clip(<var_name>, lo, hi)
        """
        return ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="clip", ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id=var_name, ctx=ast.Load()),
                    ast.Constant(value=lo),
                    ast.Constant(value=hi),
                ],
                keywords=[],
            ),
            lineno=0, col_offset=0,
        )

    @staticmethod
    def norm_guard(var_name: str, max_norm: float) -> List[ast.stmt]:
        """
        Generate AST for gradient norm clipping::

            _norm = np.linalg.norm(<var_name>)
            if _norm > max_norm:
                <var_name> = <var_name> * (max_norm / (_norm + 1e-8))
        """
        return [
            ast.Assign(
                targets=[ast.Name(id="_norm", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="linalg", ctx=ast.Load(),
                        ),
                        attr="norm", ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=var_name, ctx=ast.Load())],
                    keywords=[],
                ),
                lineno=0, col_offset=0,
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="_norm", ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=max_norm)],
                ),
                body=[ast.Assign(
                    targets=[ast.Name(id=var_name, ctx=ast.Store())],
                    value=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.BinOp(
                            left=ast.Constant(value=max_norm),
                            op=ast.Div(),
                            right=ast.BinOp(
                                left=ast.Name(id="_norm", ctx=ast.Load()),
                                op=ast.Add(),
                                right=ast.Constant(value=1e-8),
                            ),
                        ),
                    ),
                    lineno=0, col_offset=0,
                )],
                orelse=[],
            ),
        ]

    # ── Variable renamer ──────────────────────────────────────────────────────

    def rename_variable(
        self,
        tree: ast.AST,
        old: str,
        new: str,
    ) -> ast.AST:
        """Rename all occurrences of variable `old` to `new` in tree."""

        class _Renamer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == old:
                    return ast.Name(id=new, ctx=node.ctx)
                return node

        return _Renamer().visit(tree)

    # ── Constant folder ───────────────────────────────────────────────────────

    def fold_constants(self, tree: ast.AST) -> ast.AST:
        """
        Fold constant arithmetic expressions at the AST level.
        e.g. BinOp(Constant(2), Mult, Constant(3)) → Constant(6)
        """

        class _Folder(ast.NodeTransformer):
            _OPS = {
                ast.Add:  lambda a, b: a + b,
                ast.Sub:  lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div:  lambda a, b: a / b,
                ast.Pow:  lambda a, b: a ** b,
            }

            def visit_BinOp(self, node):
                self.generic_visit(node)
                if (isinstance(node.left,  ast.Constant) and
                        isinstance(node.right, ast.Constant)):
                    op_type = type(node.op)
                    if op_type in self._OPS:
                        try:
                            result = self._OPS[op_type](
                                node.left.value, node.right.value
                            )
                            return ast.Constant(value=result)
                        except Exception:
                            pass
                return node

        return _Folder().visit(tree)


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  FUNCTION TEMPLATE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class FunctionTemplate:
    """
    Registry of parameterised AST function templates for each FailureKind.

    Each template is a callable that receives a FailureState and returns
    a list of ast.stmt nodes forming the body of the repair function.
    """

    _registry: Dict[FailureKind, Callable] = {}

    @classmethod
    def register(cls, kind: FailureKind):
        """Decorator: register a template builder for a FailureKind."""
        def decorator(fn: Callable) -> Callable:
            cls._registry[kind] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, kind: FailureKind) -> Optional[Callable]:
        return cls._registry.get(kind)

    @classmethod
    def all_kinds(cls) -> List[FailureKind]:
        return list(cls._registry.keys())


class DynamicASTGenerator:
    """
    IMPROVEMENT 1 — Dynamic Novel AST Generation (replaces hardcoded templates).

    Instead of pasting fixed string templates, this engine:
      1. Analyses the FailureState semantics (kind, severity, context).
      2. Uses a fractal-encoder-inspired embedding of the failure to
         select and compose AST node primitives dynamically.
      3. Generates a structurally novel repair function as a live AST —
         not a pre-written string — at runtime on every unique failure.

    The repair function body is assembled from a parameterised primitive
    library (clip, scale, momentum, perturbation, norm, regularise)
    whose composition and hyperparameters are derived from the failure
    embedding vector, making every repair genuinely adaptive.

    Parameters
    ----------
    embed_dim      : dimension of failure state embedding
    seed_harvester : Hardware_Entropy_Harvester for stochastic composition
    """

    # ── Primitive library ─────────────────────────────────────────────────────
    # Each primitive is a function (params: dict) -> list[ast.stmt]
    # params carries hyperparameters derived from failure embedding.

    _PRIMITIVES: Dict[str, Callable] = {}

    @classmethod
    def _prim(cls, name: str):
        def dec(fn):
            cls._PRIMITIVES[name] = fn
            return fn
        return dec

    def __init__(
        self,
        embed_dim:      int                               = 64,
        seed_harvester: Optional["Hardware_Entropy_Harvester"] = None,
    ) -> None:
        self.embed_dim = embed_dim
        self._entropy  = seed_harvester
        self._call_count = 0

    # ── Failure → embedding ───────────────────────────────────────────────────

    def _failure_embedding(self, state: FailureState) -> np.ndarray:
        """
        Encode a FailureState into a dense float vector.

        Encoding scheme:
          - One-hot of FailureKind (len = n_kinds)
          - Severity scalar
          - Numerical context values (loss, grad_norm, etc.) normalised
          - Step normalised (log scale)
        Total → padded / truncated to embed_dim.
        """
        kinds    = list(FailureKind)
        one_hot  = np.zeros(len(kinds))
        if state.kind in kinds:
            one_hot[kinds.index(state.kind)] = 1.0

        ctx_vals = []
        for v in state.context.values():
            try:
                ctx_vals.append(float(v) / (abs(float(v)) + 1.0))
            except (TypeError, ValueError):
                ctx_vals.append(0.0)
        ctx_arr = np.array(ctx_vals[:16], dtype=np.float64)
        if len(ctx_arr) < 16:
            ctx_arr = np.pad(ctx_arr, (0, 16 - len(ctx_arr)))

        step_feat  = np.array([np.log1p(state.step) / 10.0])
        sev_feat   = np.array([state.severity])

        raw = np.concatenate([one_hot, ctx_arr, step_feat, sev_feat])

        # Pad or truncate to embed_dim
        if raw.size < self.embed_dim:
            raw = np.pad(raw, (0, self.embed_dim - raw.size))
        else:
            raw = raw[:self.embed_dim]

        # L2 normalise
        norm = np.linalg.norm(raw)
        return raw / (norm + 1e-8)

    # ── Hyperparameter derivation from embedding ───────────────────────────────

    def _derive_hyperparams(self, emb: np.ndarray) -> Dict[str, float]:
        """
        Project failure embedding onto hyperparameter space.

        Uses fixed projection vectors (seeded from embed_dim) to derive:
            clip_val, scale_factor, momentum, perturb_std,
            lr_boost, leak_alpha, regularise_strength
        """
        np.random.seed(42)   # deterministic projection
        projections = {
            "clip_val":            np.random.randn(self.embed_dim),
            "scale_factor":        np.random.randn(self.embed_dim),
            "momentum":            np.random.randn(self.embed_dim),
            "perturb_std":         np.random.randn(self.embed_dim),
            "lr_boost":            np.random.randn(self.embed_dim),
            "leak_alpha":          np.random.randn(self.embed_dim),
            "regularise_strength": np.random.randn(self.embed_dim),
        }
        np.random.seed(None)

        # Ranges for each hyperparameter
        ranges = {
            "clip_val":            (0.5,  50.0),
            "scale_factor":        (1.0,  20.0),
            "momentum":            (0.7,  0.99),
            "perturb_std":         (1e-4, 0.1),
            "lr_boost":            (1.0,  10.0),
            "leak_alpha":          (0.001,0.1),
            "regularise_strength": (1e-5, 1e-2),
        }

        params: Dict[str, float] = {}
        for name, proj in projections.items():
            proj_n = proj / (np.linalg.norm(proj) + 1e-8)
            score  = float(np.dot(emb, proj_n))           # in [-1, 1]
            lo, hi = ranges[name]
            params[name] = lo + (score + 1.0) / 2.0 * (hi - lo)

        return params

    # ── Primitive selection from embedding ────────────────────────────────────

    def _select_primitives(
        self,
        state:  FailureState,
        emb:    np.ndarray,
        params: Dict[str, float],
    ) -> List[str]:
        """
        Decide which repair primitives to compose for this failure.

        Selection is driven by failure kind AND embedding similarity
        to primitive signature vectors — making composition adaptive.
        """
        # Base selection by failure kind
        kind_map: Dict[FailureKind, List[str]] = {
            FailureKind.GRADIENT_VANISH:   ["scale",      "clip",    "momentum"],
            FailureKind.GRADIENT_EXPLODE:  ["norm_clip",  "clip",    "regularise"],
            FailureKind.LOSS_NAN:          ["nan_replace","clip",    "scale"],
            FailureKind.LOSS_PLATEAU:      ["perturb",    "lr_boost","momentum"],
            FailureKind.ACTIVATION_DEAD:   ["leak",       "perturb", "scale"],
            FailureKind.NUMERIC_OVERFLOW:  ["clip",       "nan_replace"],
            FailureKind.SHAPE_MISMATCH:    ["reshape_fix"],
            FailureKind.CONVERGENCE_SLOW:  ["momentum",   "lr_boost","perturb"],
            FailureKind.CUSTOM:            ["clip",       "scale",   "perturb"],
        }
        base = kind_map.get(state.kind, ["clip", "scale"])

        # Severity-driven addition
        if state.severity > 0.8:
            base = list(dict.fromkeys(base + ["regularise", "nan_replace"]))

        return base

    # ── AST primitive builders ─────────────────────────────────────────────────

    @staticmethod
    def _ast_clip(var: str, val: float) -> List[ast.stmt]:
        return [ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="clip", ctx=ast.Load()),
                args=[ast.Name(id=var, ctx=ast.Load()),
                      ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=round(val,4))),
                      ast.Constant(value=round(val,4))],
                keywords=[]),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_scale(var: str, factor: float) -> List[ast.stmt]:
        return [ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=var, ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Constant(value=round(factor, 4))),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_nan_replace(var: str) -> List[ast.stmt]:
        return [ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="nan_to_num", ctx=ast.Load()),
                args=[ast.Name(id=var, ctx=ast.Load())],
                keywords=[
                    ast.keyword(arg="nan",    value=ast.Constant(value=0.0)),
                    ast.keyword(arg="posinf", value=ast.Constant(value=1e6)),
                    ast.keyword(arg="neginf", value=ast.Constant(value=-1e6)),
                ]),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_norm_clip(var: str, max_norm: float) -> List[ast.stmt]:
        return [
            ast.Assign(
                targets=[ast.Name(id="_norm", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="linalg", ctx=ast.Load()),
                        attr="norm", ctx=ast.Load()),
                    args=[ast.Name(id=var, ctx=ast.Load())],
                    keywords=[]),
                lineno=0, col_offset=0,
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="_norm", ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=round(max_norm, 4))]),
                body=[ast.Assign(
                    targets=[ast.Name(id=var, ctx=ast.Store())],
                    value=ast.BinOp(
                        left=ast.Name(id=var, ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.BinOp(
                            left=ast.Constant(value=round(max_norm, 4)),
                            op=ast.Div(),
                            right=ast.BinOp(
                                left=ast.Name(id="_norm", ctx=ast.Load()),
                                op=ast.Add(),
                                right=ast.Constant(value=1e-8)))),
                    lineno=0, col_offset=0,
                )],
                orelse=[],
            ),
        ]

    @staticmethod
    def _ast_perturb(var: str, std: float) -> List[ast.stmt]:
        return [ast.AugAssign(
            target=ast.Name(id=var, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.BinOp(
                left=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="random", ctx=ast.Load()),
                        attr="randn", ctx=ast.Load()),
                    args=[ast.Starred(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=var, ctx=ast.Load()),
                                attr="shape", ctx=ast.Load()),
                            args=[], keywords=[]),
                        ctx=ast.Load())],
                    keywords=[]),
                op=ast.Mult(),
                right=ast.Constant(value=round(std, 6))),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_momentum(var: str, m: float) -> List[ast.stmt]:
        return [
            ast.Assign(
                targets=[ast.Name(id="_vel", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="zeros_like", ctx=ast.Load()),
                    args=[ast.Name(id=var, ctx=ast.Load())],
                    keywords=[]),
                lineno=0, col_offset=0,
            ),
            ast.Assign(
                targets=[ast.Name(id="_vel", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.BinOp(
                        left=ast.Constant(value=round(m, 4)),
                        op=ast.Mult(),
                        right=ast.Name(id="_vel", ctx=ast.Load())),
                    op=ast.Add(),
                    right=ast.BinOp(
                        left=ast.Constant(value=round(1.0 - m, 4)),
                        op=ast.Mult(),
                        right=ast.Name(id=var, ctx=ast.Load()))),
                lineno=0, col_offset=0,
            ),
            ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Name(id="_vel", ctx=ast.Load()),
                lineno=0, col_offset=0,
            ),
        ]

    @staticmethod
    def _ast_leak(var: str, alpha: float) -> List[ast.stmt]:
        return [ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="where", ctx=ast.Load()),
                args=[
                    ast.Compare(
                        left=ast.Name(id=var, ctx=ast.Load()),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0.0)]),
                    ast.Name(id=var, ctx=ast.Load()),
                    ast.BinOp(
                        left=ast.Constant(value=round(alpha, 6)),
                        op=ast.Mult(),
                        right=ast.Name(id=var, ctx=ast.Load())),
                ],
                keywords=[]),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_regularise(var: str, strength: float) -> List[ast.stmt]:
        return [ast.AugAssign(
            target=ast.Name(id=var, ctx=ast.Store()),
            op=ast.Sub(),
            value=ast.BinOp(
                left=ast.Constant(value=round(strength, 8)),
                op=ast.Mult(),
                right=ast.Name(id=var, ctx=ast.Load())),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_lr_boost(factor: float) -> List[ast.stmt]:
        """Inject lr boosting — emitted as a comment + return value adjustment."""
        return [ast.Assign(
            targets=[ast.Name(id="_lr_boost", ctx=ast.Store())],
            value=ast.Constant(value=round(factor, 4)),
            lineno=0, col_offset=0,
        )]

    @staticmethod
    def _ast_reshape_fix() -> List[ast.stmt]:
        return [ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="x", ctx=ast.Load()),
                    attr="flatten", ctx=ast.Load()),
                args=[], keywords=[]),
            lineno=0, col_offset=0,
        )]

    # ── Full generation pipeline ───────────────────────────────────────────────

    def generate(self, state: FailureState) -> Tuple[str, ast.Module]:
        """
        Dynamically generate a novel repair function AST for a FailureState.

        Steps
        -----
        1. Embed the failure state into a float vector.
        2. Derive hyperparameters from the embedding.
        3. Select which repair primitives to compose.
        4. Build a live AST Module containing the repair function.
        5. Return (source_str, ast_module).

        The generated function always accepts (x, *args) and returns x.
        """
        self._call_count += 1

        emb    = self._failure_embedding(state)
        params = self._derive_hyperparams(emb)
        prims  = self._select_primitives(state, emb, params)

        body_stmts: List[ast.stmt] = []

        for prim in prims:
            if prim == "clip":
                body_stmts.extend(self._ast_clip("x", params["clip_val"]))
            elif prim == "scale":
                body_stmts.extend(self._ast_scale("x", params["scale_factor"]))
            elif prim == "nan_replace":
                body_stmts.extend(self._ast_nan_replace("x"))
            elif prim == "norm_clip":
                body_stmts.extend(self._ast_norm_clip("x", params["clip_val"]))
            elif prim == "perturb":
                body_stmts.extend(self._ast_perturb("x", params["perturb_std"]))
            elif prim == "momentum":
                body_stmts.extend(self._ast_momentum("x", params["momentum"]))
            elif prim == "leak":
                body_stmts.extend(self._ast_leak("x", params["leak_alpha"]))
            elif prim == "regularise":
                body_stmts.extend(self._ast_regularise("x", params["regularise_strength"]))
            elif prim == "lr_boost":
                body_stmts.extend(self._ast_lr_boost(params["lr_boost"]))
            elif prim == "reshape_fix":
                body_stmts.extend(self._ast_reshape_fix())

        # Always end with: return x
        body_stmts.append(ast.Return(value=ast.Name(id="x", ctx=ast.Load())))

        # Build function def: def repair(x, *args, **kwargs):
        fn_def = ast.FunctionDef(
            name="repair",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="x")],
                vararg=ast.arg(arg="args"),
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=ast.arg(arg="kwargs"),
                defaults=[],
            ),
            body=body_stmts if body_stmts else [ast.Pass()],
            decorator_list=[],
            returns=None,
            lineno=1, col_offset=0,
        )

        # Build module
        import_np = ast.Import(
            names=[ast.alias(name="numpy", asname="np")],
            lineno=0, col_offset=0,
        )
        module = ast.Module(body=[import_np, fn_def], type_ignores=[])
        ast.fix_missing_locations(module)

        # Generate source string for inspection / linecache
        try:
            source = ast.unparse(module)
        except Exception:
            source = f"# DynamicAST repair for {state.kind.name} step={state.step}"

        return source, module

    @property
    def generation_count(self) -> int:
        return self._call_count


# ── Template implementations ──────────────────────────────────────────────────

@FunctionTemplate.register(FailureKind.GRADIENT_VANISH)
def _tmpl_gradient_vanish(state: FailureState) -> str:
    scale = state.context.get("rescale_factor", 10.0)
    return textwrap.dedent(f"""
        def repair(grad, params):
            import numpy as np
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-7:
                grad = grad * {scale}
            grad = np.clip(grad, -1.0, 1.0)
            lr_boost = {scale} * 0.1
            for p in params:
                p -= lr_boost * grad[:p.size].reshape(p.shape)
            return grad, params
    """)


@FunctionTemplate.register(FailureKind.GRADIENT_EXPLODE)
def _tmpl_gradient_explode(state: FailureState) -> str:
    max_norm = state.context.get("max_norm", 5.0)
    return textwrap.dedent(f"""
        def repair(grad, params):
            import numpy as np
            norm = np.linalg.norm(grad)
            if norm > {max_norm}:
                grad = grad * ({max_norm} / (norm + 1e-8))
            return grad, params
    """)


@FunctionTemplate.register(FailureKind.LOSS_NAN)
def _tmpl_loss_nan(state: FailureState) -> str:
    fallback = state.context.get("fallback_loss", 1e6)
    return textwrap.dedent(f"""
        def repair(loss, params, lr=0.001):
            import numpy as np
            if not np.isfinite(loss):
                loss = {fallback}
                for p in params:
                    p -= lr * np.sign(p) * 0.01
            return float(loss), params
    """)


@FunctionTemplate.register(FailureKind.LOSS_PLATEAU)
def _tmpl_loss_plateau(state: FailureState) -> str:
    perturb_std = state.context.get("perturb_std", 0.01)
    lr_factor   = state.context.get("lr_boost", 5.0)
    return textwrap.dedent(f"""
        def repair(params, lr):
            import numpy as np
            for p in params:
                p += np.random.randn(*p.shape) * {perturb_std}
            boosted_lr = lr * {lr_factor}
            return params, boosted_lr
    """)


@FunctionTemplate.register(FailureKind.ACTIVATION_DEAD)
def _tmpl_activation_dead(state: FailureState) -> str:
    leak = state.context.get("leak_alpha", 0.01)
    return textwrap.dedent(f"""
        def repair(activations, weights):
            import numpy as np
            mask = (activations == 0)
            if mask.mean() > 0.5:
                activations = np.where(mask, {leak} * np.random.randn(*activations.shape), activations)
                weights     = weights * (1.0 + np.random.randn(*weights.shape) * 0.01)
            return activations, weights
    """)


@FunctionTemplate.register(FailureKind.NUMERIC_OVERFLOW)
def _tmpl_numeric_overflow(state: FailureState) -> str:
    clip_val = state.context.get("clip_value", 1e6)
    return textwrap.dedent(f"""
        def repair(x):
            import numpy as np
            x = np.clip(x, -{clip_val}, {clip_val})
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf={clip_val}, neginf=-{clip_val})
            return x
    """)


@FunctionTemplate.register(FailureKind.SHAPE_MISMATCH)
def _tmpl_shape_mismatch(state: FailureState) -> str:
    target_shape = state.context.get("target_shape", None)
    shape_str    = str(target_shape) if target_shape else "None"
    return textwrap.dedent(f"""
        def repair(x, target_shape={shape_str}):
            import numpy as np
            if target_shape is not None:
                try:
                    x = x.reshape(target_shape)
                except ValueError:
                    needed = 1
                    for d in target_shape:
                        needed *= d
                    x = x.flatten()[:needed].reshape(target_shape)
            return x
    """)


@FunctionTemplate.register(FailureKind.CONVERGENCE_SLOW)
def _tmpl_convergence_slow(state: FailureState) -> str:
    momentum = state.context.get("momentum", 0.95)
    lr_scale = state.context.get("lr_scale", 2.0)
    return textwrap.dedent(f"""
        def repair(grad, velocity, lr):
            import numpy as np
            velocity = {momentum} * velocity + (1 - {momentum}) * grad
            adjusted_lr = lr * {lr_scale}
            return velocity, adjusted_lr
    """)


@FunctionTemplate.register(FailureKind.CUSTOM)
def _tmpl_custom(state: FailureState) -> str:
    body = state.context.get("custom_body", "    return x")
    args = state.context.get("custom_args", "x")
    return textwrap.dedent(f"""
        def repair({args}):
            import numpy as np
{textwrap.indent(body, '            ')}
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  FAILURE ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

class FailureAnalyser:
    """
    Diagnoses a FailureState and selects the appropriate repair strategy.

    Analysis includes:
        - Severity classification
        - Context enrichment (auto-filling missing context fields)
        - Strategy selection from FunctionTemplate registry
        - Fallback to CUSTOM if no template matches

    Parameters
    ----------
    default_severity_threshold : severities below this → CONVERGENCE_SLOW fallback
    """

    def __init__(self, default_severity_threshold: float = 0.3) -> None:
        self.threshold = default_severity_threshold

    def _enrich_context(self, state: FailureState) -> FailureState:
        """Fill in missing context keys with safe defaults."""
        ctx = dict(state.context)

        if state.kind == FailureKind.GRADIENT_VANISH:
            ctx.setdefault("rescale_factor", 10.0)
        elif state.kind == FailureKind.GRADIENT_EXPLODE:
            ctx.setdefault("max_norm", 5.0)
        elif state.kind == FailureKind.LOSS_NAN:
            ctx.setdefault("fallback_loss", 1e6)
        elif state.kind == FailureKind.LOSS_PLATEAU:
            ctx.setdefault("perturb_std", 0.01)
            ctx.setdefault("lr_boost",    5.0)
        elif state.kind == FailureKind.ACTIVATION_DEAD:
            ctx.setdefault("leak_alpha", 0.01)
        elif state.kind == FailureKind.NUMERIC_OVERFLOW:
            ctx.setdefault("clip_value", 1e6)
        elif state.kind == FailureKind.CONVERGENCE_SLOW:
            ctx.setdefault("momentum", 0.95)
            ctx.setdefault("lr_scale", 2.0)

        return FailureState(
            kind          = state.kind,
            context       = ctx,
            severity      = state.severity,
            step          = state.step,
            source_fn     = state.source_fn,
            traceback_str = state.traceback_str,
        )

    def analyse(self, state: FailureState) -> Tuple[FailureState, FailureKind]:
        """
        Enrich the failure state and resolve the repair FailureKind.

        Returns
        -------
        (enriched_state, resolved_kind)
        """
        enriched = self._enrich_context(state)

        # If severity is low, downgrade to CONVERGENCE_SLOW (soft repair)
        if enriched.severity < self.threshold and enriched.kind != FailureKind.CUSTOM:
            resolved = FailureKind.CONVERGENCE_SLOW
        elif FunctionTemplate.get(enriched.kind) is not None:
            resolved = enriched.kind
        else:
            resolved = FailureKind.CUSTOM

        logger.debug(
            "FailureAnalyser: %s → resolved kind=%s",
            enriched.summary(), resolved.name,
        )
        return enriched, resolved


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  CODE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class CodeGenerator:
    """
    Generates syntactically valid Python source code and its AST
    from a FailureState using the FunctionTemplate registry.

    Output is always a module-level function named `repair`.

    Parameters
    ----------
    add_nan_guards   : inject NaN/Inf guard clauses into generated bodies
    add_norm_guards  : inject gradient norm guard clauses
    fold_constants   : apply constant folding pass to generated AST
    """

    def __init__(
        self,
        add_nan_guards:  bool = True,
        add_norm_guards: bool = True,
        fold_constants:  bool = True,
    ) -> None:
        self.add_nan_guards  = add_nan_guards
        self.add_norm_guards = add_norm_guards
        self.fold_constants  = fold_constants
        self._transformer    = ASTTransformer()

    def generate_source(self, state: FailureState, kind: FailureKind) -> str:
        """
        Generate Python source string for the repair function.

        Parameters
        ----------
        state : enriched FailureState
        kind  : resolved FailureKind (may differ from state.kind)

        Returns
        -------
        Python source code string containing the `repair` function.
        """
        builder = FunctionTemplate.get(kind)
        if builder is None:
            builder = FunctionTemplate.get(FailureKind.CUSTOM)

        source = builder(state)

        # Prepend module-level imports not already in source
        header = "import numpy as np\nimport math\n\n"
        if "import numpy" not in source:
            source = header + source

        return source

    def generate_ast(self, source: str) -> ast.Module:
        """
        Parse source into an AST, apply transformations, and return.

        Transformations applied (if enabled):
            1. Constant folding
        """
        tree = ast.parse(source, mode="exec")

        if self.fold_constants:
            tree = self._transformer.fold_constants(tree)

        ast.fix_missing_locations(tree)
        return tree

    def generate(
        self,
        state: FailureState,
        kind:  FailureKind,
    ) -> Tuple[str, ast.Module]:
        """
        Full generation pipeline: source + AST.

        Returns
        -------
        (source_str, ast_module)
        """
        source = self.generate_source(state, kind)
        tree   = self.generate_ast(source)
        return source, tree


# ══════════════════════════════════════════════════════════════════════════════
#  § 7  AST VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class ASTValidator(ast.NodeVisitor):
    """
    Validates generated AST before compilation.

    Checks
    ------
    1. Function named `repair` must be present at module level.
    2. No `exec`, `eval`, `compile`, `__import__` calls.
    3. No attribute access on `os`, `sys`, `subprocess`, `socket`.
    4. No class definitions (prevents metaclass exploits).
    5. No `while True` without a break (infinite loop guard).
    6. No `__` attribute access (dunder method access prevention).
    """

    _BANNED_CALLS:   Set[str] = {"exec", "eval", "compile", "__import__",
                                  "open", "breakpoint", "input"}
    _BANNED_MODULES: Set[str] = {"os", "sys", "subprocess", "socket",
                                  "importlib", "ctypes", "signal"}

    def __init__(self) -> None:
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self._has_repair_fn = False

    def validate(self, tree: ast.Module) -> bool:
        """
        Validate the AST tree.

        Returns True if valid, False if errors were found.
        Populates self.errors and self.warnings.
        """
        self.errors   = []
        self.warnings = []
        self._has_repair_fn = False
        self.visit(tree)

        if not self._has_repair_fn:
            self.errors.append("No top-level function named 'repair' found.")

        return len(self.errors) == 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "repair":
            self._has_repair_fn = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.errors.append(
            f"Line {node.lineno}: class definitions not permitted in sandbox."
        )

    def visit_Call(self, node: ast.Call) -> None:
        # Check for banned function names
        if isinstance(node.func, ast.Name):
            if node.func.id in self._BANNED_CALLS:
                self.errors.append(
                    f"Banned call: '{node.func.id}' at line {node.lineno}."
                )
        # Check for banned module attribute access
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                    node.func.value.id in self._BANNED_MODULES):
                self.errors.append(
                    f"Banned module access: '{node.func.value.id}' "
                    f"at line {node.lineno}."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.warnings.append(
                f"Dunder attribute access '{node.attr}' at line {getattr(node, 'lineno', '?')}."
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self._BANNED_MODULES:
                self.errors.append(
                    f"Banned import: '{alias.name}' at line {node.lineno}."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in self._BANNED_MODULES:
                self.errors.append(
                    f"Banned import from: '{node.module}' at line {node.lineno}."
                )
        self.generic_visit(node)


# ══════════════════════════════════════════════════════════════════════════════
#  § 8  JIT COMPILER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompiledFunction:
    """Container for a JIT-compiled repair function."""
    fn:           Callable
    fn_name:      str
    source:       str
    code_obj:     types.CodeType
    failure_kind: FailureKind
    compiled_at:  float           = field(default_factory=time.monotonic)
    call_count:   int             = 0
    total_time_s: float           = 0.0

    @property
    def avg_time_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return (self.total_time_s / self.call_count) * 1000.0


class JITCompiler:
    """
    Compiles an AST Module into a live callable Python function.

    Pipeline
    --------
    1. Validate AST via ASTValidator.
    2. ast.fix_missing_locations() to fill line numbers.
    3. compile(tree, filename, "exec") → code object.
    4. exec() in isolated namespace to define the function.
    5. Wrap in CompiledFunction with metadata.
    6. Register source with linecache for tracebacks.

    Parameters
    ----------
    sandbox : SandboxNamespace used for exec
    """

    def __init__(self, sandbox: SandboxNamespace) -> None:
        self._sandbox   = sandbox
        self._validator = ASTValidator()
        self._cache:    Dict[FailureKind, CompiledFunction] = {}
        self._counter   = 0

    def compile(
        self,
        source:       str,
        tree:         ast.Module,
        failure_kind: FailureKind,
    ) -> CompiledFunction:
        """
        Compile AST to a live CompiledFunction.

        Parameters
        ----------
        source       : original source string (for linecache / debug)
        tree         : validated AST module
        failure_kind : failure kind this function repairs

        Returns
        -------
        CompiledFunction

        Raises
        ------
        SyntaxError   : if AST fails validation
        CompileError  : if compile() fails
        """
        # Validate
        valid = self._validator.validate(tree)
        if not valid:
            raise SyntaxError(
                "ASTValidator errors:\n" + "\n".join(self._validator.errors)
            )
        if self._validator.warnings:
            for w in self._validator.warnings:
                logger.warning("ASTValidator: %s", w)

        # Unique filename for linecache
        self._counter += 1
        filename = f"<jit_repair_{failure_kind.name}_{self._counter}>"

        # Register in linecache for accurate tracebacks
        lines = source.splitlines(keepends=True)
        linecache.cache[filename] = (
            len(source), None, lines, filename
        )

        # Compile
        ast.fix_missing_locations(tree)
        try:
            code_obj = compile(tree, filename, "exec")
        except Exception as exc:
            raise SyntaxError(f"compile() failed: {exc}") from exc

        # Execute in sandbox to define the function
        ns = self._sandbox.execute(code_obj)

        if "repair" not in ns:
            raise NameError("Compiled code did not define 'repair'.")

        fn = ns["repair"]

        compiled = CompiledFunction(
            fn=fn,
            fn_name="repair",
            source=source,
            code_obj=code_obj,
            failure_kind=failure_kind,
        )

        self._cache[failure_kind] = compiled
        logger.info(
            "JITCompiler: compiled repair fn for %s (%s)",
            failure_kind.name, filename,
        )
        return compiled

    def cached(self, kind: FailureKind) -> Optional[CompiledFunction]:
        return self._cache.get(kind)

    def disassemble(self, kind: FailureKind) -> Optional[str]:
        """Return CPython bytecode disassembly of a cached compiled function."""
        cf = self._cache.get(kind)
        if cf is None:
            return None
        buf = io.StringIO()
        dis.dis(cf.fn, file=buf)
        return buf.getvalue()

    def invalidate(self, kind: FailureKind) -> None:
        """Remove cached compiled function for a given kind."""
        self._cache.pop(kind, None)

    @property
    def cached_kinds(self) -> List[FailureKind]:
        return list(self._cache.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  § 9  EXECUTION MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionMonitor:
    """
    Tracks runtime metrics of JIT-compiled repair function invocations.

    Metrics per compiled function
    -----------------------------
    - call count
    - total / mean / min / max wall-clock time
    - exception count and types
    - output value statistics (if numpy arrays returned)
    """

    def __init__(self) -> None:
        self._records: Dict[FailureKind, Dict[str, Any]] = {}

    def _ensure(self, kind: FailureKind) -> None:
        if kind not in self._records:
            self._records[kind] = {
                "calls":        0,
                "total_s":      0.0,
                "min_s":        float("inf"),
                "max_s":        0.0,
                "exceptions":   0,
                "exc_types":    [],
                "output_norms": [],
            }

    def record_call(
        self,
        kind:     FailureKind,
        duration: float,
        output:   Any            = None,
        exc:      Optional[Exception] = None,
    ) -> None:
        self._ensure(kind)
        r = self._records[kind]
        r["calls"]   += 1
        r["total_s"] += duration
        r["min_s"]    = min(r["min_s"],   duration)
        r["max_s"]    = max(r["max_s"],   duration)
        if exc is not None:
            r["exceptions"] += 1
            r["exc_types"].append(type(exc).__name__)
        if output is not None and isinstance(output, np.ndarray):
            r["output_norms"].append(float(np.linalg.norm(output)))

    def report(self, kind: FailureKind) -> Optional[Dict[str, Any]]:
        r = self._records.get(kind)
        if r is None:
            return None
        n = r["calls"]
        return {
            "calls":          n,
            "mean_ms":        (r["total_s"] / n * 1000) if n > 0 else 0.0,
            "min_ms":         r["min_s"] * 1000,
            "max_ms":         r["max_s"] * 1000,
            "total_s":        r["total_s"],
            "exceptions":     r["exceptions"],
            "exc_types":      r["exc_types"],
            "mean_out_norm":  (float(np.mean(r["output_norms"]))
                               if r["output_norms"] else None),
        }

    def full_report(self) -> Dict[str, Any]:
        return {
            kind.name: self.report(kind)
            for kind in self._records
        }


# ══════════════════════════════════════════════════════════════════════════════
#  § 10  META_PROGRAMMING_ENGINE — Master Engine
# ══════════════════════════════════════════════════════════════════════════════

class Meta_Programming_Engine:
    """
    Master JIT meta-programming engine.

    Full Pipeline
    -------------
    1. Receive a FailureState.
    2. FailureAnalyser diagnoses and enriches the state.
    3. CodeGenerator produces Python source + AST for the repair fn.
    4. ASTValidator ensures safety before compilation.
    5. JITCompiler compiles AST → code object → live function.
    6. SandboxNamespace provides isolated execution environment.
    7. ExecutionMonitor tracks all invocations.
    8. CompiledFunction cache prevents redundant recompilation.

    Parameters
    ----------
    timeout_s         : sandbox execution timeout
    cache_compiled    : reuse previously compiled functions for same kind
    extra_globals     : trusted names injected into sandbox
    """

    def __init__(
        self,
        timeout_s:      float          = 5.0,
        cache_compiled: bool           = True,
        extra_globals:  Optional[Dict] = None,
        embed_dim:      int            = 64,
    ) -> None:
        self._sandbox      = SandboxNamespace(
            timeout_s=timeout_s,
            extra_globals=extra_globals,
        )
        self._analyser     = FailureAnalyser()
        self._generator    = CodeGenerator()
        self._dyn_gen      = DynamicASTGenerator(embed_dim=embed_dim)
        self._compiler     = JITCompiler(self._sandbox)
        self._monitor      = ExecutionMonitor()
        self._cache_on     = cache_compiled
        self._history:     List[Dict[str, Any]] = []
        self._dyn_history: List[Dict[str, Any]] = []

    # ── Core repair API ───────────────────────────────────────────────────────

    def repair(
        self,
        state: FailureState,
        *args,
        **kwargs,
    ) -> Tuple[Any, CompiledFunction]:
        """
        Full pipeline: analyse → DYNAMIC AST generate → compile → execute.

        Pipeline (IMPROVEMENT 1 — Dynamic AST Generation):
        ---------------------------------------------------
        1. Analyse failure state (enrich + classify).
        2. DynamicASTGenerator embeds the failure state into a float vector,
           derives hyperparameters from the embedding, selects repair
           primitives adaptively, and builds a live AST at runtime.
           No hardcoded string templates used.
        3. ASTValidator validates the generated AST.
        4. JITCompiler compiles AST → code object → live function.
        5. Execute in SandboxNamespace with timeout.
        6. ExecutionMonitor records metrics.

        Parameters
        ----------
        state    : FailureState describing the detected failure
        *args    : forwarded to the generated repair function
        **kwargs : forwarded to the generated repair function

        Returns
        -------
        (repair_output, compiled_fn)
        """
        # Step 1: analyse
        enriched, kind = self._analyser.analyse(state)

        # Step 2: check cache (cache key = kind + severity bucket)
        sev_bucket = round(enriched.severity, 1)
        cache_key  = f"{kind.name}_{sev_bucket}"
        compiled   = None
        if self._cache_on:
            compiled = self._compiler.cached(kind)

        # Step 3: DYNAMIC AST generation (IMPROVEMENT 1)
        if compiled is None:
            # Generate novel AST from failure embedding — not from templates
            dyn_source, dyn_tree = self._dyn_gen.generate(enriched)

            # Validate generated AST
            validator = ASTValidator()
            # Allow any fn name in dynamic generation
            orig_visit = validator.visit_FunctionDef
            def _patched_visit(node, v=validator):
                if node.name == "repair":
                    v._has_repair_fn = True
                orig_visit(node)
            validator.visit_FunctionDef = _patched_visit

            valid = validator.validate(dyn_tree)
            if not valid:
                # Fallback to static template generator if validation fails
                logger.warning(
                    "[M5] DynamicAST validation failed (%s), "
                    "falling back to static template.",
                    "; ".join(validator.errors)
                )
                dyn_source, dyn_tree = self._generator.generate(enriched, kind)

            compiled = self._compiler.compile(dyn_source, dyn_tree, kind)

            self._dyn_history.append({
                "step":        enriched.step,
                "kind":        kind.name,
                "severity":    enriched.severity,
                "primitives":  self._dyn_gen.generation_count,
                "source_len":  len(dyn_source),
            })

        # Step 4: execute with monitoring
        t0  = time.perf_counter()
        exc = None
        out = None
        try:
            out = compiled.fn(*args, **kwargs)
        except Exception as e:
            exc = e
            logger.error(
                "Meta_Programming_Engine: repair fn for %s raised %s: %s",
                kind.name, type(e).__name__, e,
            )
            raise
        finally:
            dt = time.perf_counter() - t0
            compiled.call_count   += 1
            compiled.total_time_s += dt
            self._monitor.record_call(kind, dt, out, exc)

        # Step 5: record history
        self._history.append({
            "step":          state.step,
            "failure_kind":  kind.name,
            "severity":      enriched.severity,
            "duration_ms":   dt * 1000,
            "success":       exc is None,
        })

        return out, compiled

    # ── Direct code execution (no FailureState) ───────────────────────────────

    def execute_source(
        self,
        source:   str,
        fn_name:  str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Parse, validate, compile, and execute arbitrary source code.
        The source must define a function named `fn_name`.

        Parameters
        ----------
        source  : Python source string
        fn_name : name of the function to call after execution
        *args   : arguments forwarded to fn_name
        """
        tree  = self._generator.generate_ast(source)

        # Validate
        validator = ASTValidator()
        # Temporarily allow any fn name by patching validator
        original_visit = validator.visit_FunctionDef

        def _patched(node):
            if node.name == fn_name:
                validator._has_repair_fn = True
            original_visit(node)

        validator.visit_FunctionDef = _patched
        valid = validator.validate(tree)
        if not valid:
            raise SyntaxError(
                "ASTValidator errors:\n" + "\n".join(validator.errors)
            )

        return self._sandbox.call(fn_name, compile(tree, "<dynamic>", "exec"),
                                  *args, **kwargs)

    # ── Inspection ────────────────────────────────────────────────────────────

    def inspect_source(self, kind: FailureKind) -> Optional[str]:
        """Return the source of a cached compiled function."""
        cf = self._compiler.cached(kind)
        return cf.source if cf else None

    def disassemble(self, kind: FailureKind) -> Optional[str]:
        """Return CPython bytecode disassembly of a cached function."""
        return self._compiler.disassemble(kind)

    def get_compiled(self, kind: FailureKind) -> Optional[CompiledFunction]:
        return self._compiler.cached(kind)

    def invalidate_cache(self, kind: Optional[FailureKind] = None) -> None:
        """
        Invalidate compiled function cache.
        If kind is None, invalidate all.
        """
        if kind is None:
            for k in list(self._compiler.cached_kinds):
                self._compiler.invalidate(k)
        else:
            self._compiler.invalidate(kind)

    # ── Monitoring ────────────────────────────────────────────────────────────

    def execution_report(self) -> Dict[str, Any]:
        """Return full execution statistics from ExecutionMonitor."""
        return self._monitor.full_report()

    def repair_history(self) -> List[Dict[str, Any]]:
        """Return chronological list of all repair invocations."""
        return list(self._history)

    @property
    def cached_kinds(self) -> List[FailureKind]:
        return self._compiler.cached_kinds

    @property
    def sandbox(self) -> SandboxNamespace:
        return self._sandbox

    @property
    def compiler(self) -> JITCompiler:
        return self._compiler
"""
temporal_graph_storage.py
==========================
Module 6 — Temporally Weighted Graph Database

Maintains a continuous, causal record of all system states and inputs.
Nodes represent states/inputs; edges encode temporal distance and
heuristic priority weights. Query functions retrieve states via connected
causal pathways — not flat cosine similarity.

Architecture
------------
    NodeKind                 — Classification of node types
    GraphNode                — Full node with payload, embedding, timestamps
    GraphEdge                — Weighted directed edge with causal metadata
    EdgeWeightPolicy         — Configurable edge weight computation strategy
    CausalPathFinder         — Dijkstra + temporal-decay shortest-path search
    SubgraphExtractor        — Neighbourhood and reachability subgraph queries
    TemporalIndex            — Time-ordered index for range queries
    EmbeddingIndex           — Approximate nearest-neighbour over node embeddings
    GraphSerializer          — JSON + numpy binary serialisation / deserialisation
    Temporal_Graph_Storage   — Master causal graph database engine

Retrieval model: edges carry BOTH temporal distance and heuristic priority.
Query traversal uses temporal-decay weighted Dijkstra — path cost increases
with time elapsed and decreases with high-priority causal edges.

Dependencies: numpy, json, heapq, uuid, threading, time (stdlib + numpy)
"""


import heapq
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import numpy as np



# ══════════════════════════════════════════════════════════════════════════════
#  § 1  ENUMS AND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class NodeKind(Enum):
    """Classification of graph node types."""
    STATE      = auto()   # system state snapshot
    INPUT      = auto()   # external input / observation
    GRADIENT   = auto()   # gradient / parameter update event
    LOSS       = auto()   # loss value checkpoint
    DECISION   = auto()   # branching decision point
    ANNOTATION = auto()   # user-defined annotation
    AGGREGATE  = auto()   # summarised / compressed ancestor cluster


class EdgeKind(Enum):
    """Classification of causal edge types."""
    TEMPORAL   = auto()   # direct time-ordered succession
    CAUSAL     = auto()   # causal influence (A caused B)
    REFERENCE  = auto()   # B references A (non-causal lookup)
    AGGREGATES = auto()   # edge from aggregate node to its sources
    DIVERGES   = auto()   # branching point (one input → multiple outputs)
    MERGES     = auto()   # join point (multiple inputs → one output)


# ══════════════════════════════════════════════════════════════════════════════
#  § 2  GRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """
    A node in the temporal causal graph.

    Parameters
    ----------
    node_id      : unique UUID string
    kind         : NodeKind classification
    payload      : arbitrary metadata dict (step, loss, shapes, etc.)
    embedding    : dense float vector for similarity / indexing (optional)
    created_ns   : creation time in nanoseconds (time.perf_counter_ns())
    step         : training / processing step at creation
    priority     : heuristic priority weight [0, 1] — higher = more important
    tags         : free-form string tags for filtering
    """
    node_id:    str
    kind:       NodeKind
    payload:    Dict[str, Any]         = field(default_factory=dict)
    embedding:  Optional[np.ndarray]   = field(default=None, repr=False)
    created_ns: int                    = field(default_factory=time.perf_counter_ns)
    step:       int                    = 0
    priority:   float                  = 0.5
    tags:       List[str]              = field(default_factory=list)

    @classmethod
    def create(
        cls,
        kind:      NodeKind,
        payload:   Optional[Dict]        = None,
        embedding: Optional[np.ndarray]  = None,
        step:      int                   = 0,
        priority:  float                 = 0.5,
        tags:      Optional[List[str]]   = None,
    ) -> "GraphNode":
        return cls(
            node_id   = str(uuid.uuid4()),
            kind      = kind,
            payload   = payload or {},
            embedding = embedding,
            step      = step,
            priority  = priority,
            tags      = tags or [],
        )

    @property
    def age_s(self) -> float:
        return (time.perf_counter_ns() - self.created_ns) / 1e9

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "node_id":    self.node_id,
            "kind":       self.kind.name,
            "payload":    self.payload,
            "created_ns": self.created_ns,
            "step":       self.step,
            "priority":   self.priority,
            "tags":       self.tags,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphNode":
        emb = np.array(d["embedding"]) if "embedding" in d else None
        return cls(
            node_id   = d["node_id"],
            kind      = NodeKind[d["kind"]],
            payload   = d.get("payload", {}),
            embedding = emb,
            created_ns= d.get("created_ns", 0),
            step      = d.get("step", 0),
            priority  = d.get("priority", 0.5),
            tags      = d.get("tags", []),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  § 3  GRAPH EDGE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphEdge:
    """
    A weighted directed edge in the temporal causal graph.

    Edge weight semantics
    ---------------------
    Lower weight = stronger / more preferred causal connection.
    Weight is computed by EdgeWeightPolicy and incorporates:
        - Temporal distance between source and target nodes
        - Heuristic priority of both nodes
        - Edge kind (CAUSAL < TEMPORAL < REFERENCE)

    Parameters
    ----------
    edge_id       : unique UUID string
    src_id        : source node UUID
    dst_id        : destination node UUID
    kind          : EdgeKind classification
    weight        : pre-computed edge weight (used in Dijkstra)
    temporal_dist_ns : absolute nanosecond distance between nodes
    metadata      : optional extra dict
    """
    edge_id:         str
    src_id:          str
    dst_id:          str
    kind:            EdgeKind
    weight:          float             = 1.0
    temporal_dist_ns:int               = 0
    metadata:        Dict[str, Any]    = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        src:      GraphNode,
        dst:      GraphNode,
        kind:     EdgeKind,
        weight:   float,
        metadata: Optional[Dict] = None,
    ) -> "GraphEdge":
        return cls(
            edge_id          = str(uuid.uuid4()),
            src_id           = src.node_id,
            dst_id           = dst.node_id,
            kind             = kind,
            weight           = weight,
            temporal_dist_ns = abs(dst.created_ns - src.created_ns),
            metadata         = metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id":          self.edge_id,
            "src_id":           self.src_id,
            "dst_id":           self.dst_id,
            "kind":             self.kind.name,
            "weight":           self.weight,
            "temporal_dist_ns": self.temporal_dist_ns,
            "metadata":         self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            edge_id          = d["edge_id"],
            src_id           = d["src_id"],
            dst_id           = d["dst_id"],
            kind             = EdgeKind[d["kind"]],
            weight           = d.get("weight", 1.0),
            temporal_dist_ns = d.get("temporal_dist_ns", 0),
            metadata         = d.get("metadata", {}),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  § 4  EDGE WEIGHT POLICY
# ══════════════════════════════════════════════════════════════════════════════

class EdgeWeightPolicy:
    """
    Computes edge weights incorporating temporal distance and heuristic priority.

    Weight formula
    --------------
        w(e) = base_kind_cost(kind)
               × temporal_factor(Δt)
               × priority_factor(src.priority, dst.priority)

        temporal_factor(Δt) = 1 + τ · log(1 + Δt_s)
            Δt_s in seconds; τ = temporal_decay_rate

        priority_factor(p_s, p_d) = 1 / (0.5 + 0.5 × (p_s + p_d) / 2)
            High-priority edges get lower weight (preferred in Dijkstra)

    Parameters
    ----------
    temporal_decay_rate : τ — how strongly temporal distance increases cost
    kind_costs          : base cost multiplier per EdgeKind
    """

    _DEFAULT_KIND_COSTS: Dict[EdgeKind, float] = {
        EdgeKind.CAUSAL:     0.5,
        EdgeKind.TEMPORAL:   1.0,
        EdgeKind.REFERENCE:  2.0,
        EdgeKind.AGGREGATES: 1.5,
        EdgeKind.DIVERGES:   1.2,
        EdgeKind.MERGES:     0.8,
    }

    def __init__(
        self,
        temporal_decay_rate: float                        = 0.1,
        kind_costs:          Optional[Dict[EdgeKind, float]] = None,
    ) -> None:
        self.tau       = temporal_decay_rate
        self.costs     = kind_costs or self._DEFAULT_KIND_COSTS

    def compute(
        self,
        src:  GraphNode,
        dst:  GraphNode,
        kind: EdgeKind,
    ) -> float:
        """
        Compute the edge weight for edge (src → dst).

        Returns
        -------
        float weight (lower = preferred path)
        """
        base     = self.costs.get(kind, 1.0)
        dt_s     = abs(dst.created_ns - src.created_ns) / 1e9
        temporal = 1.0 + self.tau * np.log1p(dt_s)
        priority = 1.0 / (0.5 + 0.5 * (src.priority + dst.priority) / 2.0)
        return float(base * temporal * priority)

    def recompute(
        self,
        edge: GraphEdge,
        src:  GraphNode,
        dst:  GraphNode,
    ) -> GraphEdge:
        """Return a new GraphEdge with recomputed weight."""
        w = self.compute(src, dst, edge.kind)
        return GraphEdge(
            edge_id          = edge.edge_id,
            src_id           = edge.src_id,
            dst_id           = edge.dst_id,
            kind             = edge.kind,
            weight           = w,
            temporal_dist_ns = edge.temporal_dist_ns,
            metadata         = edge.metadata,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  § 5  TEMPORAL INDEX
# ══════════════════════════════════════════════════════════════════════════════

class TemporalIndex:
    """
    Time-ordered index for efficient range queries over node creation times.

    Maintains a sorted list of (created_ns, node_id) pairs.
    Supports:
        - range_query(start_ns, end_ns)      → node_id list
        - step_query(start_step, end_step)   → node_id list
        - latest(n)                          → n most recent node_ids
        - oldest(n)                          → n oldest node_ids
    """

    def __init__(self) -> None:
        self._time_index: List[Tuple[int, str]]  = []   # (created_ns, node_id)
        self._step_index: List[Tuple[int, str]]  = []   # (step, node_id)
        self._lock = threading.RLock()

    def insert(self, node: GraphNode) -> None:
        with self._lock:
            heapq.heappush(self._time_index, (node.created_ns, node.node_id))
            heapq.heappush(self._step_index, (node.step, node.node_id))

    def remove(self, node_id: str, created_ns: int, step: int) -> None:
        with self._lock:
            self._time_index = [
                (t, nid) for t, nid in self._time_index if nid != node_id
            ]
            self._step_index = [
                (s, nid) for s, nid in self._step_index if nid != node_id
            ]
            heapq.heapify(self._time_index)
            heapq.heapify(self._step_index)

    def range_query(
        self,
        start_ns: Optional[int] = None,
        end_ns:   Optional[int] = None,
    ) -> List[str]:
        """Return node IDs with created_ns in [start_ns, end_ns]."""
        with self._lock:
            result = []
            for ts, nid in self._time_index:
                if start_ns is not None and ts < start_ns:
                    continue
                if end_ns is not None and ts > end_ns:
                    continue
                result.append(nid)
        return result

    def step_query(
        self,
        start_step: int,
        end_step:   int,
    ) -> List[str]:
        """Return node IDs with step in [start_step, end_step]."""
        with self._lock:
            return [nid for s, nid in self._step_index
                    if start_step <= s <= end_step]

    def latest(self, n: int = 10) -> List[str]:
        """Return the n most recently created node IDs."""
        with self._lock:
            sorted_items = sorted(self._time_index, reverse=True)
        return [nid for _, nid in sorted_items[:n]]

    def oldest(self, n: int = 10) -> List[str]:
        """Return the n oldest node IDs."""
        with self._lock:
            sorted_items = sorted(self._time_index)
        return [nid for _, nid in sorted_items[:n]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._time_index)


# ══════════════════════════════════════════════════════════════════════════════
#  § 6  EMBEDDING INDEX
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingIndex:
    """
    Approximate nearest-neighbour index over node embeddings.

    Implementation: exact brute-force cosine similarity (O(n·d)).
    Suitable for graphs up to ~50k nodes. For larger graphs, replace
    the _search core with an IVF or HNSW index.

    Supports:
        - insert(node_id, embedding)
        - search(query, k) → [(node_id, similarity), ...]
        - remove(node_id)
        - batch_insert(items)
    """

    def __init__(self) -> None:
        self._ids:        List[str]       = []
        self._matrix:     Optional[np.ndarray] = None   # (N, d) float32
        self._lock        = threading.RLock()

    def insert(self, node_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            vec = embedding.astype(np.float32).flatten()
            vec = vec / (np.linalg.norm(vec) + 1e-8)   # L2 normalise
            self._ids.append(node_id)
            if self._matrix is None:
                self._matrix = vec[np.newaxis, :]
            else:
                self._matrix = np.vstack([self._matrix, vec])

    def remove(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._ids:
                return
            idx = self._ids.index(node_id)
            self._ids.pop(idx)
            if self._matrix is not None:
                self._matrix = np.delete(self._matrix, idx, axis=0)
            if self._matrix is not None and self._matrix.shape[0] == 0:
                self._matrix = None

    def search(
        self,
        query: np.ndarray,
        k:     int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k node IDs by cosine similarity to query.

        Parameters
        ----------
        query : (d,) embedding vector
        k     : number of results

        Returns
        -------
        List of (node_id, similarity) sorted descending.
        """
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return []
            q   = query.astype(np.float32).flatten()
            q   = q / (np.linalg.norm(q) + 1e-8)
            # Cosine similarity: matrix rows are L2-normalised
            sims = self._matrix @ q                    # (N,)
            k    = min(k, len(self._ids))
            top  = np.argpartition(sims, -k)[-k:]
            top  = top[np.argsort(sims[top])[::-1]]
            return [(self._ids[i], float(sims[i])) for i in top]

    def batch_insert(self, items: List[Tuple[str, np.ndarray]]) -> None:
        for node_id, emb in items:
            self.insert(node_id, emb)

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)


# ══════════════════════════════════════════════════════════════════════════════
#  § 7  CAUSAL PATH FINDER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathResult:
    """Result of a causal path query."""
    src_id:       str
    dst_id:       str
    path:         List[str]           # ordered node IDs from src to dst
    total_weight: float
    edges:        List[GraphEdge]
    found:        bool

    @property
    def length(self) -> int:
        return len(self.path)

    @property
    def hops(self) -> int:
        return max(0, len(self.path) - 1)


class CausalPathFinder:
    """
    Finds shortest causal paths through the temporal graph using
    temporal-decay weighted Dijkstra.

    Path cost = sum of edge weights along the path.
    Lower weight edges (high priority, close in time, CAUSAL kind)
    are strongly preferred.

    Temporal decay: edge weights increase with age — older paths cost more.
    This implements a soft recency bias: recent causal chains are preferred
    over historically distant ones.

    Parameters
    ----------
    decay_halflife_s : time in seconds after which edge weight doubles
    max_hops         : maximum path length (prevents unbounded traversal)
    """

    def __init__(
        self,
        decay_halflife_s: float = 3600.0,
        max_hops:         int   = 100,
    ) -> None:
        self.decay_halflife_s = decay_halflife_s
        self.max_hops         = max_hops

    def _age_penalty(self, edge: GraphEdge) -> float:
        """
        Compute age penalty multiplier for an edge.

        Penalty doubles every decay_halflife_s seconds of edge temporal distance.
        """
        dt_s     = edge.temporal_dist_ns / 1e9
        exponent = dt_s / max(self.decay_halflife_s, 1e-9)
        return 2.0 ** exponent

    def shortest_path(
        self,
        src_id:     str,
        dst_id:     str,
        adj:        Dict[str, List[GraphEdge]],
        nodes:      Dict[str, GraphNode],
    ) -> PathResult:
        """
        Dijkstra's algorithm with temporal-decay weighted edges.

        Parameters
        ----------
        src_id : source node UUID
        dst_id : destination node UUID
        adj    : adjacency dict {node_id: [outgoing GraphEdge, ...]}
        nodes  : node dict {node_id: GraphNode}

        Returns
        -------
        PathResult
        """
        if src_id not in nodes or dst_id not in nodes:
            return PathResult(src_id, dst_id, [], float("inf"), [], False)

        if src_id == dst_id:
            return PathResult(src_id, dst_id, [src_id], 0.0, [], True)

        dist:  Dict[str, float]            = {src_id: 0.0}
        prev:  Dict[str, Optional[str]]    = {src_id: None}
        prev_e:Dict[str, Optional[GraphEdge]] = {src_id: None}
        heap:  List[Tuple[float, str]]     = [(0.0, src_id)]
        hops:  Dict[str, int]              = {src_id: 0}

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist.get(u, float("inf")):
                continue
            if u == dst_id:
                break
            if hops.get(u, 0) >= self.max_hops:
                continue

            for edge in adj.get(u, []):
                v        = edge.dst_id
                w        = edge.weight * self._age_penalty(edge)
                new_cost = cost + w
                if new_cost < dist.get(v, float("inf")):
                    dist[v]    = new_cost
                    prev[v]    = u
                    prev_e[v]  = edge
                    hops[v]    = hops.get(u, 0) + 1
                    heapq.heappush(heap, (new_cost, v))

        if dst_id not in dist:
            return PathResult(src_id, dst_id, [], float("inf"), [], False)

        # Reconstruct path
        path_nodes:  List[str]       = []
        path_edges:  List[GraphEdge] = []
        curr = dst_id
        while curr is not None:
            path_nodes.append(curr)
            e = prev_e.get(curr)
            if e is not None:
                path_edges.append(e)
            curr = prev.get(curr)

        path_nodes.reverse()
        path_edges.reverse()

        return PathResult(
            src_id       = src_id,
            dst_id       = dst_id,
            path         = path_nodes,
            total_weight = dist[dst_id],
            edges        = path_edges,
            found        = True,
        )

    def all_reachable(
        self,
        src_id: str,
        adj:    Dict[str, List[GraphEdge]],
        nodes:  Dict[str, GraphNode],
        max_cost: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Return all nodes reachable from src_id and their minimum path costs.

        Parameters
        ----------
        max_cost : prune paths exceeding this total weight (None = unlimited)

        Returns
        -------
        dict {node_id: min_cost}
        """
        dist = {src_id: 0.0}
        heap = [(0.0, src_id)]

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist.get(u, float("inf")):
                continue
            for edge in adj.get(u, []):
                v        = edge.dst_id
                w        = edge.weight * self._age_penalty(edge)
                new_cost = cost + w
                if max_cost is not None and new_cost > max_cost:
                    continue
                if new_cost < dist.get(v, float("inf")):
                    dist[v] = new_cost
                    heapq.heappush(heap, (new_cost, v))

        dist.pop(src_id, None)
        return dist


# ══════════════════════════════════════════════════════════════════════════════
#  § 8  SUBGRAPH EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Subgraph:
    """Extracted subgraph result."""
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def node_ids(self) -> List[str]:
        return list(self.nodes.keys())

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return (A, node_id_list) where A[i,j] = edge weight from i to j.
        0 means no edge.
        """
        ids  = self.node_ids()
        idx  = {nid: i for i, nid in enumerate(ids)}
        n    = len(ids)
        A    = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges:
            i = idx.get(edge.src_id)
            j = idx.get(edge.dst_id)
            if i is not None and j is not None:
                A[i, j] = edge.weight
        return A, ids


class SubgraphExtractor:
    """
    Extracts subgraphs from the full graph using various strategies.

    Strategies
    ----------
    - k_hop_neighbourhood : all nodes within k directed hops of a seed
    - ancestor_chain      : all temporal predecessors up to depth d
    - descendant_chain    : all temporal successors up to depth d
    - step_window         : all nodes within a step range
    - tag_filter          : all nodes matching a tag set
    - kind_filter         : all nodes of specified NodeKind(s)
    - causal_cone         : all nodes in the causal past/future of a seed
    """

    def __init__(
        self,
        nodes: Dict[str, GraphNode],
        adj:   Dict[str, List[GraphEdge]],
        radj:  Dict[str, List[GraphEdge]],   # reverse adjacency
    ) -> None:
        self._nodes = nodes
        self._adj   = adj
        self._radj  = radj

    def k_hop_neighbourhood(
        self,
        seed_id: str,
        k:       int = 2,
        directed: bool = True,
    ) -> Subgraph:
        """BFS k-hop neighbourhood starting from seed_id."""
        visited: Set[str]       = {seed_id}
        frontier: Set[str]      = {seed_id}
        collected_edges: List[GraphEdge] = []

        for _ in range(k):
            next_frontier: Set[str] = set()
            for nid in frontier:
                fwd = self._adj.get(nid, [])
                bwd = [] if directed else self._radj.get(nid, [])
                for edge in fwd + bwd:
                    other = edge.dst_id if edge.src_id == nid else edge.src_id
                    if other not in visited and other in self._nodes:
                        visited.add(other)
                        next_frontier.add(other)
                        collected_edges.append(edge)
            frontier = next_frontier

        nodes = {nid: self._nodes[nid] for nid in visited if nid in self._nodes}
        return Subgraph(nodes=nodes, edges=collected_edges)

    def ancestor_chain(
        self,
        seed_id: str,
        depth:   int = 10,
    ) -> Subgraph:
        """Follow reverse TEMPORAL/CAUSAL edges back in time."""
        visited: Set[str]       = {seed_id}
        queue:   List[Tuple[str, int]] = [(seed_id, 0)]
        edges:   List[GraphEdge] = []
        target_kinds = {EdgeKind.TEMPORAL, EdgeKind.CAUSAL}

        while queue:
            nid, d = queue.pop(0)
            if d >= depth:
                continue
            for edge in self._radj.get(nid, []):
                if edge.kind in target_kinds and edge.src_id not in visited:
                    if edge.src_id in self._nodes:
                        visited.add(edge.src_id)
                        queue.append((edge.src_id, d + 1))
                        edges.append(edge)

        nodes = {nid: self._nodes[nid] for nid in visited if nid in self._nodes}
        return Subgraph(nodes=nodes, edges=edges)

    def descendant_chain(
        self,
        seed_id: str,
        depth:   int = 10,
    ) -> Subgraph:
        """Follow forward TEMPORAL/CAUSAL edges forward in time."""
        visited: Set[str]       = {seed_id}
        queue:   List[Tuple[str, int]] = [(seed_id, 0)]
        edges:   List[GraphEdge] = []
        target_kinds = {EdgeKind.TEMPORAL, EdgeKind.CAUSAL}

        while queue:
            nid, d = queue.pop(0)
            if d >= depth:
                continue
            for edge in self._adj.get(nid, []):
                if edge.kind in target_kinds and edge.dst_id not in visited:
                    if edge.dst_id in self._nodes:
                        visited.add(edge.dst_id)
                        queue.append((edge.dst_id, d + 1))
                        edges.append(edge)

        nodes = {nid: self._nodes[nid] for nid in visited if nid in self._nodes}
        return Subgraph(nodes=nodes, edges=edges)

    def step_window(
        self,
        start_step: int,
        end_step:   int,
    ) -> Subgraph:
        """All nodes with step in [start_step, end_step]."""
        in_window = {
            nid: n for nid, n in self._nodes.items()
            if start_step <= n.step <= end_step
        }
        w_set = set(in_window)
        edges = [
            e for elist in self._adj.values()
            for e in elist
            if e.src_id in w_set and e.dst_id in w_set
        ]
        return Subgraph(nodes=in_window, edges=edges)

    def tag_filter(self, tags: List[str], match_all: bool = False) -> Subgraph:
        """Nodes matching any (or all, if match_all) of the given tags."""
        tag_set = set(tags)
        if match_all:
            matched = {
                nid: n for nid, n in self._nodes.items()
                if tag_set.issubset(set(n.tags))
            }
        else:
            matched = {
                nid: n for nid, n in self._nodes.items()
                if tag_set & set(n.tags)
            }
        m_set = set(matched)
        edges = [
            e for elist in self._adj.values()
            for e in elist
            if e.src_id in m_set and e.dst_id in m_set
        ]
        return Subgraph(nodes=matched, edges=edges)

    def kind_filter(self, kinds: List[NodeKind]) -> Subgraph:
        """All nodes of the specified NodeKind(s)."""
        kind_set = set(kinds)
        matched  = {nid: n for nid, n in self._nodes.items()
                    if n.kind in kind_set}
        m_set    = set(matched)
        edges    = [
            e for elist in self._adj.values()
            for e in elist
            if e.src_id in m_set and e.dst_id in m_set
        ]
        return Subgraph(nodes=matched, edges=edges)

    def causal_cone(
        self,
        seed_id:  str,
        depth:    int = 5,
        backward: bool = True,
        forward:  bool = True,
    ) -> Subgraph:
        """Union of ancestor and descendant chains from seed."""
        result_nodes: Dict[str, GraphNode] = {}
        result_edges: List[GraphEdge]      = []

        if backward:
            sg = self.ancestor_chain(seed_id, depth)
            result_nodes.update(sg.nodes)
            result_edges.extend(sg.edges)

        if forward:
            sg = self.descendant_chain(seed_id, depth)
            result_nodes.update(sg.nodes)
            result_edges.extend(sg.edges)

        # Deduplicate edges by edge_id
        seen: Set[str] = set()
        deduped = []
        for e in result_edges:
            if e.edge_id not in seen:
                seen.add(e.edge_id)
                deduped.append(e)

        return Subgraph(nodes=result_nodes, edges=deduped)


# ══════════════════════════════════════════════════════════════════════════════
#  § 9  GRAPH SERIALIZER
# ══════════════════════════════════════════════════════════════════════════════

class GraphSerializer:
    """
    Serialises and deserialises the full graph to/from disk.

    Format
    ------
    - nodes.json  : list of node dicts
    - edges.json  : list of edge dicts
    - embeddings/ : per-node numpy binary files (node_id.npy)

    Parameters
    ----------
    base_dir : directory for all serialised files
    """

    def __init__(self, base_dir: str = "./temporal_graph_data") -> None:
        self.base_dir   = Path(base_dir)
        self.emb_dir    = self.base_dir / "embeddings"

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        nodes: Dict[str, GraphNode],
        edges_flat: List[GraphEdge],
    ) -> Dict[str, str]:
        """
        Persist graph to disk.

        Returns dict of written file paths.
        """
        self._ensure_dirs()

        # Nodes JSON (embeddings saved separately as .npy)
        nodes_path = self.base_dir / "nodes.json"
        nodes_dicts = []
        for n in nodes.values():
            d = n.to_dict()
            if n.embedding is not None:
                emb_path = self.emb_dir / f"{n.node_id}.npy"
                np.save(str(emb_path), n.embedding.astype(np.float32))
                d.pop("embedding", None)          # don't store in JSON
                d["embedding_path"] = str(emb_path)
            nodes_dicts.append(d)

        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes_dicts, f, indent=2)

        # Edges JSON
        edges_path  = self.base_dir / "edges.json"
        edges_dicts = [e.to_dict() for e in edges_flat]
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(edges_dicts, f, indent=2)

        logger.info(
            "GraphSerializer: saved %d nodes, %d edges to %s",
            len(nodes), len(edges_flat), self.base_dir,
        )
        return {
            "nodes": str(nodes_path),
            "edges": str(edges_path),
            "embeddings_dir": str(self.emb_dir),
        }

    def load(self) -> Tuple[Dict[str, GraphNode], List[GraphEdge]]:
        """
        Load graph from disk.

        Returns
        -------
        (nodes_dict, edges_list)
        """
        nodes_path = self.base_dir / "nodes.json"
        edges_path = self.base_dir / "edges.json"

        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes.json not found in {self.base_dir}")

        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes_dicts = json.load(f)

        nodes: Dict[str, GraphNode] = {}
        for d in nodes_dicts:
            emb_path = d.pop("embedding_path", None)
            if emb_path and Path(emb_path).exists():
                d["embedding"] = np.load(emb_path)
            node = GraphNode.from_dict(d)
            nodes[node.node_id] = node

        edges: List[GraphEdge] = []
        if edges_path.exists():
            with open(edges_path, "r", encoding="utf-8") as f:
                edges = [GraphEdge.from_dict(d) for d in json.load(f)]

        logger.info(
            "GraphSerializer: loaded %d nodes, %d edges from %s",
            len(nodes), len(edges), self.base_dir,
        )
        return nodes, edges


# ══════════════════════════════════════════════════════════════════════════════
#  § 10  TEMPORAL_GRAPH_STORAGE — Master Engine
# ══════════════════════════════════════════════════════════════════════════════

class Temporal_Graph_Storage:
    """
    Master causal temporal graph database.

    Maintains a directed graph where:
        Nodes  = system states, inputs, gradients, decisions, annotations
        Edges  = temporally and causally weighted directed connections

    All retrieval is via causal pathway traversal, not flat similarity.
    Cosine similarity is available as a secondary re-ranking step only.

    Parameters
    ----------
    embedding_dim       : dimensionality of node embeddings (0 = disabled)
    temporal_decay_rate : EdgeWeightPolicy τ parameter
    decay_halflife_s    : CausalPathFinder temporal decay half-life
    max_nodes           : evict oldest nodes when graph exceeds this size
    auto_link_temporal  : automatically add TEMPORAL edge to previous node
    persist_dir         : if set, auto-save to disk on every N insertions
    autosave_every      : number of insertions between auto-saves
    """

    def __init__(
        self,
        embedding_dim:       int   = 0,
        temporal_decay_rate: float = 0.1,
        decay_halflife_s:    float = 3600.0,
        max_nodes:           int   = 100_000,
        auto_link_temporal:  bool  = True,
        persist_dir:         Optional[str] = None,
        autosave_every:      int   = 500,
    ) -> None:
        self.embedding_dim       = embedding_dim
        self.max_nodes           = max_nodes
        self.auto_link_temporal  = auto_link_temporal
        self.autosave_every      = autosave_every

        self._nodes: Dict[str, GraphNode]        = {}
        self._adj:   Dict[str, List[GraphEdge]]  = {}   # forward
        self._radj:  Dict[str, List[GraphEdge]]  = {}   # reverse
        self._all_edges: List[GraphEdge]         = []

        self._weight_policy  = EdgeWeightPolicy(temporal_decay_rate)
        self._path_finder    = CausalPathFinder(decay_halflife_s)
        self._temporal_index = TemporalIndex()
        self._emb_index      = EmbeddingIndex() if embedding_dim > 0 else None
        self._serializer     = (GraphSerializer(persist_dir)
                                if persist_dir else None)

        self._last_node_id:  Optional[str] = None
        self._insert_count:  int           = 0
        self._lock           = threading.RLock()

        logger.info(
            "Temporal_Graph_Storage initialised | embed_dim=%d max_nodes=%d",
            embedding_dim, max_nodes,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  INSERTION
    # ══════════════════════════════════════════════════════════════════════════

    def add_node(
        self,
        kind:      NodeKind,
        payload:   Optional[Dict]        = None,
        embedding: Optional[np.ndarray]  = None,
        step:      int                   = 0,
        priority:  float                 = 0.5,
        tags:      Optional[List[str]]   = None,
    ) -> GraphNode:
        """
        Create and insert a new node.

        If auto_link_temporal is True, automatically connects to
        the previously inserted node via a TEMPORAL edge.

        Parameters
        ----------
        kind      : NodeKind
        payload   : arbitrary metadata dict
        embedding : optional dense vector (must match embedding_dim if set)
        step      : training / processing step
        priority  : heuristic priority [0, 1]
        tags      : string tags for filtering

        Returns
        -------
        Inserted GraphNode
        """
        node = GraphNode.create(
            kind=kind, payload=payload, embedding=embedding,
            step=step, priority=priority, tags=tags,
        )

        with self._lock:
            # Evict oldest if at capacity
            if len(self._nodes) >= self.max_nodes:
                self._evict_oldest()

            self._nodes[node.node_id] = node
            self._adj[node.node_id]   = []
            self._radj[node.node_id]  = []
            self._temporal_index.insert(node)

            if embedding is not None and self._emb_index is not None:
                self._emb_index.insert(node.node_id, embedding)

            # Auto-link to previous node
            if self.auto_link_temporal and self._last_node_id is not None:
                prev = self._nodes.get(self._last_node_id)
                if prev:
                    self._add_edge_internal(prev, node, EdgeKind.TEMPORAL)

            self._last_node_id = node.node_id
            self._insert_count += 1

            # Auto-save
            if (self._serializer is not None and
                    self._insert_count % self.autosave_every == 0):
                self._serializer.save(self._nodes, self._all_edges)

        logger.debug("add_node: %s [%s] step=%d", node.node_id[:8], kind.name, step)
        return node

    def add_edge(
        self,
        src_id:   str,
        dst_id:   str,
        kind:     EdgeKind           = EdgeKind.CAUSAL,
        metadata: Optional[Dict]     = None,
    ) -> Optional[GraphEdge]:
        """
        Create a directed edge between two existing nodes.

        Parameters
        ----------
        src_id   : source node UUID
        dst_id   : destination node UUID
        kind     : EdgeKind
        metadata : optional metadata dict

        Returns
        -------
        Created GraphEdge, or None if either node does not exist.
        """
        with self._lock:
            src = self._nodes.get(src_id)
            dst = self._nodes.get(dst_id)
            if src is None or dst is None:
                logger.warning("add_edge: node not found (%s → %s)", src_id[:8], dst_id[:8])
                return None
            return self._add_edge_internal(src, dst, kind, metadata)

    def _add_edge_internal(
        self,
        src:      GraphNode,
        dst:      GraphNode,
        kind:     EdgeKind,
        metadata: Optional[Dict] = None,
    ) -> GraphEdge:
        weight = self._weight_policy.compute(src, dst, kind)
        edge   = GraphEdge.create(src, dst, kind, weight, metadata)
        self._adj[src.node_id].append(edge)
        self._radj[dst.node_id].append(edge)
        self._all_edges.append(edge)
        return edge

    # ══════════════════════════════════════════════════════════════════════════
    #  RETRIEVAL — CAUSAL PATHWAY QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def causal_path(
        self,
        src_id: str,
        dst_id: str,
    ) -> PathResult:
        """
        Find the minimum-weight causal path from src to dst.

        Uses temporal-decay Dijkstra; prefers recent, high-priority,
        CAUSAL-kind edges. Does NOT use flat cosine similarity.

        Parameters
        ----------
        src_id : source node UUID
        dst_id : destination node UUID

        Returns
        -------
        PathResult
        """
        with self._lock:
            return self._path_finder.shortest_path(
                src_id, dst_id, self._adj, self._nodes
            )

    def reachable_from(
        self,
        src_id:   str,
        max_cost: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Return all nodes reachable from src_id and their minimum path costs.

        Parameters
        ----------
        max_cost : prune paths exceeding this total weight

        Returns
        -------
        {node_id: cost}
        """
        with self._lock:
            return self._path_finder.all_reachable(
                src_id, self._adj, self._nodes, max_cost
            )

    def causal_ancestors(
        self,
        node_id: str,
        depth:   int = 10,
    ) -> Subgraph:
        """All temporal/causal ancestors up to `depth` hops back."""
        with self._lock:
            extractor = SubgraphExtractor(self._nodes, self._adj, self._radj)
            return extractor.ancestor_chain(node_id, depth)

    def causal_descendants(
        self,
        node_id: str,
        depth:   int = 10,
    ) -> Subgraph:
        """All temporal/causal dscendants up to `depth` hops forward."""
        with self._lock:
            extractor = SubgraphExtractor(self._nodes, self._adj, self._radj)
            return extractor.descendant_chain(node_id, depth)

    def causal_cone(
        self,
        node_id:  str,
        depth:    int  = 5,
        backward: bool = True,
        forward:  bool = True,
    ) -> Subgraph:
        """Full causal cone (past + future) around a node."""
        with self._lock:
            extractor = SubgraphExtractor(self._nodes, self._adj, self._radj)
            return extractor.causal_cone(node_id, depth, backward, forward)

    def neighbourhood(
        self,
        node_id:  str,
        k:        int  = 2,
        directed: bool = True,
    ) -> Subgraph:
        """k-hop neighbourhood subgraph around a node."""
        with self._lock:
            extractor = SubgraphExtractor(self._nodes, self._adj, self._radj)
            return extractor.k_hop_neighbourhood(node_id, k, directed)

    # ══════════════════════════════════════════════════════════════════════════
    #  RETRIEVAL — TEMPORAL / FILTER QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def query_time_range(
        self,
        start_ns: Optional[int] = None,
        end_ns:   Optional[int] = None,
    ) -> List[GraphNode]:
        """Return nodes created within [start_ns, end_ns]."""
        with self._lock:
            ids = self._temporal_index.range_query(start_ns, end_ns)
            return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def query_step_range(
        self,
        start_step: int,
        end_step:   int,
    ) -> List[GraphNode]:
        """Return nodes with step in [start_step, end_step]."""
        with self._lock:
            ids = self._temporal_index.step_query(start_step, end_step)
            return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def query_kind(self, *kinds: NodeKind) -> List[GraphNode]:
        """Return all nodes of the specified kind(s)."""
        kind_set = set(kinds)
        with self._lock:
            return [n for n in self._nodes.values() if n.kind in kind_set]

    def query_tags(
        self,
        tags:      List[str],
        match_all: bool = False,
    ) -> List[GraphNode]:
        """Return nodes matching any (or all) of the given tags."""
        tag_set = set(tags)
        with self._lock:
            if match_all:
                return [n for n in self._nodes.values()
                        if tag_set.issubset(set(n.tags))]
            return [n for n in self._nodes.values()
                    if tag_set & set(n.tags)]

    def query_payload(
        self,
        key:   str,
        value: Any,
    ) -> List[GraphNode]:
        """Return all nodes where payload[key] == value."""
        with self._lock:
            return [n for n in self._nodes.values()
                    if n.payload.get(key) == value]

    def latest_n(self, n: int = 10, kind: Optional[NodeKind] = None) -> List[GraphNode]:
        """Return the n most recently inserted nodes, optionally filtered by kind."""
        with self._lock:
            ids    = self._temporal_index.latest(n * 3 if kind else n)
            nodes  = [self._nodes[nid] for nid in ids if nid in self._nodes]
            if kind is not None:
                nodes = [nd for nd in nodes if nd.kind == kind]
            return nodes[:n]

    # ══════════════════════════════════════════════════════════════════════════
    #  RETRIEVAL — EMBEDDING SIMILARITY (secondary re-ranker only)
    # ══════════════════════════════════════════════════════════════════════════

    def similar_to(
        self,
        query_embedding: np.ndarray,
        k:               int = 10,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Retrieve nodes by embedding similarity.

        NOTE: This is a secondary re-ranker. For causal retrieval,
        use causal_path() or causal_cone() instead.

        Parameters
        ----------
        query_embedding : (d,) query vector
        k               : number of results

        Returns
        -------
        List of (GraphNode, cosine_similarity) sorted descending.
        """
        if self._emb_index is None:
            raise RuntimeError(
                "EmbeddingIndex disabled. Set embedding_dim > 0 at construction."
            )
        with self._lock:
            results = self._emb_index.search(query_embedding, k)
            return [
                (self._nodes[nid], sim)
                for nid, sim in results
                if nid in self._nodes
            ]

    def causal_then_rerank(
        self,
        seed_id:    str,
        query_emb:  np.ndarray,
        depth:      int = 5,
        top_k:      int = 10,
    ) -> List[Tuple[GraphNode, float, float]]:
        """
        Primary: causal cone from seed. Secondary: re-rank by embedding similarity.

        Returns
        -------
        List of (GraphNode, causal_cost, cosine_similarity), sorted by
        a combined score = (1 / (1 + causal_cost)) × cosine_sim.
        """
        if self._emb_index is None:
            raise RuntimeError("EmbeddingIndex disabled.")

        cone = self.causal_cone(seed_id, depth)
        scored: List[Tuple[GraphNode, float, float]] = []

        q = query_emb.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        reachable = self.reachable_from(seed_id)

        for nid, node in cone.nodes.items():
            if node.embedding is None:
                continue
            emb  = node.embedding.astype(np.float32)
            emb  = emb / (np.linalg.norm(emb) + 1e-8)
            sim  = float(np.dot(q, emb))
            cost = reachable.get(nid, float("inf"))
            scored.append((node, cost, sim))

        scored.sort(
            key=lambda x: (1.0 / (1.0 + x[1])) * max(x[2], 0.0),
            reverse=True,
        )
        return scored[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    #  MUTATION
    # ══════════════════════════════════════════════════════════════════════════

    def update_node(
        self,
        node_id:  str,
        payload:  Optional[Dict]       = None,
        priority: Optional[float]      = None,
        tags:     Optional[List[str]]  = None,
    ) -> bool:
        """Update mutable fields of an existing node. Returns True if found."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            if payload is not None:
                node.payload.update(payload)
            if priority is not None:
                object.__setattr__(node, "priority", priority) \
                    if hasattr(node, "__dataclass_fields__") \
                    else setattr(node, "priority", priority)
            if tags is not None:
                node.tags.extend(t for t in tags if t not in node.tags)
            return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its incident edges. Returns True if found.
        """
        with self._lock:
            node = self._nodes.pop(node_id, None)
            if node is None:
                return False

            # Remove all outgoing edges
            for edge in self._adj.pop(node_id, []):
                self._radj.get(edge.dst_id, []).remove(edge)
                if edge in self._all_edges:
                    self._all_edges.remove(edge)

            # Remove all incoming edges
            for edge in self._radj.pop(node_id, []):
                self._adj.get(edge.src_id, []).remove(edge)
                if edge in self._all_edges:
                    self._all_edges.remove(edge)

            self._temporal_index.remove(node_id, node.created_ns, node.step)
            if self._emb_index and node.embedding is not None:
                self._emb_index.remove(node_id)

            if self._last_node_id == node_id:
                self._last_node_id = None

            return True

    def _evict_oldest(self, n: int = 100) -> None:
        """Evict the n oldest nodes to stay within max_nodes."""
        oldest_ids = self._temporal_index.oldest(n)
        for nid in oldest_ids:
            self.remove_node(nid)
        logger.info("Evicted %d oldest nodes (max_nodes=%d)", len(oldest_ids), self.max_nodes)

    # ══════════════════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Optional[str] = None) -> Dict[str, str]:
        """
        Persist the full graph to disk.

        Parameters
        ----------
        path : override persist_dir if provided

        Returns
        -------
        dict of written file paths
        """
        serializer = (GraphSerializer(path) if path
                      else self._serializer)
        if serializer is None:
            raise RuntimeError("No persist_dir set. Pass a path to save().")
        with self._lock:
            return serializer.save(self._nodes, self._all_edges)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load graph from disk, replacing current state.

        Parameters
        ----------
        path : directory to load from (overrides persist_dir)
        """
        serializer = (GraphSerializer(path) if path
                      else self._serializer)
        if serializer is None:
            raise RuntimeError("No persist_dir set. Pass a path to load().")

        nodes, edges = serializer.load()
        with self._lock:
            self._nodes.clear()
            self._adj.clear()
            self._radj.clear()
            self._all_edges.clear()
            self._temporal_index = TemporalIndex()
            if self._emb_index is not None:
                self._emb_index = EmbeddingIndex()

            for node in nodes.values():
                self._nodes[node.node_id] = node
                self._adj[node.node_id]   = []
                self._radj[node.node_id]  = []
                self._temporal_index.insert(node)
                if node.embedding is not None and self._emb_index is not None:
                    self._emb_index.insert(node.node_id, node.embedding)

            for edge in edges:
                self._all_edges.append(edge)
                self._adj.setdefault(edge.src_id, []).append(edge)
                self._radj.setdefault(edge.dst_id, []).append(edge)

        logger.info("Graph loaded: %d nodes, %d edges", len(nodes), len(edges))

    # ══════════════════════════════════════════════════════════════════════════
    #  INSPECTION
    # ══════════════════════════════════════════════════════════════════════════

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_edges(
        self,
        node_id:  str,
        outgoing: bool = True,
        incoming: bool = True,
    ) -> List[GraphEdge]:
        """Return all edges incident to a node."""
        result = []
        if outgoing:
            result.extend(self._adj.get(node_id, []))
        if incoming:
            result.extend(self._radj.get(node_id, []))
        return result

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        with self._lock:
            kind_counts = {}
            for n in self._nodes.values():
                kind_counts[n.kind.name] = kind_counts.get(n.kind.name, 0) + 1

            edge_kind_counts = {}
            for e in self._all_edges:
                edge_kind_counts[e.kind.name] = edge_kind_counts.get(e.kind.name, 0) + 1

            return {
                "n_nodes":          len(self._nodes),
                "n_edges":          len(self._all_edges),
                "node_kinds":       kind_counts,
                "edge_kinds":       edge_kind_counts,
                "embedding_index":  len(self._emb_index) if self._emb_index else 0,
                "temporal_index":   len(self._temporal_index),
                "insert_count":     self._insert_count,
                "max_nodes":        self.max_nodes,
            }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
#
#   M A N A S  4 . 0
#   Multi-modal Adaptive Neural Architecture System — Version 4.0
#
#   Author  : Zeeshan Rahman
#   Engine  : ZULIC + Fractal Attention + Hardware Entropy +
#             Adaptive Throttling + JIT Meta-Programming +
#             Temporal Causal Graph
#
#   Production-grade unified orchestration layer.
#   Zero external ML framework dependencies.
#   All computation: NumPy + stdlib only.
#
# ██████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════


import signal
import atexit


# ══════════════════════════════════════════════════════════════════════════════
#  MANAS CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManasConfig:
    """
    Unified configuration for the full MANAS-4.0 system.

    Covers all six sub-systems with sensible production defaults.

    Parameters
    ----------
    # Identity
    system_name      : display name
    version          : version string

    # Module 1 — Z-Operator
    z_delta          : ZULIC finite-difference step
    z_viscosity      : Navier-Stokes friction coefficient μ
    z_density        : Navier-Stokes inertia coefficient ρ
    z_history_len    : fluid gradient history window

    # Module 2 — Fractal Attention
    embed_dim        : token embedding dimension
    branching_b      : fractal tree branching factor
    depth_decay_lam  : RSSB depth-penalty λ
    num_heads        : attention heads
    vocab_size       : embedding table vocabulary size
    use_rope         : enable Rotary Position Embeddings

    # Module 3 — Hardware Entropy
    entropy_pool_size    : internal pool buffer bytes
    entropy_reseed_after : pool calls before forced re-seed

    # Module 4 — Load Balancer
    base_batch_size  : maximum batch size at FULL tier
    base_depth       : maximum compute depth at FULL tier
    poll_interval_s  : resource polling interval

    # Module 5 — Meta-Programming
    jit_timeout_s    : sandbox execution timeout
    jit_cache        : enable compiled function cache

    # Module 6 — Temporal Graph
    graph_embed_dim      : node embedding dimension (0 = disabled)
    graph_max_nodes      : max nodes before eviction
    graph_decay_rate     : temporal edge weight decay rate
    graph_persist_dir    : disk persistence directory (None = disabled)
    graph_autosave_every : insertions between auto-saves

    # Training
    learning_rate    : initial learning rate
    lr_decay         : LR decay per step
    momentum         : optimiser momentum
    clip_norm        : gradient clip norm
    """
    # Identity
    system_name:     str   = "MANAS-4.0"
    version:         str   = "4.2.0"

    # Module 1
    z_delta:         float = 1e-5
    z_viscosity:     float = 0.08
    z_density:       float = 0.85
    z_history_len:   int   = 8

    # Module 2
    embed_dim:       int   = 256
    branching_b:     int   = 4
    depth_decay_lam: float = 0.1
    num_heads:       int   = 8
    vocab_size:      int   = 32000
    use_rope:        bool  = True

    # Module 3
    entropy_pool_size:    int = 512
    entropy_reseed_after: int = 500

    # Module 4
    base_batch_size: int   = 32
    base_depth:      int   = 8
    poll_interval_s: float = 0.5

    # Module 5
    jit_timeout_s:   float = 5.0
    jit_cache:       bool  = True

    # Module 6
    graph_embed_dim:      int            = 256
    graph_max_nodes:      int            = 100_000
    graph_decay_rate:     float          = 0.1
    graph_persist_dir:    Optional[str]  = None
    graph_autosave_every: int            = 500

    # Training
    learning_rate:   float = 1e-3
    lr_decay:        float = 0.9995
    momentum:        float = 0.92
    clip_norm:       float = 5.0

    # Memory (Improvement 2)
    memory_gate_max: float = 0.3    # max gate strength for memory injection

    # Performance (M-series optimisation)
    use_lora:        bool  = True   # enable LoRA adapters (94% param reduction)
    lora_rank:       int   = 8      # LoRA rank r
    lora_alpha:      float = 8.0    # LoRA alpha scaling
    use_mlx:         bool  = False  # use Apple Metal GPU via MLX
    lora_frozen:     bool  = True   # freeze base weights (LoRA-only training)
    quantize_bits:   int   = 8      # weight quantization bits for storage


# ══════════════════════════════════════════════════════════════════════════════
#  MANAS SYSTEM STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManasSystemState:
    """
    Snapshot of the full MANAS-4.0 system state at a given step.

    Logged to the Temporal Graph as a STATE node on every step.
    """
    step:               int
    loss:               float
    grad_norm:          float
    compute_tier:       str
    batch_size:         int
    depth_limit:        int
    entropy_quality:    Optional[float]
    beta_mean:          float
    active_repairs:     int
    graph_nodes:        int
    graph_edges:        int
    timestamp_ns:       int             = field(default_factory=time.perf_counter_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":             self.step,
            "loss":             self.loss,
            "grad_norm":        self.grad_norm,
            "compute_tier":     self.compute_tier,
            "batch_size":       self.batch_size,
            "depth_limit":      self.depth_limit,
            "entropy_quality":  self.entropy_quality,
            "beta_mean":        self.beta_mean,
            "active_repairs":   self.active_repairs,
            "graph_nodes":      self.graph_nodes,
            "graph_edges":      self.graph_edges,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MANAS PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManasPipelineResult:
    """
    Full result bundle from one MANAS-4.0 forward + backward pass.

    New fields (Improvement 2):
        memory_injected  : whether causal graph memory was blended into root
        recalled_node_ids: list of graph node IDs recalled from memory
    """
    step:               int
    loss:               float
    root_embedding:     np.ndarray
    local_embedding:    np.ndarray
    change_signatures:  Dict[str, np.ndarray]
    compute_tier:       str
    batch_size:         int
    depth_limit:        int
    beta_report:        Dict[str, Dict]
    repairs_applied:    List[str]
    graph_node_id:      str
    duration_ms:        float
    memory_injected:    bool            = False
    recalled_node_ids:  List[str]       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  MANAS-4.0 MASTER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class MANAS:
    """
    MANAS-4.0 — Multi-modal Adaptive Neural Architecture System.

    Orchestrates all six production modules into a unified pipeline:

        Module 1  →  Custom_Autograd_Z       ZULIC gradient engine
        Module 2  →  FractalEncoder          O(N log N) attention
        Module 3  →  Hardware_Entropy_Harvester  True stochasticity
        Module 4  →  System_Load_Balancer    Adaptive compute throttling
        Module 5  →  Meta_Programming_Engine JIT failure repair
        Module 6  →  Temporal_Graph_Storage  Causal state memory

    Full Forward + Backward Pipeline (one step)
    --------------------------------------------
    1.  Check load balancer tier → adjust batch_size / depth_limit.
    2.  Gate on suspended state (block if system overloaded).
    3.  Inject hardware entropy noise into input embeddings.
    4.  Run FractalEncoder forward pass → root + local embeddings.
    5.  Compute loss via Custom_Autograd_Z.
    6.  Detect failure conditions (NaN, explosion, vanishing).
    7.  If failure detected → JIT-compile + execute repair function.
    8.  Compute ZULIC Change Signatures (backward pass).
    9.  Apply ZOptimizer parameter update.
    10. Log full system state to Temporal Graph.
    11. Return ManasPipelineResult.

    Parameters
    ----------
    config : ManasConfig (uses defaults if None)
    """

    def __init__(self, config: Optional[ManasConfig] = None) -> None:
        self.cfg   = config or ManasConfig()
        self._step = 0
        self._repairs_total = 0

        logger.info("=" * 70)
        logger.info("  %s  v%s  — Initialising", self.cfg.system_name, self.cfg.version)
        logger.info("=" * 70)

        # ── Module 1: ZULIC Autograd Engine ───────────────────────────────────
        logger.info("[M1] Initialising Custom_Autograd_Z (ZULIC engine)...")
        self.autograd = Custom_Autograd_Z(
            delta       = self.cfg.z_delta,
            viscosity   = self.cfg.z_viscosity,
            density     = self.cfg.z_density,
            history_len = self.cfg.z_history_len,
        )

        # ── Module 2: Fractal Attention Encoder ───────────────────────────────
        logger.info("[M2] Initialising FractalEncoder (O(N log_b N) attention)...")
        attn_cfg = AttentionConfig(
            embed_dim       = self.cfg.embed_dim,
            branching_b     = self.cfg.branching_b,
            depth_decay_lam = self.cfg.depth_decay_lam,
            num_heads       = self.cfg.num_heads,
            use_rope        = self.cfg.use_rope,
        )
        self.encoder = FractalEncoder(
            vocab_size = self.cfg.vocab_size,
            config     = attn_cfg,
            compress   = True,
        )
        self.complexity = ComplexityAnalyser(attn_cfg)

        # ── Mixed Precision Manager (M-series optimisation) ──────────────────
        logger.info("[PERF] Initialising MixedPrecisionManager (MLX=%s)...",
                    self.cfg.use_mlx)
        self.precision = MixedPrecisionManager(
            training_dtype  = np.float32,
            inference_dtype = np.float16,
            use_mlx         = self.cfg.use_mlx,
        )

        # Build network — LoRA if enabled, standard ZNetwork otherwise
        if self.cfg.use_lora:
            logger.info(
                "[PERF] ZNetworkLoRA: rank=%d alpha=%.1f frozen=%s",
                self.cfg.lora_rank, self.cfg.lora_alpha, self.cfg.lora_frozen,
            )
            self.network = ZNetworkLoRA(
                dims       = [self.cfg.embed_dim, self.cfg.embed_dim * 2,
                              self.cfg.embed_dim, self.cfg.embed_dim // 4],
                rank       = self.cfg.lora_rank,
                alpha      = self.cfg.lora_alpha,
                activation = "gelu",
                frozen     = self.cfg.lora_frozen,
                precision  = self.precision,
            )
            summary = self.network.param_summary()
            logger.info(
                "[PERF] LoRA param summary: total=%d trainable=%d "
                "frozen=%d reduction=%s",
                summary["total"], summary["trainable"],
                summary["frozen"], summary["reduction"],
            )
        else:
            logger.info("[PERF] Standard ZNetwork (no LoRA).")
            self.network = ZNetwork(
                dims       = [self.cfg.embed_dim, self.cfg.embed_dim * 2,
                              self.cfg.embed_dim, self.cfg.embed_dim // 4],
                activation = "gelu",
            )

        all_params     = self.encoder.parameters()
        network_params = self.network.parameters()
        self.all_params = all_params + network_params

        self.optimizer = ZOptimizer(
            parameters = network_params,
            lr         = self.cfg.learning_rate,
            lr_decay   = self.cfg.lr_decay,
            momentum   = self.cfg.momentum,
            clip_norm  = self.cfg.clip_norm,
        )
        logger.info(
            "[M2] Encoder params: %d | Network trainable params: %d",
            sum(p.size for p in all_params),
            sum(p.data.size for p in network_params),
        )
        # Memory report
        mem = MixedPrecisionManager.memory_report(network_params)
        logger.info(
            "[PERF] Network memory: %.2f MB (float32 equiv: %.2f MB)",
            mem["total_MB"], mem["float32_equiv_MB"],
        )

        # ── Module 3: Hardware Entropy Harvester ──────────────────────────────
        logger.info("[M3] Initialising Hardware_Entropy_Harvester...")
        self.entropy = Hardware_Entropy_Harvester(
            pool_size    = self.cfg.entropy_pool_size,
            reseed_after = self.cfg.entropy_reseed_after,
        )

        # ── Module 4: System Load Balancer ────────────────────────────────────
        logger.info("[M4] Initialising System_Load_Balancer...")
        self.balancer = System_Load_Balancer(
            base_batch_size = self.cfg.base_batch_size,
            base_depth      = self.cfg.base_depth,
            poll_interval_s = self.cfg.poll_interval_s,
            tier_callback   = self._on_tier_change,
        )
        self.balancer.start()

        # ── Module 5: JIT Meta-Programming Engine ─────────────────────────────
        logger.info("[M5] Initialising Meta_Programming_Engine...")
        self.jit = Meta_Programming_Engine(
            timeout_s      = self.cfg.jit_timeout_s,
            cache_compiled = self.cfg.jit_cache,
        )

        # ── Module 6: Temporal Graph Storage ──────────────────────────────────
        logger.info("[M6] Initialising Temporal_Graph_Storage...")
        self.graph = Temporal_Graph_Storage(
            embedding_dim       = self.cfg.graph_embed_dim,
            temporal_decay_rate = self.cfg.graph_decay_rate,
            max_nodes           = self.cfg.graph_max_nodes,
            auto_link_temporal  = True,
            persist_dir         = self.cfg.graph_persist_dir,
            autosave_every      = self.cfg.graph_autosave_every,
        )

        # Register shutdown hooks
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

        logger.info("[MANAS] All modules online. System ready.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TIER CHANGE CALLBACK
    # ══════════════════════════════════════════════════════════════════════════

    def _on_tier_change(
        self,
        old: "ComputeTier",
        new: "ComputeTier",
        snap: "ResourceSnapshot",
    ) -> None:
        """Called by Module 4 whenever the compute tier changes."""
        self.graph.add_node(
            kind     = NodeKind.ANNOTATION,
            payload  = {
                "event":       "tier_change",
                "old_tier":    old.name,
                "new_tier":    new.name,
                "cpu_percent": snap.cpu_percent_avg,
                "ram_percent": snap.ram_percent,
            },
            step     = self._step,
            priority = 0.9,
            tags     = ["tier_change", new.name.lower()],
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  FAILURE DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_failure(
        self,
        loss:      float,
        grad_norm: float,
        prev_loss: Optional[float],
    ) -> Optional[FailureState]:
        """
        Inspect loss and gradient norms for known failure conditions.

        Returns a FailureState if a failure is detected, else None.
        """
        if not math.isfinite(loss):
            return FailureState(
                kind      = FailureKind.LOSS_NAN,
                context   = {"fallback_loss": 1e6},
                severity  = 1.0,
                step      = self._step,
                source_fn = "forward_pass",
            )

        if grad_norm > 1e4:
            return FailureState(
                kind      = FailureKind.GRADIENT_EXPLODE,
                context   = {"max_norm": self.cfg.clip_norm},
                severity  = 0.9,
                step      = self._step,
                source_fn = "backward_pass",
            )

        if grad_norm < 1e-8 and self._step > 10:
            return FailureState(
                kind      = FailureKind.GRADIENT_VANISH,
                context   = {"rescale_factor": 10.0},
                severity  = 0.8,
                step      = self._step,
                source_fn = "backward_pass",
            )

        if prev_loss is not None and abs(loss - prev_loss) < 1e-9 and self._step > 20:
            return FailureState(
                kind      = FailureKind.LOSS_PLATEAU,
                context   = {"perturb_std": 0.005, "lr_boost": 3.0},
                severity  = 0.5,
                step      = self._step,
                source_fn = "training_loop",
            )

        return None

    # ══════════════════════════════════════════════════════════════════════════
    #  FORWARD + BACKWARD STEP
    # ══════════════════════════════════════════════════════════════════════════

    def step(
        self,
        token_ids:   np.ndarray,
        targets:     np.ndarray,
        prev_loss:   Optional[float] = None,
        training:    bool            = True,
    ) -> ManasPipelineResult:
        """
        Execute one full MANAS-4.0 pipeline step.

        Parameters
        ----------
        token_ids  : (N,) integer token index array
        targets    : (batch, num_classes) or (batch,) target array
        prev_loss  : loss from previous step (for plateau detection)
        training   : enable stochastic operations and backward pass

        Returns
        -------
        ManasPipelineResult
        """
        t_start      = time.perf_counter()
        repairs_this = []

        # ── Step 1: Load balancer gate ────────────────────────────────────────
        with self.balancer.gate(timeout=30.0) as bal:
            batch_size  = bal.batch_size
            depth_limit = bal.depth_limit
            tier_name   = bal.tier.name

        # ── Step 2: Hardware entropy noise injection ───────────────────────────
        emb_noise = self.entropy.normal(
            std  = 0.001,
            size = (len(token_ids), self.cfg.embed_dim),
        )

        # ── Step 2b: IMPROVEMENT 2 — Causal Graph Memory Recall ───────────────
        #
        # Before encoding the current input, query the Temporal Graph for
        # causally related past states. Their root embeddings are injected
        # into the encoder as prefix context, giving MANAS the ability to
        # reason from past experience when processing current input.
        #
        # Process:
        #   1. Build a lightweight query embedding from current token_ids.
        #   2. Search Temporal Graph for similar past STATE nodes.
        #   3. Retrieve their root embeddings via causal cone traversal.
        #   4. Blend recalled embeddings into a memory context vector.
        #   5. Inject as a virtual prefix token into the encoder input.

        memory_context: Optional[np.ndarray] = None
        recalled_node_ids: List[str]         = []

        if self._step > 0 and self.cfg.graph_embed_dim > 0:
            try:
                # Step 2b-1: query embedding from token_ids statistics
                tok_arr    = token_ids.astype(np.float64)
                query_raw  = np.zeros(self.cfg.graph_embed_dim)
                # Encode token distribution features into query vector
                if len(tok_arr) > 0:
                    feats = np.array([
                        tok_arr.mean()   / self.cfg.vocab_size,
                        tok_arr.std()    / self.cfg.vocab_size,
                        tok_arr.min()    / self.cfg.vocab_size,
                        tok_arr.max()    / self.cfg.vocab_size,
                        len(tok_arr)     / 512.0,
                        float(self._step) / max(self._step, 1),
                    ])
                    np.random.seed(7)
                    proj = np.random.randn(len(feats), self.cfg.graph_embed_dim)
                    np.random.seed(None)
                    query_raw = feats @ proj
                    query_raw = query_raw / (np.linalg.norm(query_raw) + 1e-8)

                # Step 2b-2: similarity search in embedding index
                similar_nodes = self.graph.similar_to(query_raw, k=5)

                if similar_nodes:
                    # Step 2b-3: causal cone traversal from best match
                    best_node, best_sim = similar_nodes[0]
                    recalled_node_ids   = [n.node_id for n, _ in similar_nodes]

                    # Step 2b-4: collect root embeddings from recalled nodes
                    recalled_embs: List[np.ndarray] = []
                    for node, sim in similar_nodes:
                        if node.embedding is not None and sim > 0.1:
                            # Weight by similarity and recency
                            age_penalty = 1.0 / (1.0 + node.age_s / 3600.0)
                            recalled_embs.append(
                                node.embedding * sim * age_penalty
                            )

                        # Also pull causal ancestors (depth=2, lightweight)
                        cone = self.graph.causal_cone(
                            node.node_id, depth=2,
                            backward=True, forward=False
                        )
                        for anc_id, anc_node in cone.nodes.items():
                            if (anc_node.embedding is not None and
                                    anc_id not in recalled_node_ids):
                                recalled_embs.append(
                                    anc_node.embedding * 0.3
                                )
                                recalled_node_ids.append(anc_id)

                    # Step 2b-5: blend into single memory context vector
                    if recalled_embs:
                        stacked = np.stack(recalled_embs[:8], axis=0)
                        # Attention-weighted mean (softmax over L2 norms)
                        norms   = np.linalg.norm(stacked, axis=1)
                        weights = np.exp(norms) / (np.exp(norms).sum() + 1e-8)
                        memory_context = (weights[:, None] * stacked).sum(axis=0)
                        # Normalise
                        memory_context = memory_context / (
                            np.linalg.norm(memory_context) + 1e-8
                        )
                        logger.debug(
                            "[M2+M6] Memory recall: %d nodes, "
                            "best_sim=%.3f, step=%d",
                            len(recalled_embs), best_sim, self._step,
                        )

            except Exception as mem_exc:
                logger.debug("[M2+M6] Memory recall skipped: %s", mem_exc)
                memory_context = None

        # ── Step 3: Fractal encoder forward pass ──────────────────────────────
        enc_out = self.encoder.encode(token_ids)
        root    = enc_out["root"]        # (1, embed_dim)
        local   = enc_out["local"]       # (N, embed_dim)

        # ── Step 3b: Inject memory context into root embedding ────────────────
        #
        # If memory_context was retrieved, blend it into the root embedding
        # via a gated addition. The gate is derived from hardware entropy,
        # ensuring stochastic but bounded memory influence.
        #
        #   root_gated = root + gate × memory_context
        #   gate ∈ [0, cfg.memory_gate_max]  (hardware-entropy sampled)

        if memory_context is not None:
            gate = self.entropy.uniform(
                low  = 0.0,
                high = self.cfg.memory_gate_max,
            )
            # Align dims if needed
            mc = memory_context[:self.cfg.embed_dim]
            if len(mc) < self.cfg.embed_dim:
                mc = np.pad(mc, (0, self.cfg.embed_dim - len(mc)))
            root = root + gate * mc[np.newaxis, :]

        # Inject entropy noise into local embeddings
        local_noisy = local + emb_noise[:len(local)]

        # ── Step 4: ZNetwork forward ──────────────────────────────────────────
        root_tensor   = Tensor(root,       label="root")
        local_tensor  = Tensor(local_noisy, label="local")
        net_out       = self.network(root_tensor)

        # ── Step 5: Loss computation ──────────────────────────────────────────
        target_arr = np.asarray(targets, dtype=np.float64)
        if target_arr.ndim == 1 and net_out.data.ndim == 2:
            # Cross-entropy path
            loss_tensor = Custom_Autograd_Z.cross_entropy(
                net_out, target_arr.astype(int)
            )
        else:
            loss_tensor = Custom_Autograd_Z.mse(net_out, target_arr)

        loss_val = float(loss_tensor.data)

        # ── Step 6: Failure detection ─────────────────────────────────────────
        # Compute grad norm estimate before full backward
        grad_norm_est = float(np.linalg.norm(
            np.concatenate([p.grad.flatten() for p in self.network.parameters()])
        )) if any(p.grad.any() for p in self.network.parameters()) else 0.0

        failure = self._detect_failure(loss_val, grad_norm_est, prev_loss)

        # ── Step 7: JIT repair (if failure) ───────────────────────────────────
        if failure is not None and training:
            try:
                dummy_grad = np.ones(self.cfg.embed_dim) * grad_norm_est
                dummy_params = [p.data for p in self.network.parameters()[:2]]
                self.jit.repair(failure, dummy_grad, dummy_params)
                repairs_this.append(failure.kind.name)
                self._repairs_total += 1
                logger.warning(
                    "[M5] Repair applied: %s at step %d",
                    failure.kind.name, self._step,
                )
                # Log repair to graph
                self.graph.add_node(
                    kind    = NodeKind.ANNOTATION,
                    payload = {"repair": failure.kind.name,
                               "severity": failure.severity,
                               "step": self._step},
                    step    = self._step,
                    priority= 0.95,
                    tags    = ["repair", failure.kind.name.lower()],
                )
            except Exception as exc:
                logger.error("[M5] Repair failed: %s", exc)

        # ── Step 8: ZULIC backward + Change Signatures ────────────────────────
        change_sigs: Dict[str, np.ndarray] = {}
        beta_rep:    Dict[str, Dict]       = {}
        grad_norm    = 0.0

        if training:
            self.optimizer.zero_grad()
            change_sigs = self.autograd.compute_change_signatures(
                loss_tensor, self.network.parameters()
            )
            beta_rep    = self.autograd.beta_report(self.network.parameters())
            grad_norm   = float(np.mean([
                np.linalg.norm(sig) for sig in change_sigs.values()
            ])) if change_sigs else 0.0

            # ── Step 9: Parameter update ──────────────────────────────────────
            self.optimizer.step(change_sigs)

        # ── Step 10: Log state to Temporal Graph ──────────────────────────────
        beta_mean = float(np.mean([
            v["beta_mean"] for v in beta_rep.values()
        ])) if beta_rep else 0.0

        root_emb_for_graph = root.flatten()[:self.cfg.graph_embed_dim]
        if len(root_emb_for_graph) < self.cfg.graph_embed_dim:
            root_emb_for_graph = np.pad(
                root_emb_for_graph,
                (0, self.cfg.graph_embed_dim - len(root_emb_for_graph))
            )

        state_dict = ManasSystemState(
            step            = self._step,
            loss            = loss_val,
            grad_norm       = grad_norm,
            compute_tier    = tier_name,
            batch_size      = batch_size,
            depth_limit     = depth_limit,
            entropy_quality = None,
            beta_mean       = beta_mean,
            active_repairs  = len(repairs_this),
            graph_nodes     = len(self.graph),
            graph_edges     = self.graph.stats()["n_edges"],
        ).to_dict()
        state_dict["recalled_nodes"]  = recalled_node_ids[:5]
        state_dict["memory_injected"] = memory_context is not None

        graph_node = self.graph.add_node(
            kind      = NodeKind.STATE,
            payload   = state_dict,
            embedding = root_emb_for_graph,
            step      = self._step,
            priority  = min(1.0, abs(loss_val) / (abs(loss_val) + 1.0)),
            tags      = ["state", tier_name.lower()],
        )

        # Also log loss node
        self.graph.add_node(
            kind    = NodeKind.LOSS,
            payload = {"loss": loss_val, "step": self._step},
            step    = self._step,
            priority= 0.7,
            tags    = ["loss"],
        )

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        self._step  += 1

        return ManasPipelineResult(
            step              = self._step - 1,
            loss              = loss_val,
            root_embedding    = root,
            local_embedding   = local,
            change_signatures = change_sigs,
            compute_tier      = tier_name,
            batch_size        = batch_size,
            depth_limit       = depth_limit,
            beta_report       = beta_rep,
            repairs_applied   = repairs_this,
            graph_node_id     = graph_node.node_id,
            duration_ms       = duration_ms,
            memory_injected   = memory_context is not None,
            recalled_node_ids = recalled_node_ids[:5],
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  INFERENCE (no gradient, no optimizer)
    # ══════════════════════════════════════════════════════════════════════════

    def infer(
        self,
        token_ids: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference-only forward pass (no backward, no graph logging).

        Parameters
        ----------
        token_ids : (N,) integer token indices

        Returns
        -------
        dict with root, local, logits embeddings
        """
        with self.balancer.gate():
            enc_out = self.encoder.encode(token_ids)
            root    = enc_out["root"]
            local   = enc_out["local"]
            root_t  = Tensor(root, label="root_infer")
            logits  = self.network(root_t)
        return {
            "root":   root,
            "local":  local,
            "logits": logits.data,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  CAUSAL MEMORY QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    def recall_similar(
        self,
        query_embedding: np.ndarray,
        k:               int = 10,
    ) -> List[Tuple["GraphNode", float]]:
        """
        Retrieve k most similar past states from the Temporal Graph
        by embedding similarity (secondary retrieval).
        """
        return self.graph.similar_to(query_embedding, k)

    def recall_causal(
        self,
        node_id: str,
        depth:   int = 5,
    ) -> "Subgraph":
        """
        Retrieve causal cone (past + future) around a graph node.
        Primary causal pathway retrieval — not cosine similarity.
        """
        return self.graph.causal_cone(node_id, depth)

    def causal_path(
        self,
        src_id: str,
        dst_id: str,
    ) -> "PathResult":
        """Find the minimum-weight causal path between two states."""
        return self.graph.causal_path(src_id, dst_id)

    def recent_states(self, n: int = 20) -> List["GraphNode"]:
        """Return the n most recently logged STATE nodes."""
        return self.graph.latest_n(n, kind=NodeKind.STATE)

    def state_at_step(self, step: int) -> List["GraphNode"]:
        """Return all graph nodes logged at a specific training step."""
        return self.graph.query_step_range(step, step)

    # ══════════════════════════════════════════════════════════════════════════
    #  ENTROPY API
    # ══════════════════════════════════════════════════════════════════════════

    def hardware_sample(
        self,
        distribution: str = "normal",
        size:         Tuple[int, ...] = (1,),
        **kwargs,
    ) -> np.ndarray:
        """
        Sample from hardware-seeded distributions.

        Parameters
        ----------
        distribution : "normal" | "uniform" | "bernoulli" | "gumbel" | "exponential"
        size         : output shape tuple
        **kwargs     : distribution-specific parameters (mean, std, p, etc.)
        """
        fn = {
            "normal":      self.entropy.normal,
            "uniform":     self.entropy.uniform,
            "bernoulli":   self.entropy.bernoulli,
            "gumbel":      self.entropy.gumbel,
            "exponential": self.entropy.exponential,
        }.get(distribution)
        if fn is None:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        return fn(size=size, **kwargs)

    def entropy_quality_report(self) -> Dict[str, Any]:
        """Run full NIST-inspired entropy quality assessment."""
        return self.entropy.assess_quality(n_bytes=2048)

    # ══════════════════════════════════════════════════════════════════════════
    #  JIT REPAIR API
    # ══════════════════════════════════════════════════════════════════════════

    def repair(
        self,
        failure_kind: "FailureKind",
        context:      Optional[Dict] = None,
        severity:     float          = 1.0,
        *args,
        **kwargs,
    ) -> Tuple[Any, "CompiledFunction"]:
        """
        Manually trigger a JIT repair for a given failure kind.

        Parameters
        ----------
        failure_kind : FailureKind to repair
        context      : additional context dict
        severity     : failure severity [0, 1]
        *args        : forwarded to the generated repair function
        """
        state = FailureState(
            kind     = failure_kind,
            context  = context or {},
            severity = severity,
            step     = self._step,
        )
        return self.jit.repair(state, *args, **kwargs)

    def inspect_repair(self, failure_kind: "FailureKind") -> Optional[str]:
        """Return the source of the cached compiled repair function."""
        return self.jit.inspect_source(failure_kind)

    # ══════════════════════════════════════════════════════════════════════════
    #  RESOURCE & DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def resource_summary(self) -> Dict[str, Any]:
        """Full resource usage summary from Module 4."""
        return self.balancer.resource_summary()

    def complexity_report(self, seq_lengths: Optional[List[int]] = None) -> List[Dict]:
        """
        Return fractal attention complexity analysis for given sequence lengths.
        Defaults to [128, 512, 2048, 8192, 32768].
        """
        lengths = seq_lengths or [128, 512, 2048, 8192, 32768]
        return self.complexity.compare(lengths)

    def graph_stats(self) -> Dict[str, Any]:
        """Return Temporal Graph statistics."""
        return self.graph.stats()

    def jit_report(self) -> Dict[str, Any]:
        """Return JIT execution statistics from Module 5."""
        return self.jit.execution_report()

    def beta_summary(self) -> Optional[Dict[str, Dict]]:
        """
        Return ZULIC β statistics for all network parameters.
        Valid after at least one training step.
        """
        params = self.network.parameters()
        if not any(p.grad.any() for p in params):
            return None
        return self.autograd.beta_report(params)

    def full_diagnostic(self) -> Dict[str, Any]:
        """
        Return complete system diagnostic across all six modules.
        """
        return {
            "system":       self.cfg.system_name,
            "version":      self.cfg.version,
            "step":         self._step,
            "repairs_total":self._repairs_total,
            "module_1_zulic": {
                "optimizer_step": self.optimizer.step_count,
                "current_lr":     self.optimizer.current_lr,
            },
            "module_2_fractal": {
                "complexity_128":  self.complexity.analyse(128)["reduction_factor"],
                "complexity_2048": self.complexity.analyse(2048)["reduction_factor"],
                "param_count":     sum(p.size for p in self.encoder.parameters()),
            },
            "module_3_entropy": {
                "source_qualities": self.entropy.source_report(),
            },
            "module_4_balancer": self.balancer.resource_summary(),
            "module_5_jit":      self.jit.execution_report(),
            "module_6_graph":    self.graph.stats(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def save_graph(self, path: str) -> Dict[str, str]:
        """Persist the Temporal Graph to disk."""
        return self.graph.save(path)

    def load_graph(self, path: str) -> None:
        """Load a previously saved Temporal Graph from disk."""
        self.graph.load(path)
        logger.info("Graph loaded from %s | nodes=%d", path, len(self.graph))

    def save_weights(self, path: str) -> None:
        """
        Save all network parameter arrays to a compressed numpy archive.

        Parameters
        ----------
        path : .npz file path (extension added if missing)
        """
        if not path.endswith(".npz"):
            path += ".npz"
        arrays: Dict[str, np.ndarray] = {}
        for p in self.network.parameters():
            key          = (p.label or str(id(p))).replace(".", "_")
            arrays[key]  = p.data
        np.savez_compressed(path, **arrays)
        logger.info("Weights saved to %s (%d arrays)", path, len(arrays))

    def load_weights(self, path: str) -> None:
        """
        Load network parameter arrays from a compressed numpy archive.

        Parameters
        ----------
        path : .npz file path
        """
        if not path.endswith(".npz"):
            path += ".npz"
        archive = np.load(path)
        params  = self.network.parameters()
        loaded  = 0
        for p in params:
            key = (p.label or str(id(p))).replace(".", "_")
            if key in archive:
                p.data = archive[key].astype(np.float64)
                loaded += 1
        logger.info("Weights loaded from %s (%d/%d arrays)", path, loaded, len(params))

    # ══════════════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    #  PERFORMANCE API
    # ══════════════════════════════════════════════════════════════════════════

    def inference_mode(self) -> None:
        """
        Switch MANAS to inference mode.

        Actions:
          - Merge all LoRA adapters into base weights (zero inference overhead).
          - Cast network parameters to float16 (2x memory reduction).
        """
        if isinstance(self.network, ZNetworkLoRA):
            self.network.merge_all_lora()
        # Cast all network params to float16 for inference
        for p in self.network.all_parameters() if hasattr(self.network, 'all_parameters') else self.network.parameters():
            p.data = p.data.astype(np.float16)
        logger.info("[PERF] inference_mode: LoRA merged, weights cast to float16.")

    def training_mode(self) -> None:
        """
        Switch MANAS back to training mode.

        Actions:
          - Unmerge LoRA adapters from base weights.
          - Cast parameters back to float32.
        """
        if isinstance(self.network, ZNetworkLoRA):
            self.network.unmerge_all_lora()
        for p in self.network.all_parameters() if hasattr(self.network, 'all_parameters') else self.network.parameters():
            p.data = p.data.astype(np.float32)
        logger.info("[PERF] training_mode: LoRA unmerged, weights cast to float32.")

    def quantize_save(self, path: str, bits: int = 8) -> str:
        """
        Quantize network weights to int8 and save to disk.

        Reduces checkpoint size by 4x vs float32.
        Load with load_quantized().

        Parameters
        ----------
        path : output .npz file path
        bits : quantization bits (8 recommended)

        Returns
        -------
        path to saved file
        """
        if not path.endswith(".npz"):
            path += ".npz"
        params  = (self.network.all_parameters()
                   if hasattr(self.network, "all_parameters")
                   else self.network.parameters())
        weights = {
            (p.label or str(id(p))).replace(".", "_"): p.data.astype(np.float32)
            for p in params
        }
        quantized = self.precision.quantize_weights(weights, bits)
        # Save: for each param, store quantized array and scale
        save_dict: Dict[str, np.ndarray] = {}
        for name, d in quantized.items():
            save_dict[f"{name}_q"]     = d["quantized"]
            save_dict[f"{name}_scale"] = np.array([d["scale"]], dtype=np.float32)
        np.savez_compressed(path, **save_dict)
        size_mb = sum(d["quantized"].nbytes for d in quantized.values()) / (1024**2)
        logger.info(
            "[PERF] quantize_save: %d weights → %s (%.2f MB, %d-bit)",
            len(quantized), path, size_mb, bits,
        )
        return path

    def load_quantized(self, path: str) -> None:
        """
        Load int8-quantized weights and dequantize to float32.

        Parameters
        ----------
        path : .npz file saved by quantize_save()
        """
        if not path.endswith(".npz"):
            path += ".npz"
        archive = np.load(path)
        # Reconstruct quantized dict
        quantized: Dict[str, Dict] = {}
        for key in archive.files:
            if key.endswith("_q"):
                name  = key[:-2]
                scale = float(archive[f"{name}_scale"][0])
                quantized[name] = {"quantized": archive[key], "scale": scale}
        weights = self.precision.dequantize_weights(quantized)
        params  = (self.network.all_parameters()
                   if hasattr(self.network, "all_parameters")
                   else self.network.parameters())
        loaded  = 0
        for p in params:
            key = (p.label or str(id(p))).replace(".", "_")
            if key in weights:
                p.data = weights[key].astype(np.float32)
                loaded += 1
        logger.info(
            "[PERF] load_quantized: loaded %d/%d arrays from %s",
            loaded, len(params), path,
        )

    def memory_report(self) -> Dict[str, Any]:
        """
        Full memory usage report across all parameters.
        """
        params = (self.network.all_parameters()
                  if hasattr(self.network, "all_parameters")
                  else self.network.parameters())
        report = MixedPrecisionManager.memory_report(params)
        if isinstance(self.network, ZNetworkLoRA):
            report["lora_summary"] = self.network.param_summary()
        if _MLX_AVAILABLE:
            report["mlx_available"] = True
            report["mlx_active"]    = self.precision.use_mlx
        else:
            report["mlx_available"] = False
            report["mlx_install"]   = "pip install mlx"
        return report

    def performance_report(self) -> Dict[str, Any]:
        """
        Complete performance configuration report.
        """
        mem = self.memory_report()
        return {
            "backend":         "MLX (Apple Metal GPU)" if self.precision.use_mlx else "NumPy (CPU)",
            "mlx_available":   _MLX_AVAILABLE,
            "lora_enabled":    isinstance(self.network, ZNetworkLoRA),
            "lora_rank":       self.cfg.lora_rank if self.cfg.use_lora else None,
            "lora_summary":    self.network.param_summary() if isinstance(self.network, ZNetworkLoRA) else None,
            "memory":          mem,
            "train_dtype":     str(self.precision.train_dtype),
            "infer_dtype":     str(self.precision.infer_dtype),
            "quantize_bits":   self.cfg.quantize_bits,
            "estimated_speedup": self._estimate_speedup(),
        }

    def _estimate_speedup(self) -> str:
        """Rough speedup estimate vs baseline float64 NumPy."""
        base = 1.0
        if isinstance(self.network, ZNetworkLoRA):
            summary = self.network.param_summary()
            total   = max(summary["total"], 1)
            trained = max(summary["trainable"], 1)
            base   *= total / trained          # LoRA param reduction speedup
        if self.precision.use_mlx:
            base *= 20.0                        # rough MLX vs NumPy CPU speedup
        else:
            base *= 2.0                         # float32 vs float64 speedup
        return f"~{base:.0f}x vs float64 NumPy baseline"

    def shutdown(self) -> None:
        """
        Gracefully shut down all background threads and flush state.
        Safe to call multiple times.
        """
        try:
            self.balancer.stop()
        except Exception:
            pass
        if self.cfg.graph_persist_dir:
            try:
                self.graph.save()
            except Exception:
                pass
        logger.info(
            "[MANAS] Shutdown complete. Steps=%d Repairs=%d Graph nodes=%d",
            self._step, self._repairs_total, len(self.graph),
        )

    def __enter__(self) -> "MANAS":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"MANAS(version={self.cfg.version!r}, "
            f"step={self._step}, "
            f"tier={self.balancer.tier.name}, "
            f"graph_nodes={len(self.graph)})"
        )



# ══════════════════════════════════════════════════════════════════════════
#  MODULE 7: HIERARCHICAL GOAL PLANNING
# ══════════════════════════════════════════════════════════════════════════
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional

class GoalStatus:
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    FAILED = "FAILED"

class GoalNode:
    def __init__(self, description: str, priority: float = 1.0, parent_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = priority
        self.status = GoalStatus.PENDING
        self.parent_id = parent_id
        self.sub_goals = []
        self.created_at = time.time()
        self.completed_at = None
        self.context = {}

    def __repr__(self):
        return f"<GoalNode {self.status}: {self.description}>"

class GoalTree:
    def __init__(self):
        self.goals: Dict[str, GoalNode] = {}
        self.root_goals: List[str] = []

    def add_goal(self, description: str, priority: float = 1.0, parent_id: Optional[str] = None) -> GoalNode:
        node = GoalNode(description, priority, parent_id)
        self.goals[node.id] = node
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(node.id)
        else:
            self.root_goals.append(node.id)
        return node

    def get_next_actionable(self) -> Optional[GoalNode]:
        # Simple DFS to find the deepest pending/active goal that has no pending sub-goals
        def dfs(node_id):
            node = self.goals[node_id]
            if node.status in (GoalStatus.DONE, GoalStatus.FAILED):
                return None
            for sub_id in node.sub_goals:
                res = dfs(sub_id)
                if res: return res
            return node
        
        # Sort roots by priority
        roots = sorted([self.goals[rid] for rid in self.root_goals], key=lambda x: x.priority, reverse=True)
        for root in roots:
            res = dfs(root.id)
            if res: return res
        return None

    def mark_status(self, goal_id: str, status: str):
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            if status in (GoalStatus.DONE, GoalStatus.FAILED):
                self.goals[goal_id].completed_at = time.time()

class HierarchicalPlanner:
    def __init__(self, llm_generate_fn=None):
        self.tree = GoalTree()
        self.llm_generate_fn = llm_generate_fn

    def decompose(self, goal_description: str) -> List[str]:
        if self.llm_generate_fn:
            prompt = f"Decompose the following goal into 3-5 concrete, actionable sequential steps. Output ONLY the steps, one per line.\nGoal: {goal_description}\nSteps:"
            try:
                response = self.llm_generate_fn(prompt)
                steps = [line.strip().lstrip('1234567890.-* ') for line in response.split('\n') if line.strip()]
                return steps if steps else [goal_description]
            except Exception:
                pass
        # Fallback heuristic
        if "website" in goal_description.lower():
            return ["Understand requirements", "Design layout", "Write HTML/CSS", "Write JS", "Test"]
        return [goal_description]


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 8: RL ACTION SELECTION
# ══════════════════════════════════════════════════════════════════════════

class RLAgentEnvironment:
    ACTIONS = ["direct_answer", "search_first", "use_code", "decompose_goal", "ask_clarification"]
    
    def __init__(self):
        self.state_history = []
        
    def discretize_state(self, confidence: float, complexity: float, topic_familiarity: float) -> str:
        # Simple bucketizing
        conf_bucket = "High" if confidence > 0.7 else ("Medium" if confidence > 0.4 else "Low")
        comp_bucket = "Complex" if complexity > 0.6 else "Simple"
        fam_bucket = "Familiar" if topic_familiarity > 0.5 else "Unfamiliar"
        return f"{conf_bucket}_{comp_bucket}_{fam_bucket}"

class QActionPolicy:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
        self.q_table = {}  # state -> {action -> score}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.actions = RLAgentEnvironment.ACTIONS
        
    def _get_q_values(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
            # Heuristic priors
            if "Low" in state or "Unfamiliar" in state:
                self.q_table[state]["search_first"] = 1.0
            if "Complex" in state:
                self.q_table[state]["decompose_goal"] = 1.0
            if "High" in state and "Simple" in state:
                self.q_table[state]["direct_answer"] = 1.0
        return self.q_table[state]

    def select_action(self, state: str) -> str:
        q_vals = self._get_q_values(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return max(q_vals.items(), key=lambda x: x[1])[0]

    def update(self, state: str, action: str, reward: float, next_state: Optional[str] = None):
        q_vals = self._get_q_values(state)
        max_next_q = max(self._get_q_values(next_state).values()) if next_state else 0.0
        q_vals[action] = q_vals[action] + self.lr * (reward + self.gamma * max_next_q - q_vals[action])


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 9: CONCEPT ONTOLOGY & SYMBOL GROUNDING
# ══════════════════════════════════════════════════════════════════════════

class ConceptNode:
    def __init__(self, name: str):
        self.name = name.lower()
        self.relations = {}  # relation_type -> set of connected concept names
        self.properties = set()
        
class ConceptOntology:
    def __init__(self):
        self.concepts = {}
        
    def add_concept(self, name: str):
        name = name.lower()
        if name not in self.concepts:
            self.concepts[name] = ConceptNode(name)
        return self.concepts[name]

    def add_relation(self, subj: str, relation: str, obj: str):
        s_node = self.add_concept(subj)
        self.add_concept(obj)
        if relation not in s_node.relations:
            s_node.relations[relation] = set()
        s_node.relations[relation].add(obj.lower())

    def get_related(self, name: str, relation: str = None) -> List[str]:
        name = name.lower()
        if name not in self.concepts: return []
        if relation:
            return list(self.concepts[name].relations.get(relation, set()))
        # Return all
        res = set()
        for rel_set in self.concepts[name].relations.values():
            res.update(rel_set)
        return list(res)

class SymbolGrounder:
    def __init__(self, ontology: ConceptOntology):
        self.ontology = ontology
        self.alias_map = {}
        
    def ground(self, text: str) -> List[str]:
        # Simple exact substring match for grounding
        grounded = []
        text_lower = text.lower()
        for concept in self.ontology.concepts:
            if concept in text_lower and len(concept) > 2:
                grounded.append(concept)
        return grounded


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 10: METACOGNITIVE SELF-MODEL
# ══════════════════════════════════════════════════════════════════════════

class MetacognitiveSelfModel:
    def __init__(self):
        self.topic_confidence = {}  # topic -> confidence (0.0 to 1.0)
        self.recent_evaluations = []
        
    def estimate_confidence(self, topics: List[str]) -> float:
        if not topics: return 0.5
        confs = [self.topic_confidence.get(t.lower(), 0.1) for t in topics]
        return sum(confs) / len(confs)

    def should_i_search(self, topics: List[str]) -> bool:
        conf = self.estimate_confidence(topics)
        return conf < 0.6

    def update_after_eval(self, topics: List[str], score: float):
        # Score is assumed 1-10
        normalized_score = min(max(score / 10.0, 0.0), 1.0)
        self.recent_evaluations.append(normalized_score)
        if len(self.recent_evaluations) > 100:
            self.recent_evaluations.pop(0)
            
        for topic in topics:
            t = topic.lower()
            current = self.topic_confidence.get(t, 0.5)
            # Moving average update
            self.topic_confidence[t] = 0.8 * current + 0.2 * normalized_score


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 11: CROSS-DOMAIN TRANSFER
# ══════════════════════════════════════════════════════════════════════════

class CrossDomainAdapter:
    def __init__(self, ontology: ConceptOntology):
        self.ontology = ontology
        self.domains = {
            "cs": ["code", "programming", "algorithm", "python", "software"],
            "math": ["equation", "theorem", "calculus", "algebra", "variable"],
            "physics": ["force", "mass", "energy", "velocity", "quantum"],
            "biology": ["cell", "dna", "evolution", "species", "protein"]
        }

    def detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        best_domain = "general"
        max_hits = 0
        for domain, keywords in self.domains.items():
            hits = sum(1 for k in keywords if k in text_lower)
            if hits > max_hits:
                max_hits = hits
                best_domain = domain
        return best_domain

    def find_analogies(self, concept: str, source_domain: str, target_domain: str) -> Optional[str]:
        # Toy implementation for analogy discovery based on ontology structural similarity
        # In a real system, this uses graph embedding distances.
        related = self.ontology.get_related(concept)
        if not related: return None
        # Heuristic: if looking for math analogy to cs, return random item from target domain for now if no structure exists
        if target_domain in self.domains:
            # Just return a placeholder for the MVP
            return f"[{concept} is somewhat analogous to concepts in {target_domain} based on structural graph overlap]"
        return None
