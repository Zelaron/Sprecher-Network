# sn_core/model.py

"""Sprecher Network model components: splines, layers, and networks."""

import torch
import torch.nn as nn
import numpy as np
from .config import CONFIG, Q_VALUES_FACTOR


class Interval:
    """Interval arithmetic for optimal domain computation."""
    def __init__(self, min_val, max_val):
        self.min = float(min_val)
        self.max = float(max_val)
        assert self.min <= self.max, f"Invalid interval: [{self.min}, {self.max}]"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.min + other, self.max + other)
        elif isinstance(other, Interval):
            return Interval(self.min + other.min, self.max + other.max)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar >= 0:
                return Interval(self.min * scalar, self.max * scalar)
            else:
                return Interval(self.max * scalar, self.min * scalar)
        else:
            return NotImplemented
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.min - other, self.max - other)
        elif isinstance(other, Interval):
            return Interval(self.min - other.max, self.max - other.min)
        else:
            return NotImplemented
    
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar > 0:
                return Interval(self.min / scalar, self.max / scalar)
            elif scalar < 0:
                return Interval(self.max / scalar, self.min / scalar)
            else:
                raise ValueError("Division by zero")
        else:
            return NotImplemented
    
    def union(self, other):
        """Union of two intervals."""
        return Interval(min(self.min, other.min), max(self.max, other.max))
    
    def __repr__(self):
        return f"[{self.min:.6f}, {self.max:.6f}]"


class TheoreticalRange:
    """Tracks theoretical min/max values for a layer's input/output."""
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def __repr__(self):
        return f"TheoreticalRange({self.min:.3f}, {self.max:.3f})"


class PhiCodomainParams(nn.Module):
    """
    Trainable codomain parameters for Φ splines.
    The codomain is defined via a center and a radius:
      codomain: (cc - cr, cc + cr)
    """
    def __init__(self, cc=0.0, cr=10.0):
        super().__init__()
        self.cc = nn.Parameter(torch.tensor(cc, dtype=torch.float32))  # cc: codomain center for Φ
        self.cr = nn.Parameter(torch.tensor(cr, dtype=torch.float32))  # cr: codomain radius for Φ


class SimpleSpline(nn.Module):
    """
    A spline function (Piecewise Linear or Cubic B-Spline).
    
    Order is determined by CONFIG['spline_order']:
      - 1: Piecewise Linear (coeffs correspond to values at knots)
      - 3: Cubic B-Spline (coeffs are control points)

    For φ splines: domain is dynamically updated, codomain is fixed [0,1]
    For Φ splines: domain is dynamically updated, codomain can be fixed or trainable
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False,
                 train_codomain=False, codomain_params=None):
        super().__init__()
        self.num_knots = num_knots
        self.monotonic = monotonic
        self.train_codomain = train_codomain
        self.codomain_params = codomain_params
        self.spline_order = CONFIG.get('spline_order', 1)
        
        # Store initial ranges
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        
        # Create knots buffer - will be updated dynamically
        # These define the grid points of the domain
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        # Domain violation tracking
        self.domain_violations = 0
        self.total_evaluations = 0
        
        # Determine number of coefficients
        # PWL (order 1): num_coeffs = num_knots
        # Cubic (order 3): num_coeffs = num_knots + 2 (for G intervals, G+3 basis funcs)
        # General: num_coeffs = num_knots + order - 1
        self.num_coeffs = num_knots + self.spline_order - 1
        
        # Initialize spline coefficients
        torch.manual_seed(CONFIG['seed'])
        if monotonic:
            # For monotonic splines, use log-space parameters to ensure monotonicity
            # Note: Monotonic control points imply monotonic B-spline
            self.log_increments = nn.Parameter(torch.zeros(self.num_coeffs))
            with torch.no_grad():
                # Initialize to approximate linear function
                self.log_increments.data = torch.log(torch.ones(self.num_coeffs) / self.num_coeffs + 1e-6)
        else:
            # For general splines (Φ), initialize to zeros
            self.coeffs = nn.Parameter(torch.zeros(self.num_coeffs))

    def update_domain(self, new_range, allow_resampling=True, force_resample=False):
        """
        Update the domain of the spline by adjusting knot positions.
        
        For non-monotonic splines (Φ), this method resamples the spline to preserve
        its learned functional shape in the new domain.
        
        For Cubic splines, resampling includes a sharpening step to correct for
        the difference between control points and function values.
        """
        with torch.no_grad():
            new_min, new_max = new_range
            
            if abs(self.in_min - new_min) < 1e-6 and abs(self.in_max - new_max) < 1e-6:
                return

            # --- Spline Resampling to Preserve Learned Shape (for Φ splines) ---
            if not self.monotonic and (self.total_evaluations > 0 or force_resample) and allow_resampling:
                # 1. Create a temporary evaluator for the OLD spline
                old_knots = self.knots.clone()
                old_coeffs = self.get_coeffs().clone()
                
                # Closure to evaluate the frozen old spline state
                def old_spline_eval_closure(x_vals):
                    return self._evaluate_spline_raw(x_vals, old_knots, old_coeffs)

                # 2. Define new knot positions (the grid)
                new_knots_pos = torch.linspace(new_min, new_max, self.num_knots, 
                                               device=self.knots.device, dtype=self.knots.dtype)
                
                # 3. Sample the old spline at the new grid positions
                # Note: We need enough points to determine the new coefficients.
                # For PWL, sampling at knots is exact.
                # For Cubic, we sample at knots and approximate/solve for coeffs.
                
                # To determine N+2 cubic coeffs, we need N+2 values or appropriate BCs.
                # We sample at the N grid points (knots).
                # For the extra 2 coeffs (boundary handling), the linear extrapolation inside
                # `_evaluate_spline_raw` handles the extended grid implicitly if we sample 
                # points slightly outside or just use the N grid points and extend coeffs.
                
                # Strategy: Sample at the N grid points, then compute coefficients.
                # For Cubic B-splines, c_i ≈ y_i - 1/6 * (y_{i-1} - 2y_i + y_{i+1}) (Sharpening)
                
                if self.spline_order == 1:
                    # PWL: Coeffs are just function values at knots
                    resampled_coeffs = old_spline_eval_closure(new_knots_pos)
                else:
                    # Higher order: Quasi-interpolation / Sharpening
                    # Sample at grid points
                    y_samples = old_spline_eval_closure(new_knots_pos) # Size num_knots
                    
                    if self.spline_order == 3:
                        # Apply sharpening correction: c ≈ y - 1/6 Δ²y
                        # This reduces interpolation error from O(h^2) to O(h^4)
                        # Handle boundaries by replication for difference
                        y_padded = torch.cat([y_samples[:1], y_samples, y_samples[-1:]])
                        d2y = y_padded[:-2] - 2 * y_padded[1:-1] + y_padded[2:]
                        c_inner = y_samples - (1.0/6.0) * d2y
                        
                        # We need N+2 coefficients. The samples cover the "inner" grid.
                        # We simply extend the boundary coefficients linearly from the corrected inner ones.
                        # Indices: c_0 corresponds to t_{-1}, c_1 to t_0 ...
                        # c_inner has size N. Let's assign them to c_1 ... c_N.
                        
                        resampled_coeffs = torch.zeros(self.num_coeffs, device=self.coeffs.device)
                        resampled_coeffs[1:-1] = c_inner
                        
                        # Linear extension for boundary control points
                        resampled_coeffs[0] = 2 * c_inner[0] - c_inner[1]
                        resampled_coeffs[-1] = 2 * c_inner[-1] - c_inner[-2]
                        
                    else:
                        # Generic fallback for other orders: just map values directly (O(h^2) error)
                        # Pad to match size
                        pad_size = self.num_coeffs - self.num_knots
                        resampled_coeffs = torch.cat([y_samples[:1].repeat(pad_size//2),
                                                    y_samples,
                                                    y_samples[-1:].repeat(pad_size - pad_size//2)])

                # Update the learnable parameters
                self.coeffs.data = resampled_coeffs

            # --- Update the internal state for the new domain ---
            self.in_min, self.in_max = new_min, new_max

            # Add safety margin if configured
            if CONFIG.get('domain_safety_margin', 0.0) > 0:
                margin = CONFIG['domain_safety_margin'] * (self.in_max - self.in_min)
                self.in_min -= margin
                self.in_max += margin
            
            self.knots.data = torch.linspace(self.in_min, self.in_max, self.num_knots, 
                                           device=self.knots.device, dtype=self.knots.dtype)

    def initialize_as_identity(self, domain_min, domain_max):
        """Initialize coefficients to approximate identity function y=x."""
        with torch.no_grad():
            if self.monotonic:
                # For monotonic splines, uniform increments implies linear
                self.log_increments.data = torch.log(torch.ones(self.num_coeffs) / self.num_coeffs + 1e-6)
            else:
                # Setup coefficients to produce linear function y=x
                # For PWL and Uniform B-splines, setting control points on the line works.
                # c_k = x(node_k).
                # For Cubic B-spline defined on t_0...t_{N-1}, coeffs are c_0...c_{N+1}.
                # Grid spacing h. c_k corresponds to x = t_0 + (k-1)h.
                
                h = (domain_max - domain_min) / (self.num_knots - 1)
                if self.spline_order == 1:
                    vals = torch.linspace(domain_min, domain_max, self.num_knots)
                elif self.spline_order == 3:
                    # Indices 0 to N+1. c_k location: min + (k-1)*h
                    indices = torch.arange(self.num_coeffs, device=self.coeffs.device, dtype=self.coeffs.dtype)
                    vals = domain_min + (indices - 1) * h
                else:
                    # Generic linear spacing
                    indices = torch.arange(self.num_coeffs, device=self.coeffs.device, dtype=self.coeffs.dtype)
                    offset = (self.spline_order - 1) / 2.0
                    vals = domain_min + (indices - offset) * h
                    
                self.coeffs.data = vals
    
    def get_coeffs(self):
        """Get the actual spline coefficients."""
        if self.monotonic:
            # Use softplus to ensure positive increments, then cumsum
            increments = torch.nn.functional.softplus(self.log_increments)
            cumulative = torch.cumsum(increments, 0)
            # Normalize to [0, 1] range
            cumulative = cumulative / (cumulative[-1] + 1e-8)
            return cumulative
        else:
            return self.coeffs
    
    def get_actual_output_range(self):
        """Compute the actual range of outputs this spline can produce."""
        with torch.no_grad():
            coeffs = self.get_coeffs()
            
            if self.monotonic:
                return TheoreticalRange(0.0, 1.0)
            else:
                # Convex hull property: spline is contained within min/max of control points
                # This holds for B-splines of any order.
                if self.train_codomain and self.codomain_params is not None:
                    cc = self.codomain_params.cc.item()
                    cr = self.codomain_params.cr.item()
                    coeff_min = coeffs.min().item()
                    coeff_max = coeffs.max().item()
                    
                    # Transform coefficients (worst-case bounds)
                    normalized_min = (coeff_min - coeff_min) / (coeff_max - coeff_min + 1e-8) # 0
                    normalized_max = (coeff_max - coeff_min) / (coeff_max - coeff_min + 1e-8) # 1
                    
                    # Actually we just need to transform the min/max coeffs directly
                    # y = cc - cr + 2*cr * (c - c_min)/(c_max - c_min)
                    #   = cc - cr + 2*cr * norm_c
                    
                    # The range of the spline is bounded by the range of transformed coeffs
                    # Since the transform is monotonic (linear), we map min/max coeffs.
                    actual_min = cc - cr # maps to norm_c=0
                    actual_max = cc + cr # maps to norm_c=1
                    
                    return TheoreticalRange(actual_min, actual_max)
                else:
                    return TheoreticalRange(coeffs.min().item(), coeffs.max().item())
    
    def _evaluate_spline_raw(self, x, knots, coeffs):
        """
        Internal evaluator that handles different spline orders.
        Does NOT apply codomain transforms or domain violation tracking.
        """
        # Handle out-of-domain inputs by extending linearly from boundary
        # For Cubic, we use the polynomial of the first/last interval (natural extension)
        # or strictly linear. For stability and consistency with PWL, we use linear extension.
        
        x_clamped = torch.clamp(x, knots[0], knots[-1])
        below_domain = x < knots[0]
        above_domain = x > knots[-1]
        
        # Calculate grid position
        # x = knots[0] + grid_x * spacing
        knot_spacing = (knots[-1] - knots[0]) / (len(knots) - 1)
        # Avoid division by zero
        if knot_spacing < 1e-8:
            grid_x = torch.zeros_like(x)
        else:
            grid_x = (x_clamped - knots[0]) / knot_spacing
            
        if self.spline_order == 1:
            # --- Piecewise Linear ---
            idx = torch.floor(grid_x).long()
            idx = torch.clamp(idx, 0, len(knots) - 2)
            t = grid_x - idx
            
            # Interpolate
            c0 = coeffs[idx]
            c1 = coeffs[idx + 1]
            interpolated = (1 - t) * c0 + t * c1
            
            # Slope for extension
            left_slope = (coeffs[1] - coeffs[0]) / knot_spacing
            right_slope = (coeffs[-1] - coeffs[-2]) / knot_spacing
            
        elif self.spline_order == 3:
            # --- Cubic B-Spline ---
            # We use uniform B-spline matrix evaluation
            # Grid index i corresponds to interval [t_i, t_{i+1}]
            # Uses control points c_i, c_{i+1}, c_{i+2}, c_{i+3} 
            # (if we map coeffs indices appropriately)
            
            # Let's align: Interval 0 (x in [t_0, t_1]) uses c_0, c_1, c_2, c_3.
            # Our coeffs size is N+2.
            # Interval index
            idx = torch.floor(grid_x).long()
            idx = torch.clamp(idx, 0, len(knots) - 2)
            
            # Local parameter u in [0, 1]
            u = grid_x - idx
            
            # Gather 4 control points for the interval
            # indices: idx, idx+1, idx+2, idx+3
            # Check bounds: coeffs has size num_knots + 2.
            # Max idx is num_knots - 2.
            # Max access is (num_knots - 2) + 3 = num_knots + 1. 
            # This fits exactly into size num_knots + 2 (indices 0..N+1).
            
            # Unpack basis matrix multiplication: M * [c0, c1, c2, c3]^T
            # M = 1/6 * [[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]
            # Values:
            # b0 = 1/6 * (1 - u)^3
            # b1 = 1/6 * (3u^3 - 6u^2 + 4)
            # b2 = 1/6 * (-3u^3 + 3u^2 + 3u + 1)
            # b3 = 1/6 * u^3
            
            u2 = u * u
            u3 = u2 * u
            
            # Optimized calculation
            b0 = (1 - u)**3
            b1 = (3 * u3 - 6 * u2 + 4)
            b2 = (-3 * u3 + 3 * u2 + 3 * u + 1)
            b3 = u3
            
            # Get coefficients
            p0 = coeffs[idx]
            p1 = coeffs[idx + 1]
            p2 = coeffs[idx + 2]
            p3 = coeffs[idx + 3]
            
            interpolated = (b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3) / 6.0
            
            # Compute derivatives at boundaries for linear extension
            # Derivative basis:
            # db0 = -0.5 * (1-u)^2
            # db1 = 0.5 * (3u^2 - 4u)
            # db2 = 0.5 * (-3u^2 + 2u + 1)
            # db3 = 0.5 * u^2
            # Note: scaled by 1/h_grid for dx
            
            # Left boundary: idx=0, u=0
            # db0=-0.5, db1=0, db2=0.5, db3=0
            # slope = (-0.5*c0 + 0.5*c2) / h = (c2 - c0) / (2h)
            left_slope = (coeffs[2] - coeffs[0]) / (2 * knot_spacing)
            
            # Right boundary: idx=last, u=1
            # db0=0, db1=-0.5, db2=0, db3=0.5
            # slope = (-0.5*c_{N-1} + 0.5*c_{N+1}) / h
            # last idx = N-2. p1=c_{N-1}, p3=c_{N+1}
            last_idx = len(knots) - 2
            right_slope = (coeffs[last_idx + 3] - coeffs[last_idx + 1]) / (2 * knot_spacing)
            
        else:
            raise NotImplementedError(f"Spline order {self.spline_order} not implemented")
            
        # Apply linear extension
        result = torch.where(below_domain,
                           interpolated + left_slope * (x - knots[0]),
                           interpolated)
        result = torch.where(above_domain,
                           interpolated + right_slope * (x - knots[-1]),
                           result)
        
        return result

    def forward(self, x):
        x = x.to(self.knots.device)
        
        # Track domain violations
        if CONFIG.get('track_domain_violations', False):
            self.total_evaluations += x.numel()
            violations = ((x < self.in_min) | (x > self.in_max)).sum().item()
            self.domain_violations += violations
            if violations > 0 and CONFIG.get('verbose_domain_violations', False):
                print(f"Domain violation: {violations}/{x.numel()} values outside [{self.in_min:.3f}, {self.in_max:.3f}]")
        
        # Evaluate the spline (raw value)
        coeffs = self.get_coeffs()
        interpolated = self._evaluate_spline_raw(x, self.knots, coeffs)
        
        # Apply codomain logic
        if self.train_codomain and self.codomain_params is not None:
            # Apply trainable codomain scaling
            cc = self.codomain_params.cc
            cr = self.codomain_params.cr
            # Normalize based on coeffs range
            coeff_min = coeffs.min()
            coeff_max = coeffs.max()
            denom = coeff_max - coeff_min + 1e-8
            
            normalized = (interpolated - coeff_min) / denom
            result = cc - cr + 2 * cr * normalized
        else:
            result = interpolated
        
        return result
    
    def get_domain_violation_stats(self):
        """Return domain violation statistics."""
        if self.total_evaluations == 0:
            return 0.0
        return self.domain_violations / self.total_evaluations
    
    def reset_domain_violation_stats(self):
        """Reset domain violation tracking."""
        self.domain_violations = 0
        self.total_evaluations = 0
    
    def get_domain_state(self):
        """Get the complete domain state for checkpoint saving."""
        state = {
            'knots': self.knots.data.clone(),
            'in_min': self.in_min,
            'in_max': self.in_max,
            'coeffs': self.get_coeffs().clone(),
            'domain_violations': self.domain_violations,
            'total_evaluations': self.total_evaluations
        }
        
        # Save the raw parameters to ensure exact restoration
        if self.monotonic:
            state['log_increments'] = self.log_increments.data.clone()
        else:
            state['raw_coeffs'] = self.coeffs.data.clone()
            
        return state
    
    def set_domain_state(self, state):
        """Restore domain state from checkpoint."""
        self.knots.data = state['knots']
        self.in_min = state['in_min']
        self.in_max = state['in_max']
        
        if self.monotonic and 'log_increments' in state:
            self.log_increments.data = state['log_increments']
        elif not self.monotonic and 'raw_coeffs' in state:
            self.coeffs.data = state['raw_coeffs']
        
        self.domain_violations = state.get('domain_violations', 0)
        self.total_evaluations = state.get('total_evaluations', 0)


def compute_batchnorm_bounds(input_interval, norm_layer, training_mode=True):
    """
    Compute output interval after BatchNorm.
    
    Args:
        input_interval: Interval of possible input values
        norm_layer: BatchNorm1d layer
        training_mode: Whether in training mode (affects statistics used)
    
    Returns:
        Output interval after BatchNorm
    """
    eps = norm_layer.eps
    
    if training_mode:
        # Conservative standardized range during training
        k = 4.0  # Covers ~99.99% of standard normal
        standardized = Interval(-k, k)
        
        # Apply affine transformation if enabled (conservative across channels)
        if norm_layer.affine:
            weight = norm_layer.weight
            bias = norm_layer.bias
            
            weight_min = weight.min().item()
            weight_max = weight.max().item()
            bias_min = bias.min().item()
            bias_max = bias.max().item()
            
            corners = []
            for w in [weight_min, weight_max]:
                for b in [bias_min, bias_max]:
                    if w >= 0:
                        corners.append(Interval(
                            standardized.min * w + b,
                            standardized.max * w + b
                        ))
                    else:
                        corners.append(Interval(
                            standardized.max * w + b,
                            standardized.min * w + b
                        ))
            result = corners[0]
            for corner in corners[1:]:
                result = result.union(corner)
            return result
        else:
            return standardized
    else:
        # EVAL MODE: per-channel running stats (tight union across channels)
        running_mean = norm_layer.running_mean
        running_var = norm_layer.running_var
        
        a = input_interval.min
        b = input_interval.max
        
        std = torch.sqrt(running_var + eps)
        # Standardize per channel, then get per-channel intervals
        lo = (a - running_mean) / std
        hi = (b - running_mean) / std
        std_min = torch.minimum(lo, hi)
        std_max = torch.maximum(lo, hi)
        
        if norm_layer.affine:
            w = norm_layer.weight
            b_bias = norm_layer.bias
            out_min = torch.where(w >= 0, w * std_min + b_bias, w * std_max + b_bias)
            out_max = torch.where(w >= 0, w * std_max + b_bias, w * std_min + b_bias)
            return Interval(out_min.min().item(), out_max.max().item())
        else:
            return Interval(std_min.min().item(), std_max.max().item())


def compute_layernorm_bounds(input_interval, norm_layer, num_features):
    """
    Compute output interval after LayerNorm.
    
    LayerNorm normalizes across features for each sample independently,
    making exact bounds intractable. We use conservative bounds.
    
    Args:
        input_interval: Interval of possible input values per feature
        norm_layer: LayerNorm layer  
        num_features: Number of features being normalized
    
    Returns:
        Output interval after LayerNorm
    """
    eps = norm_layer.eps
    
    # Exact worst-case bound for LayerNorm
    # This occurs when one feature is at the max and all others at min (or vice versa)
    k = (num_features - 1) ** 0.5  # Exact worst-case bound
    standardized = Interval(-k, k)
    
    # Apply affine transformation
    if norm_layer.elementwise_affine:
        weight = norm_layer.weight
        bias = norm_layer.bias
        
        # Get bounds on weight and bias
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        bias_min = bias.min().item()
        bias_max = bias.max().item()
        
        # Compute output bounds
        corners = []
        for w in [weight_min, weight_max]:
            for b in [bias_min, bias_max]:
                if w >= 0:
                    corners.append(Interval(
                        standardized.min * w + b,
                        standardized.max * w + b
                    ))
                else:
                    corners.append(Interval(
                        standardized.max * w + b,
                        standardized.min * w + b
                    ))
        
        result = corners[0]
        for corner in corners[1:]:
            result = result.union(corner)
        return result
    else:
        return standardized


class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ with dynamically computed domain and fixed [0,1] codomain
      - A general spline Φ with dynamically computed domain and optionally trainable codomain
      - A trainable shift (η)
      - A trainable weight VECTOR (λ) - one weight per input dimension
      - Optional lateral mixing for cross-output communication
      - Residual connection (configurable style):
          * 'node'   (default): node-centric residuals (scalar/pooling/broadcast)
          * 'linear': standard residuals (scalar if d_in==d_out, else projection matrix)
    If is_final is True, the block sums its outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=100, Phi_knots=100,
                 phi_domain=None, Phi_domain=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final
        
        # Theoretical range tracking
        self.input_range = TheoreticalRange(0.0, 1.0) if layer_num == 0 else None
        self.output_range = None

        # NEW: per-channel input/output intervals (populated by network during updates)
        self.input_min_per_dim = None   # tensor[d_in] or None
        self.input_max_per_dim = None   # tensor[d_in] or None
        self.output_min_per_q = None    # tensor[d_out] (post-Φ+residual, pre-sum) or None
        self.output_max_per_q = None    # tensor[d_out] or None
        
        # Track whether codomain has been initialized from domain
        self.codomain_initialized_from_domain = False
        
        # Create codomain parameters for Φ if trainable
        if CONFIG['train_phi_codomain']:
            self.phi_codomain_params = PhiCodomainParams(cc=0.0, cr=10.0)
        else:
            self.phi_codomain_params = None
        
        # The inner monotonic spline φ (with fixed codomain [0,1])
        # Domain will be updated dynamically based on inputs and η
        phi_in_range = phi_domain if phi_domain is not None else (0, 1)
        self.phi = SimpleSpline(
            num_knots=phi_knots,
            in_range=phi_in_range,  # Use provided domain or default
            out_range=(0, 1),  # Fixed codomain
            monotonic=True
        )
        
        # The outer general spline Φ
        # Both domain and (optionally) codomain are dynamic
        Phi_in_range = Phi_domain if Phi_domain is not None else (0, 1)
        self.Phi = SimpleSpline(
            num_knots=Phi_knots,
            in_range=Phi_in_range,  # Use provided domain or default
            out_range=(0, 1),  # Temporary - will be updated immediately
            train_codomain=CONFIG['train_phi_codomain'],
            codomain_params=self.phi_codomain_params
        )
        
        # Weight vector (TRUE Sprecher λ: one per input dim)
        self.lambdas = nn.Parameter(torch.randn(d_in) * np.sqrt(2.0 / d_in))
        
        # Initialize eta to a reasonable value based on d_out
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10)))
        
        # Q-values for indexing
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))
        
        # Lateral mixing parameters
        if CONFIG['use_lateral_mixing'] and d_out > 1:
            self.lateral_scale = nn.Parameter(torch.tensor(CONFIG['lateral_scale_init']))
            
            if CONFIG['lateral_mixing_type'] == 'bidirectional':
                # Bidirectional mixing - mix with both neighbors
                self.lateral_weights_forward = nn.Parameter(
                    torch.ones(d_out) * CONFIG['lateral_weight_init'] * 0.5
                )
                self.lateral_weights_backward = nn.Parameter(
                    torch.ones(d_out) * CONFIG['lateral_weight_init'] * 0.5
                )
                # Indices for cyclic neighbors
                self.register_buffer('lateral_indices_forward', 
                                   torch.arange(d_out).roll(-1))
                self.register_buffer('lateral_indices_backward', 
                                   torch.arange(d_out).roll(1))
            else:  # 'cyclic' - default
                # Simple cyclic shift mixing
                self.lateral_weights = nn.Parameter(
                    torch.ones(d_out) * CONFIG['lateral_weight_init']
                )
                # Each output q receives from (q+1) % d_out
                self.register_buffer('lateral_indices', 
                                   torch.arange(d_out).roll(-1))
        else:
            # No lateral mixing
            self.lateral_scale = None
            self.lateral_weights = None
            self.lateral_weights_forward = None
            self.lateral_weights_backward = None
        
        # -------------------------------
        # Residual connection parameters
        # -------------------------------
        self.residual_style = str(CONFIG.get('residual_style', 'node')).lower()
        # Normalize potential synonyms
        if self.residual_style in ['standard', 'matrix']:
            self.residual_style = 'linear'

        # Initialize all residual attributes to None (then set as needed)
        self.residual_weight = None                  # scalar (only when d_in == d_out)
        self.residual_pooling_weights = None         # node-centric (d_in > d_out)
        self.residual_broadcast_weights = None       # node-centric (d_in < d_out)
        self.residual_projection = None              # standard linear projection (d_in != d_out)

        if CONFIG['use_residual_weights']:
            if d_in == d_out:
                # When dimensions match, use a simple scalar weight in both styles
                self.residual_weight = nn.Parameter(torch.tensor(0.1))
            else:
                if self.residual_style == 'linear':
                    # Standard residual: full projection matrix W (d_in x d_out)
                    self.residual_projection = nn.Parameter(torch.empty(d_in, d_out))
                    nn.init.xavier_uniform_(self.residual_projection)
                else:
                    # Node-centric (existing) behavior
                    if d_in > d_out:
                        # Pooling case: aggregate multiple inputs per output
                        init_value = 1.0 / d_in  # Normalized initialization
                        self.residual_pooling_weights = nn.Parameter(
                            torch.ones(d_in) * init_value
                        )
                        # Create balanced cyclic assignment
                        assignments = [(i % d_out) for i in range(d_in)]
                        self.register_buffer('pooling_assignment', torch.tensor(assignments, dtype=torch.long))
                        counts = torch.zeros(d_out)
                        for i in range(d_in):
                            counts[assignments[i]] += 1
                        self.register_buffer('pooling_counts', counts)
                    else:  # d_in < d_out
                        # Broadcasting case: expand inputs to outputs
                        self.residual_broadcast_weights = nn.Parameter(
                            torch.ones(d_out) * 0.1
                        )
                        sources = [(i % d_in) for i in range(d_out)]
                        self.register_buffer('broadcast_sources', torch.tensor(sources, dtype=torch.long))
        # else: all residual attributes remain None
    
    def update_phi_domain_theoretical(self, input_range, allow_resampling=True, force_resample=False):
        """Update φ domain based on theoretical bounds."""
        self.input_range = input_range
        
        # Domain needs to handle [in_min, in_max] + η*q for q in [0, d_out-1]
        # We need to handle both positive and negative η
        eta_val = self.eta.item()
        
        if eta_val >= 0:
            # Normal case: η is positive
            phi_min = input_range.min
            phi_max = input_range.max + eta_val * (self.d_out - 1)
        else:
            # When η is negative, the max shift is at q=0 and min shift is at q=(d_out-1)
            phi_min = input_range.min + eta_val * (self.d_out - 1)  # This will be less than input_range.min
            phi_max = input_range.max  # No shift when q=0
        
        # Ensure domain is valid (min < max)
        if phi_min >= phi_max:
            # This shouldn't happen in theory, but let's add a small epsilon to ensure validity
            phi_max = phi_min + 1e-4
        
        self.phi.update_domain((phi_min, phi_max), allow_resampling=allow_resampling, force_resample=force_resample)
    
    def _apply_lateral_mixing_bounds_per_q(self, s_min_per_q, s_max_per_q):
        """
        Apply **sign-aware** lateral mixing bounds to per-q intervals.
        """
        if self.lateral_scale is None:
            return s_min_per_q, s_max_per_q
        
        device = s_min_per_q.device
        
        if CONFIG['lateral_mixing_type'] == 'bidirectional':
            eff_f = (self.lateral_scale * self.lateral_weights_forward).to(device)
            eff_b = (self.lateral_scale * self.lateral_weights_backward).to(device)
            idx_f = self.lateral_indices_forward.to(device)
            idx_b = self.lateral_indices_backward.to(device)
            
            efp, efn = torch.clamp(eff_f, min=0), torch.clamp(eff_f, max=0)
            ebp, ebn = torch.clamp(eff_b, min=0), torch.clamp(eff_b, max=0)
            
            neighbor_min = (
                efp * s_min_per_q[idx_f] + efn * s_max_per_q[idx_f] +
                ebp * s_min_per_q[idx_b] + ebn * s_max_per_q[idx_b]
            )
            neighbor_max = (
                efp * s_max_per_q[idx_f] + efn * s_min_per_q[idx_f] +
                ebp * s_max_per_q[idx_b] + ebn * s_min_per_q[idx_b]
            )
            
            mixed_min = s_min_per_q + neighbor_min
            mixed_max = s_max_per_q + neighbor_max
        else:  # 'cyclic'
            eff = (self.lateral_scale * self.lateral_weights).to(device)
            idx = self.lateral_indices.to(device)
            ep, en = torch.clamp(eff, min=0), torch.clamp(eff, max=0)
            
            mixed_min = s_min_per_q + ep * s_min_per_q[idx] + en * s_max_per_q[idx]
            mixed_max = s_max_per_q + ep * s_max_per_q[idx] + en * s_min_per_q[idx]
        
        return mixed_min, mixed_max
    
    def _compute_s_bounds_per_q(self, a_vec, b_vec):
        """
        Helper: compute tight s_min/s_max per q using per-dimension intervals.
        a_vec, b_vec: tensors of shape [d_in] on correct device.
        Returns: s_min_per_q, s_max_per_q (tensors [d_out])
        """
        device = self.phi.knots.device
        q_values = torch.arange(self.d_out, device=device, dtype=torch.float32)

        # Broadcast a_i + η q and b_i + η q: shapes (d_in, d_out)
        eta_val = self.eta.item()
        a_mat = a_vec.view(-1, 1) + eta_val * q_values.view(1, -1)
        b_mat = b_vec.view(-1, 1) + eta_val * q_values.view(1, -1)

        # Evaluate φ elementwise on these matrices
        phi_a = self.phi(a_mat)  # (d_in, d_out)
        phi_b = self.phi(b_mat)  # (d_in, d_out)

        lambdas_col = self.lambdas.view(-1, 1)  # (d_in, 1)

        # Sign-aware contributions
        contrib_min = torch.where(lambdas_col >= 0, lambdas_col * phi_a, lambdas_col * phi_b)  # (d_in,d_out)
        contrib_max = torch.where(lambdas_col >= 0, lambdas_col * phi_b, lambdas_col * phi_a)  # (d_in,d_out)

        s_min_per_q = contrib_min.sum(dim=0) + Q_VALUES_FACTOR * q_values
        s_max_per_q = contrib_max.sum(dim=0) + Q_VALUES_FACTOR * q_values
        return s_min_per_q, s_max_per_q
    
    def update_Phi_domain_theoretical(self, allow_resampling=True, force_resample=False):
        """Update Φ domain based on theoretical bounds using per-q tight bounds (uses per-channel intervals when available)."""
        with torch.no_grad():
            device = self.phi.knots.device
            # Use per-channel input intervals if provided, else fall back to global [a,b]
            if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                    a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                    b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                s_min_per_q, s_max_per_q = self._compute_s_bounds_per_q(a_vec, b_vec)
            else:
                # Fallback: use global bounds with λ+ / λ− trick (original behavior)
                eta_val = self.eta.item()
                q_values = torch.arange(self.d_out, device=device, dtype=torch.float32)
                phi_inputs_min = self.input_range.min + eta_val * q_values
                phi_inputs_max = self.input_range.max + eta_val * q_values
                phi_at_min = self.phi(phi_inputs_min)
                phi_at_max = self.phi(phi_inputs_max)
                phi_min_per_q = torch.minimum(phi_at_min, phi_at_max)
                phi_max_per_q = torch.maximum(phi_at_min, phi_at_max)
                lambda_pos = torch.clamp(self.lambdas, min=0).sum().item()
                lambda_neg = torch.clamp(self.lambdas, max=0).sum().item()
                s_min_per_q = lambda_pos * phi_min_per_q + lambda_neg * phi_max_per_q + Q_VALUES_FACTOR * q_values
                s_max_per_q = lambda_pos * phi_max_per_q + lambda_neg * phi_min_per_q + Q_VALUES_FACTOR * q_values
            
            # Apply lateral mixing bounds if enabled (now sign-aware)
            if self.lateral_scale is not None:
                s_min_per_q, s_max_per_q = self._apply_lateral_mixing_bounds_per_q(s_min_per_q, s_max_per_q)
            
            # Union across q to form Φ input domain
            Phi_domain_min = s_min_per_q.min().item()
            Phi_domain_max = s_max_per_q.max().item()
            if Phi_domain_min >= Phi_domain_max:
                Phi_domain_max = Phi_domain_min + 1e-4
            
            self.Phi.update_domain((Phi_domain_min, Phi_domain_max), 
                                   allow_resampling=allow_resampling,
                                   force_resample=force_resample)
            
            # Initialize codomain to match domain (one time only)
            if CONFIG['train_phi_codomain'] and not self.codomain_initialized_from_domain:
                self.initialize_codomain_from_domain(Phi_domain_min, Phi_domain_max)
                self.codomain_initialized_from_domain = True
    
    def initialize_codomain_from_domain(self, domain_min, domain_max):
        """Initialize Φ codomain to match its computed domain."""
        with torch.no_grad():
            domain_center = (domain_min + domain_max) / 2
            domain_radius = (domain_max - domain_min) / 2
            
            # Update codomain parameters
            self.phi_codomain_params.cc.data = torch.tensor(domain_center, 
                                                           device=self.phi_codomain_params.cc.device,
                                                           dtype=self.phi_codomain_params.cc.dtype)
            self.phi_codomain_params.cr.data = torch.tensor(domain_radius,
                                                           device=self.phi_codomain_params.cr.device,
                                                           dtype=self.phi_codomain_params.cr.dtype)
            
            # Also initialize the spline coefficients to be approximately linear
            # This makes Φ start as identity-like
            self.Phi.initialize_as_identity(domain_min, domain_max)
            
            if CONFIG.get('debug_domains', False):
                print(f"Layer {self.layer_num}: Initialized Phi codomain to match domain [{domain_min:.3f}, {domain_max:.3f}]")
                print(f"  cc = {domain_center:.3f}, cr = {domain_radius:.3f}")
    
    def compute_pooling_residual_bounds(self, input_interval):
        """
        Compute tight bounds for pooling residual contributions (union across outputs).
        """
        if self.residual_pooling_weights is None:
            return Interval(0, 0)
        
        with torch.no_grad():
            # Initialize bounds for each output
            output_bounds = []
            
            for out_idx in range(self.d_out):
                # Find which inputs contribute to this output
                contributing_inputs = (self.pooling_assignment == out_idx).nonzero(as_tuple=True)[0]
                
                if len(contributing_inputs) == 0:
                    output_bounds.append(Interval(0, 0))
                    continue
                
                # Compute contribution from each assigned input
                contribution_min = 0
                contribution_max = 0
                
                for in_idx in contributing_inputs:
                    weight = self.residual_pooling_weights[in_idx].item()
                    
                    if weight >= 0:
                        contribution_min += weight * input_interval.min
                        contribution_max += weight * input_interval.max
                    else:
                        contribution_min += weight * input_interval.max
                        contribution_max += weight * input_interval.min
                
                output_bounds.append(Interval(contribution_min, contribution_max))
            
            # Return the union of all output bounds
            final_bounds = output_bounds[0]
            for bounds in output_bounds[1:]:
                final_bounds = final_bounds.union(bounds)
            
            return final_bounds

    def compute_pooling_residual_bounds_per_output(self, input_interval, a_per_dim=None, b_per_dim=None):
        """
        NEW (tight): Compute per-output pooling residual intervals for composition.
        Supports optional per-dimension intervals (a_per_dim, b_per_dim).
        Returns:
            r_min_per_q (tensor shape [d_out]), r_max_per_q (tensor shape [d_out])
        """
        device = self.phi.knots.device
        if self.residual_pooling_weights is None:
            return (torch.zeros(self.d_out, device=device),
                    torch.zeros(self.d_out, device=device))
        
        with torch.no_grad():
            w = self.residual_pooling_weights.to(device)  # (d_in,)
            assign = self.pooling_assignment.to(device)   # (d_in,)
            
            if a_per_dim is not None and b_per_dim is not None:
                a_vec = torch.as_tensor(a_per_dim, device=device, dtype=torch.float32)
                b_vec = torch.as_tensor(b_per_dim, device=device, dtype=torch.float32)
                # Contribution of each input (min/max depending on weight sign) USING per-dim intervals
                contrib_min = torch.where(w >= 0, w * a_vec, w * b_vec)
                contrib_max = torch.where(w >= 0, w * b_vec, w * a_vec)
            else:
                # Fallback to scalar interval
                a, b = input_interval.min, input_interval.max
                contrib_min = torch.where(w >= 0, w * a, w * b)
                contrib_max = torch.where(w >= 0, w * b, w * a)
            
            # Sum contributions per assigned output using scatter_add
            r_min_per_q = torch.zeros(self.d_out, device=device)
            r_max_per_q = torch.zeros(self.d_out, device=device)
            r_min_per_q.scatter_add_(0, assign, contrib_min)
            r_max_per_q.scatter_add_(0, assign, contrib_max)
            
            return r_min_per_q, r_max_per_q

    def _compute_linear_residual_bounds_per_output(self, a_vec, b_vec):
        """
        STANDARD residual bounds for projection matrix W (d_in x d_out):
            r_q = sum_i x_i * W_{i,q}
        Per-output tight intervals using per-dimension bounds:
            min = a·W^+ + b·W^- ,  max = b·W^+ + a·W^-
        Returns: r_min_per_q, r_max_per_q with shape [d_out].
        """
        device = self.phi.knots.device
        W = self.residual_projection.to(device)  # (d_in, d_out)
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        a_vec = torch.as_tensor(a_vec, device=device, dtype=torch.float32)
        b_vec = torch.as_tensor(b_vec, device=device, dtype=torch.float32)
        # Row-vector times matrix -> (d_out,)
        r_min = torch.matmul(a_vec, W_pos) + torch.matmul(b_vec, W_neg)
        r_max = torch.matmul(b_vec, W_pos) + torch.matmul(a_vec, W_neg)
        return r_min, r_max
    
    def compute_output_range_theoretical(self):
        """Compute theoretical output range of this block using per-q tight composition (uses per-channel intervals when available)."""
        with torch.no_grad():
            device = self.phi.knots.device
            q_values = torch.arange(self.d_out, device=device, dtype=torch.float32)
            
            # Prepare per-dimension input intervals if provided
            if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                    a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                    b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                # Tight s-bounds using per-dim intervals
                s_min_per_q, s_max_per_q = self._compute_s_bounds_per_q(a_vec, b_vec)
            else:
                # Fallback: use global bounds with λ+ / λ− trick (original behavior)
                eta_val = self.eta.item()
                phi_inputs_min = self.input_range.min + eta_val * q_values
                phi_inputs_max = self.input_range.max + eta_val * q_values
                phi_at_min = self.phi(phi_inputs_min)
                phi_at_max = self.phi(phi_inputs_max)
                phi_min_per_q = torch.minimum(phi_at_min, phi_at_max)
                phi_max_per_q = torch.maximum(phi_at_min, phi_at_max)
                lambda_pos = torch.clamp(self.lambdas, min=0).sum().item()
                lambda_neg = torch.clamp(self.lambdas, max=0).sum().item()
                s_min_per_q = lambda_pos * phi_min_per_q + lambda_neg * phi_max_per_q + Q_VALUES_FACTOR * q_values
                s_max_per_q = lambda_pos * phi_max_per_q + lambda_neg * phi_min_per_q + Q_VALUES_FACTOR * q_values
            
            # Lateral mixing (now sign-aware endpoint bounds)
            if self.lateral_scale is not None:
                s_min_per_q, s_max_per_q = self._apply_lateral_mixing_bounds_per_q(s_min_per_q, s_max_per_q)
            
            # Per-q Φ output ranges (evaluate endpoints + interior knots)
            Phi_at_smin = self.Phi(s_min_per_q)
            Phi_at_smax = self.Phi(s_max_per_q)
            y_min_per_q = torch.minimum(Phi_at_smin, Phi_at_smax)
            y_max_per_q = torch.maximum(Phi_at_smin, Phi_at_smax)
            
            # Check interior knots per q
            knots = self.Phi.knots  # (K,)
            Phi_at_knots = self.Phi(knots)  # (K,)
            K = knots.shape[0]
            mask = (knots.unsqueeze(0) >= s_min_per_q.unsqueeze(1)) & (knots.unsqueeze(0) <= s_max_per_q.unsqueeze(1))
            if mask.any():
                Yk = Phi_at_knots.unsqueeze(0).expand(self.d_out, K)
                Yk_min = torch.where(mask, Yk, torch.full_like(Yk, float('inf'))).min(dim=1).values
                Yk_max = torch.where(mask, Yk, torch.full_like(Yk, float('-inf'))).max(dim=1).values
                has_knots = mask.any(dim=1)
                y_min_per_q = torch.where(has_knots, torch.minimum(y_min_per_q, Yk_min), y_min_per_q)
                y_max_per_q = torch.where(has_knots, torch.maximum(y_max_per_q, Yk_max), y_max_per_q)
            
            # Residuals per q (now includes standard 'linear' projection case)
            if CONFIG['use_residual_weights'] and self.input_range is not None:
                if self.residual_weight is not None:
                    # Scalar residual per output (d_in == d_out): per-q bounds via per-dim intervals
                    w = self.residual_weight.to(device)
                    if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                        a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                        b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                        if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                            a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                            b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                        r_min_per_q = torch.where(w >= 0, w * a_vec, w * b_vec)
                        r_max_per_q = torch.where(w >= 0, w * b_vec, w * a_vec)
                    else:
                        # Fallback to global interval
                        a, b = self.input_range.min, self.input_range.max
                        r_lo, r_hi = (w * a, w * b) if w.item() >= 0 else (w * b, w * a)
                        r_min_per_q = torch.full((self.d_out,), r_lo.item(), device=device)
                        r_max_per_q = torch.full((self.d_out,), r_hi.item(), device=device)
                elif self.residual_projection is not None:
                    # STANDARD linear projection case (d_in != d_out)
                    if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                        a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                        b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                        if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                            a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                            b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                    else:
                        a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                        b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                    r_min_per_q, r_max_per_q = self._compute_linear_residual_bounds_per_output(a_vec, b_vec)
                elif self.residual_pooling_weights is not None:
                    if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                        a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                        b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                        r_min_per_q, r_max_per_q = self.compute_pooling_residual_bounds_per_output(
                            Interval(self.input_range.min, self.input_range.max),
                            a_per_dim=a_vec, b_per_dim=b_vec
                        )
                    else:
                        # Fallback to global interval
                        a, b = self.input_range.min, self.input_range.max
                        r_min_per_q, r_max_per_q = self.compute_pooling_residual_bounds_per_output(Interval(a, b))
                        r_min_per_q = r_min_per_q.to(device)
                        r_max_per_q = r_max_per_q.to(device)
                elif self.residual_broadcast_weights is not None:
                    # Broadcasting case: per-q via per-dim intervals if available
                    w = self.residual_broadcast_weights.to(device)  # (d_out,)
                    src = self.broadcast_sources.to(device)         # (d_out,)
                    if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                        a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                        b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                        if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                            a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                            b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                        a_src = a_vec[src]  # (d_out,)
                        b_src = b_vec[src]  # (d_out,)
                        r_min_per_q = torch.where(w >= 0, w * a_src, w * b_src)
                        r_max_per_q = torch.where(w >= 0, w * b_src, w * a_src)
                    else:
                        a, b = self.input_range.min, self.input_range.max
                        r_min_per_q = torch.where(w >= 0, w * a, w * b)
                        r_max_per_q = torch.where(w >= 0, w * b, w * a)
                else:
                    r_min_per_q = torch.zeros(self.d_out, device=device)
                    r_max_per_q = torch.zeros(self.d_out, device=device)
            else:
                r_min_per_q = torch.zeros(self.d_out, device=device)
                r_max_per_q = torch.zeros(self.d_out, device=device)
            
            # Compose Φ and residual per-q (safe Minkowski sum)
            final_min_per_q = y_min_per_q + r_min_per_q
            final_max_per_q = y_max_per_q + r_max_per_q

            # Store per-q outputs for downstream per-channel propagation
            self.output_min_per_q = final_min_per_q.detach().clone()
            self.output_max_per_q = final_max_per_q.detach().clone()

            # Aggregate to layer output_range
            if self.is_final:
                # Final block sums its outputs -> use Minkowski sum across q
                final_min = final_min_per_q.sum().item()
                final_max = final_max_per_q.sum().item()
            else:
                # Vector output: union across components for global min/max
                final_min = final_min_per_q.min().item()
                final_max = final_max_per_q.max().item()
        
        self.output_range = TheoreticalRange(final_min, final_max)
        return self.output_range
    
    def get_theoretical_ranges(self):
        """Get theoretical ranges for checkpoint saving."""
        ranges = {}
        if self.input_range is not None:
            ranges['input_range'] = (self.input_range.min, self.input_range.max)
        if self.output_range is not None:
            ranges['output_range'] = (self.output_range.min, self.output_range.max)
        # Also save the codomain initialization flag
        ranges['codomain_initialized'] = self.codomain_initialized_from_domain
        return ranges
    
    def set_theoretical_ranges(self, ranges):
        """Restore theoretical ranges from checkpoint."""
        if 'input_range' in ranges:
            self.input_range = TheoreticalRange(ranges['input_range'][0], ranges['input_range'][1])
        if 'output_range' in ranges:
            self.output_range = TheoreticalRange(ranges['output_range'][0], ranges['output_range'][1])
        # Restore the codomain initialization flag
        if 'codomain_initialized' in ranges:
            self.codomain_initialized_from_domain = ranges['codomain_initialized']
    
    def forward(self, x, x_original=None):
        """Forward pass with configurable memory mode."""
        # Check if we should use memory-efficient mode
        if CONFIG.get('low_memory_mode', False):
            return self._forward_memory_efficient(x, x_original)
        else:
            return self._forward_original(x, x_original)
    
    def _forward_original(self, x, x_original=None):
        """Original forward pass - O(B × d_in × d_out) memory."""
        # Ensure the q_values buffer is on the same device as the input tensor.
        q_on_device = self.q_values.to(x.device)

        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = q_on_device.view(1, 1, -1)  # shape: (1, 1, d_out)
        
        # Apply translation by η * q (part of Sprecher's construction)
        shifted = x_expanded + self.eta * q
        
        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)
        
        # Weight by λ (now a VECTOR) and sum over input dimension
        weighted = phi_out * self.lambdas.view(1, -1, 1)  # Broadcast: (1, d_in, 1)
        # Use the device-corrected q_on_device tensor here as well
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * q_on_device  # shape: (batch_size, d_out)
        
        # Apply lateral mixing if enabled
        if self.lateral_scale is not None:
            s = self._apply_lateral_mixing_to_s(s, x.device)
        
        # Apply Phi
        activated = self.Phi(s)
        
        # Add residual connections
        activated = self._add_residual(activated, x_original)
        
        if self.is_final:
            # Sum outputs if this is the final block (produces scalar output)
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated
    
    def _forward_memory_efficient(self, x, x_original=None):
        """Memory-efficient forward pass - O(B × max(d_in, d_out)) memory."""
        batch_size = x.shape[0]
        device = x.device
        
        # Pre-allocate output tensor for s values
        s = torch.zeros(batch_size, self.d_out, device=device, dtype=x.dtype)
        
        # Process each output dimension sequentially
        for q_idx in range(self.d_out):
            # Get the q value for this output dimension (as scalar to avoid tensor ops)
            q_val = float(q_idx)
            
            # Compute phi for all inputs with this specific q
            # Memory: O(B × d_in) instead of O(B × d_in × d_out)
            shifted = x + self.eta * q_val  # (batch_size, d_in)
            phi_out = self.phi(shifted)     # (batch_size, d_in)
            
            # Weight by lambdas and sum over input dimension
            weighted = phi_out * self.lambdas  # (batch_size, d_in) 
            s[:, q_idx] = weighted.sum(dim=1) + Q_VALUES_FACTOR * q_val  # (batch_size,)
        
        # Apply lateral mixing if enabled (on s before Phi)
        if self.lateral_scale is not None:
            s = self._apply_lateral_mixing_to_s(s, device)
        
        # Apply Phi to all outputs (this is memory-efficient as it's element-wise)
        activated = self.Phi(s)  # (batch_size, d_out)
        
        # Add residual connections
        activated = self._add_residual(activated, x_original)
        
        # Handle final summing if needed
        if self.is_final:
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated
    
    def _apply_lateral_mixing_to_s(self, s, device):
        """Apply lateral mixing to the intermediate representation s."""
        if CONFIG['lateral_mixing_type'] == 'bidirectional':
            s_forward = s[:, self.lateral_indices_forward.to(device)]
            s_backward = s[:, self.lateral_indices_backward.to(device)]
            s_mixed = s + self.lateral_scale * (
                self.lateral_weights_forward * s_forward + 
                self.lateral_weights_backward * s_backward
            )
        else:  # 'cyclic'
            s_shifted = s[:, self.lateral_indices.to(device)]
            s_mixed = s + self.lateral_scale * (self.lateral_weights * s_shifted)
        return s_mixed
    
    def _add_residual(self, activated, x_original):
        """Add residual connections if enabled."""
        if not CONFIG['use_residual_weights'] or x_original is None:
            return activated
        
        if self.residual_projection is not None:
            # STANDARD linear projection residual: x @ W
            activated = activated + torch.matmul(x_original, self.residual_projection)
        
        elif self.residual_weight is not None:
            # Same dimension case - simple scalar weight
            activated = activated + self.residual_weight * x_original
            
        elif self.residual_pooling_weights is not None:
            # Pooling case: d_in > d_out
            weighted_input = x_original * self.residual_pooling_weights.view(1, -1)
            residual_contribution = torch.zeros(
                x_original.shape[0], self.d_out, 
                device=x_original.device, dtype=x_original.dtype
            )
            residual_contribution.scatter_add_(
                1, 
                self.pooling_assignment.unsqueeze(0).expand(x_original.shape[0], -1),
                weighted_input
            )
            activated = activated + residual_contribution
            
        elif self.residual_broadcast_weights is not None:
            # Broadcasting case: d_in < d_out
            residual_contribution = torch.gather(
                x_original, 1,
                self.broadcast_sources.unsqueeze(0).expand(x_original.shape[0], -1)
            )
            residual_contribution = residual_contribution * self.residual_broadcast_weights.view(1, -1)
            activated = activated + residual_contribution
        
        return activated
    
    def get_output_range(self):
        """Get the current output range of this block (for the next layer's input)."""
        # Return the computed theoretical output range
        if self.output_range is None:
            return self.compute_output_range_theoretical()
        return self.output_range
    
    def get_domain_violation_stats(self):
        """Get domain violation statistics for this block."""
        stats = {
            'phi': self.phi.get_domain_violation_stats(),
            'Phi': self.Phi.get_domain_violation_stats()
        }
        return stats
    
    def reset_domain_violation_stats(self):
        """Reset domain violation tracking for this block."""
        self.phi.reset_domain_violation_stats()
        self.Phi.reset_domain_violation_stats()


class SprecherMultiLayerNetwork(nn.Module):
    """
    Builds the Sprecher network with a given hidden-layer architecture and final output dimension.
    
    Args:
        input_dim: Input dimension
        architecture: List of hidden layer sizes
        final_dim: Output dimension
        phi_knots: Number of knots for phi splines
        Phi_knots: Number of knots for Phi splines
        norm_type: Type of normalization ('none', 'batch', 'layer')
        norm_position: Position of normalization ('before', 'after')
        norm_skip_first: Whether to skip normalization for first block
        initialize_domains: Whether to initialize domains on creation (set False when loading checkpoint)
        domain_ranges: Dict of domain ranges for each layer's splines (optional)
    Note:
        Residual style is controlled by CONFIG['residual_style'] ∈ {'node', 'linear'}.
        Default is 'node' to preserve original behavior.
    """
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=100, Phi_knots=100,
                 norm_type='none', norm_position='after', norm_skip_first=True, initialize_domains=True,
                 domain_ranges=None):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.norm_skip_first = norm_skip_first
        
        layers = []
        if not architecture: # No hidden layers
            is_final = (final_dim == 1)
            # Get domain ranges if provided
            phi_domain = domain_ranges.get('layer_0_phi') if domain_ranges else None
            Phi_domain = domain_ranges.get('layer_0_Phi') if domain_ranges else None
            layers.append(SprecherLayerBlock(
                d_in=input_dim, d_out=final_dim,
                layer_num=0, is_final=is_final,
                phi_knots=phi_knots, Phi_knots=Phi_knots,
                phi_domain=phi_domain, Phi_domain=Phi_domain
            ))
        else:
            L = len(architecture)
            # Create hidden layers
            d_in = input_dim
            for i in range(L):
                d_out = architecture[i]
                # The last hidden block is summed if the final output is scalar
                is_final_block = (i == L - 1) and (self.final_dim == 1)
                # Get domain ranges if provided
                phi_domain = domain_ranges.get(f'layer_{i}_phi') if domain_ranges else None
                Phi_domain = domain_ranges.get(f'layer_{i}_Phi') if domain_ranges else None
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=d_out,
                    layer_num=i, is_final=is_final_block,
                    phi_knots=phi_knots, Phi_knots=Phi_knots,
                    phi_domain=phi_domain, Phi_domain=Phi_domain
                ))
                d_in = d_out
            
            # Add a final output block if output is vector-valued
            if self.final_dim > 1:
                # Get domain ranges if provided
                phi_domain = domain_ranges.get(f'layer_{L}_phi') if domain_ranges else None
                Phi_domain = domain_ranges.get(f'layer_{L}_Phi') if domain_ranges else None
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=self.final_dim,
                    layer_num=L, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots,
                    phi_domain=phi_domain, Phi_domain=Phi_domain
                ))
        
        self.layers = nn.ModuleList(layers)
        
        # Create normalization layers if requested
        self.norm_layers = nn.ModuleList()
        if norm_type != 'none':
            for i, layer in enumerate(self.layers):
                # Skip normalization for first block if requested
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    # Determine the number of features based on position
                    if norm_position == 'before':
                        # Before the block, use input dimension
                        num_features = layer.d_in
                    else:  # after
                        # After the block, use output dimension
                        # Note: if is_final, output is scalar (1D)
                        num_features = 1 if layer.is_final else layer.d_out
                    
                    if norm_type == 'batch':
                        self.norm_layers.append(nn.BatchNorm1d(num_features))
                    elif norm_type == 'layer':
                        self.norm_layers.append(nn.LayerNorm(num_features))
                    else:
                        raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # Output scaling parameters for better initialization
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
        # Initialize domains based on theoretical bounds
        # Can be deferred when loading from checkpoint to avoid conflicts
        if initialize_domains and CONFIG.get('use_theoretical_domains', True):
            self.update_all_domains()
    
    def update_all_domains(self, allow_resampling=True, force_resample=False):
        """
        Update all spline domains based on theoretical bounds WITH normalization handling.
        
        Args:
            allow_resampling: If False, prevent spline coefficient resampling during domain updates
            force_resample: If True, force resampling regardless of total_evaluations
        """
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

        # Start with input in [0,1]^n
        current_interval = Interval(0.0, 1.0)
        # NEW: per-channel intervals for propagation (start uniform)
        current_min_per_dim = torch.zeros(self.input_dim, device=device)
        current_max_per_dim = torch.ones(self.input_dim, device=device)
        
        for layer_idx, layer in enumerate(self.layers):
            # Handle normalization BEFORE block if configured
            if self.norm_type != 'none' and self.norm_position == 'before':
                norm_layer = self.norm_layers[layer_idx]
                if not isinstance(norm_layer, nn.Identity):
                    if isinstance(norm_layer, nn.BatchNorm1d):
                        # Compute global interval (training conservative)
                        current_interval = compute_batchnorm_bounds(
                            current_interval, norm_layer, training_mode=self.training
                        )
                        # Propagate per-dim as uniform (conservative)
                        current_min_per_dim = torch.full((layer.d_in,), current_interval.min, device=device)
                        current_max_per_dim = torch.full((layer.d_in,), current_interval.max, device=device)
                    elif isinstance(norm_layer, nn.LayerNorm):
                        current_interval = compute_layernorm_bounds(
                            current_interval, norm_layer, layer.d_in
                        )
                        # Propagate per-dim as uniform (conservative)
                        current_min_per_dim = torch.full((layer.d_in,), current_interval.min, device=device)
                        current_max_per_dim = torch.full((layer.d_in,), current_interval.max, device=device)
            
            # Update layer's per-channel inputs for tighter computations
            layer.input_min_per_dim = current_min_per_dim.detach().clone().to(layer.phi.knots.device)
            layer.input_max_per_dim = current_max_per_dim.detach().clone().to(layer.phi.knots.device)

            # Update layer's input range (global)
            current_range = TheoreticalRange(current_interval.min, current_interval.max)
            
            # Update φ domain based on input range
            layer.update_phi_domain_theoretical(current_range, allow_resampling, force_resample)
            
            # Update Φ domain using available per-channel info
            layer.update_Phi_domain_theoretical(allow_resampling, force_resample)
            
            # Compute this layer's output range
            layer.compute_output_range_theoretical()
            
            # Prepare per-channel outputs for next layer (pre-normalization)
            if layer.is_final:
                # If final (sum) block, output is scalar; next layer doesn't exist in standard arch
                next_min_per_dim = torch.tensor([layer.output_range.min], device=device)
                next_max_per_dim = torch.tensor([layer.output_range.max], device=device)
            else:
                next_min_per_dim = layer.output_min_per_q.to(device)
                next_max_per_dim = layer.output_max_per_q.to(device)
            
            # Get global output interval (for normalization and logging)
            output_interval = Interval(layer.output_range.min, layer.output_range.max)
            
            # Handle normalization AFTER block if configured
            if self.norm_type != 'none' and self.norm_position == 'after':
                norm_layer = self.norm_layers[layer_idx]
                if not isinstance(norm_layer, nn.Identity):
                    if isinstance(norm_layer, nn.BatchNorm1d):
                        # Global (safe) interval for BN (training conservative)
                        output_interval = compute_batchnorm_bounds(
                            output_interval, norm_layer, training_mode=self.training
                        )
                        # Per-channel: if in EVAL, we can transform per-dim bounds using running stats+affine
                        if not self.training:
                            eps = norm_layer.eps
                            running_mean = norm_layer.running_mean.to(device)
                            running_var = norm_layer.running_var.to(device)
                            std = torch.sqrt(running_var + eps)
                            # Standardize per channel
                            lo = (next_min_per_dim - running_mean) / std
                            hi = (next_max_per_dim - running_mean) / std
                            std_min = torch.minimum(lo, hi)
                            std_max = torch.maximum(lo, hi)
                            if norm_layer.affine:
                                w = norm_layer.weight.to(device)
                                b = norm_layer.bias.to(device)
                                next_min_per_dim = torch.where(w >= 0, w * std_min + b, w * std_max + b)
                                next_max_per_dim = torch.where(w >= 0, w * std_max + b, w * std_min + b)
                            else:
                                next_min_per_dim = std_min
                                next_max_per_dim = std_max
                        else:
                            # Training: use uniform conservative range
                            next_min_per_dim = torch.full_like(next_min_per_dim, output_interval.min)
                            next_max_per_dim = torch.full_like(next_max_per_dim, output_interval.max)
                    elif isinstance(norm_layer, nn.LayerNorm):
                        # Apply conservative global bound (feature-coupled)
                        num_features = 1 if layer.is_final else layer.d_out
                        output_interval = compute_layernorm_bounds(
                            output_interval, norm_layer, num_features
                        )
                        # Per-dim arrays become uniform conservative bounds
                        next_min_per_dim = torch.full_like(next_min_per_dim, output_interval.min)
                        next_max_per_dim = torch.full_like(next_max_per_dim, output_interval.max)
            
            # Update for next layer
            current_min_per_dim = next_min_per_dim
            current_max_per_dim = next_max_per_dim
            # Global interval for next φ update
            current_interval = Interval(current_min_per_dim.min().item(), current_max_per_dim.max().item())
            
            # Debug output if configured
            if CONFIG.get('debug_domains', False):
                print(f"Layer {layer.layer_num}: input_range={layer.input_range}, output_range={layer.output_range}")
                print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
                print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
                if layer_idx < len(self.norm_layers) and self.norm_type != 'none':
                    print(f"  After normalization: [{current_interval.min:.3f}, {current_interval.max:.3f}]")
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.layers):
            # Apply normalization before block if requested
            if self.norm_type != 'none' and self.norm_position == 'before':
                x_in = self.norm_layers[i](x_in)
            
            # For residual connections, pass the input of the *current* layer
            x_out = layer(x_in, x_in) 
            
            # Apply normalization after block if requested
            if self.norm_type != 'none' and self.norm_position == 'after':
                x_out = self.norm_layers[i](x_out)
            
            x_in = x_out # The output of this layer is the input to the next
        
        # Final output after all layers
        x = x_in
        
        # Apply output scaling and bias
        x = self.output_scale * x + self.output_bias
        return x
    
    def get_domain_violation_stats(self):
        """Get domain violation statistics for all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_domain_violation_stats()
        return stats
    
    def reset_domain_violation_stats(self):
        """Reset domain violation tracking for all layers."""
        for layer in self.layers:
            layer.reset_domain_violation_stats()
    
    def print_domain_violation_report(self):
        """Print a summary of domain violations."""
        if not CONFIG.get('track_domain_violations', False):
            print("Domain violation tracking is not enabled.")
            return
        
        print("\nDomain Violation Report:")
        print("-" * 50)
        stats = self.get_domain_violation_stats()
        for layer_name, layer_stats in stats.items():
            print(f"{layer_name}:")
            print(f"  phi violations: {layer_stats['phi']:.2%}")
            print(f"  Phi violations: {layer_stats['Phi']:.2%}")
    
    def get_all_domain_states(self):
        """Get domain states for all splines in all layers."""
        domain_states = {}
        for i, layer in enumerate(self.layers):
            domain_states[f'layer_{i}_phi'] = layer.phi.get_domain_state()
            domain_states[f'layer_{i}_Phi'] = layer.Phi.get_domain_state()
            # Also save theoretical ranges
            domain_states[f'layer_{i}_ranges'] = layer.get_theoretical_ranges()
        return domain_states
    
    def get_domain_ranges(self):
        """Get just the domain ranges for all splines (for initialization)."""
        domain_ranges = {}
        for i, layer in enumerate(self.layers):
            domain_ranges[f'layer_{i}_phi'] = (layer.phi.in_min, layer.phi.in_max)
            domain_ranges[f'layer_{i}_Phi'] = (layer.Phi.in_min, layer.Phi.in_max)
        return domain_ranges
    
    def set_all_domain_states(self, domain_states):
        """Restore domain states for all splines in all layers."""
        for i, layer in enumerate(self.layers):
            phi_key = f'layer_{i}_phi'
            Phi_key = f'layer_{i}_Phi'
            ranges_key = f'layer_{i}_ranges'
            
            if phi_key in domain_states:
                layer.phi.set_domain_state(domain_states[phi_key])
            if Phi_key in domain_states:
                layer.Phi.set_domain_state(domain_states[Phi_key])
            # Restore theoretical ranges
            if ranges_key in domain_states:
                layer.set_theoretical_ranges(domain_states[ranges_key])


def test_domain_tightness(model, dataset, n_samples=10000):
    """Test that computed domains are both safe and tight."""
    device = next(model.parameters()).device
    
    # Generate many samples to test domain coverage
    x_test = torch.rand(n_samples, dataset.input_dim, device=device)
    
    # Track actual min/max values seen at each spline
    actual_ranges = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if hasattr(module, 'in_min'):  # It's a spline
                actual_min = input[0].min().item()
                actual_max = input[0].max().item()
                
                if name not in actual_ranges:
                    actual_ranges[name] = Interval(actual_min, actual_max)
                else:
                    actual_ranges[name] = actual_ranges[name].union(
                        Interval(actual_min, actual_max)
                    )
                
                # Check safety: no values outside computed domain
                if actual_min < module.in_min - 1e-6 or actual_max > module.in_max + 1e-6:
                    print(f"SAFETY VIOLATION in {name}:")
                    print(f"  Computed: [{module.in_min:.6f}, {module.in_max:.6f}]")
                    print(f"  Actual:   [{actual_min:.6f}, {actual_max:.6f}]")
        return hook
    
    # Register hooks
    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.phi.register_forward_hook(hook_fn(f"layer_{i}_phi")))
        handles.append(layer.Phi.register_forward_hook(hook_fn(f"layer_{i}_Phi")))
    
    # Run forward passes
    with torch.no_grad():
        for batch_start in range(0, n_samples, 100):
            batch_end = min(batch_start + 100, n_samples)
            _ = model(x_test[batch_start:batch_end])
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Check tightness: how much slack in computed domains?
    print("\nDomain Tightness Analysis:")
    print("=" * 60)
    violations_found = False
    
    for i, layer in enumerate(model.layers):
        phi_name = f"layer_{i}_phi"
        Phi_name = f"layer_{i}_Phi"
        
        if phi_name in actual_ranges:
            actual = actual_ranges[phi_name]
            computed = Interval(layer.phi.in_min, layer.phi.in_max)
            slack_min = actual.min - computed.min
            slack_max = computed.max - actual.max
            
            # Check for violations
            if actual.min < computed.min - 1e-6 or actual.max > computed.max + 1e-6:
                violations_found = True
                print(f"{phi_name}: VIOLATION DETECTED")
            else:
                print(f"✓ {phi_name}:")
            
            print(f"  Computed: {computed}")
            print(f"  Actual:   {actual}")
            print(f"  Slack:    [{slack_min:.6f}, {slack_max:.6f}]")
        
        if Phi_name in actual_ranges:
            actual = actual_ranges[Phi_name]
            computed = Interval(layer.Phi.in_min, layer.Phi.in_max)
            slack_min = actual.min - computed.min
            slack_max = computed.max - actual.max
            
            # Check for violations
            if actual.min < computed.min - 1e-6 or actual.max > computed.max + 1e-6:
                violations_found = True
                print(f"{Phi_name}: VIOLATION DETECTED")
            else:
                print(f"✓ {Phi_name}:")
            
            print(f"  Computed: {computed}")
            print(f"  Actual:   {actual}")
            print(f"  Slack:    [{slack_min:.6f}, {slack_max:.6f}]")
    
    print("=" * 60)
    if violations_found:
        print("DOMAIN VIOLATIONS DETECTED - Computed domains are not safe!")
    else:
        print("✓ All domains are safe - no violations detected")
    
    return not violations_found