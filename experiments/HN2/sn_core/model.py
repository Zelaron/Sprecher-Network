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
    A piecewise-linear or higher-order spline.
    
    For φ splines: domain is dynamically updated, codomain is fixed [0,1]
    For Φ splines: domain is dynamically updated, codomain can be fixed or trainable
    
    Args:
        order: Spline order (1 for Piecewise Linear, 3 for Cubic B-Spline, etc.)
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False,
                 train_codomain=False, codomain_params=None, order=1):
        super().__init__()
        self.num_knots = num_knots
        self.monotonic = monotonic
        self.train_codomain = train_codomain
        self.codomain_params = codomain_params
        self.order = order
        
        # Store initial ranges
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        
        # Create knots buffer - will be updated dynamically
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        # Domain violation tracking
        self.domain_violations = 0
        self.total_evaluations = 0
        
        # Initialize spline coefficients
        torch.manual_seed(CONFIG['seed'])
        
        # For monotonicity with any order spline (B-spline property: monotonic control points -> monotonic curve)
        if monotonic:
            # Use log-space parameters to ensure monotonicity via cumsum
            self.log_increments = nn.Parameter(torch.zeros(num_knots))
            with torch.no_grad():
                # Initialize to approximate linear function
                self.log_increments.data = torch.log(torch.ones(num_knots) / num_knots + 1e-6)
        else:
            # For general splines (Φ), initialize to zeros
            # Will be properly initialized when domain is computed
            self.coeffs = nn.Parameter(torch.zeros(num_knots))
            
        # Precompute basis matrix for cubic B-splines if needed
        if self.order == 3:
            # Uniform Cubic B-Spline basis matrix (for t in [0,1])
            # [1, t, t^2, t^3] @ M
            M = torch.tensor([
                [1.0, 4.0, 1.0, 0.0],
                [-3.0, 0.0, 3.0, 0.0],
                [3.0, -6.0, 3.0, 0.0],
                [-1.0, 3.0, -3.0, 1.0]
            ]) / 6.0
            self.register_buffer('cubic_basis_matrix', M)

    def _get_greville_points(self, domain_min, domain_max, count):
        """
        Compute Greville abscissae (node points) for the given domain and count.
        For order 1 (linear): Uniform grid points (knots).
        For order k: Averaged knot locations. For uniform B-splines, these are just uniform points.
        """
        return torch.linspace(domain_min, domain_max, count, 
                              device=self.knots.device, dtype=self.knots.dtype)

    def _values_to_control_points(self, values):
        """
        Approximate inverse B-spline transform: function values -> control points.
        For Order 1: Identity (control points = values).
        For Order 3: Iterative sharpening to invert the smoothing effect of B-splines.
        """
        if self.order == 1:
            return values
            
        # For cubic B-splines (order 3), the relation between control points (c) 
        # and curve values (y) at knots is: y[i] = 1/6*c[i-1] + 4/6*c[i] + 1/6*c[i+1].
        # This is a convolution. To get c from y, we roughly invert this.
        
        c = values.clone()
        # 3 iterations of Richardson iteration / sharpening is sufficient
        for _ in range(3):
            # Forward pass: what curve values would these control points 'c' produce?
            # Pad for convolution (replicate boundary)
            c_pad = torch.cat([c[:1], c, c[-1:]])
            # Convolve with [1/6, 4/6, 1/6]
            y_approx = (c_pad[:-2] + 4*c_pad[1:-1] + c_pad[2:]) / 6.0
            
            # Update c to correct the error
            error = values - y_approx
            c = c + error
            
        return c

    def update_domain(self, new_range, allow_resampling=True, force_resample=False):
        """
        Update the domain of the spline by adjusting knot positions.
        Resamples the spline to preserve its learned functional shape.
        """
        with torch.no_grad():
            new_min, new_max = new_range
            
            # Guard clause: If the domain isn't changing significantly, do nothing.
            if abs(self.in_min - new_min) < 1e-6 and abs(self.in_max - new_max) < 1e-6:
                return

            # --- Spline Resampling ---
            # For Phi splines (non-monotonic), preserving shape is crucial.
            # For monotonic splines (phi), strict shape preservation is less critical if they just map input to [0,1],
            # but we still do it to maintain optimization progress.
            if (self.total_evaluations > 0 or force_resample) and allow_resampling:
                # Capture old state
                old_knots = self.knots.clone()
                old_coeffs = self.get_coeffs().clone()
                old_min, old_max = self.in_min, self.in_max
                
                # Define evaluator for the OLD spline
                def old_spline_eval(x_vals):
                    return self._evaluate_spline_logic(x_vals, old_coeffs, old_min, old_max, old_knots)

                # Determine target node positions (Greville abscissae) in the NEW domain
                new_nodes = self._get_greville_points(new_min, new_max, self.num_knots)
                
                # Sample the old curve at the new locations
                sampled_values = old_spline_eval(new_nodes)

                # Convert sampled values to control points (Inverse B-Spline transform)
                # This prevents "smoothing degradation" over many updates
                new_coeffs = self._values_to_control_points(sampled_values)

                # Update parameters
                if self.monotonic:
                    # Normalize to [0, 1]
                    val_clamped = torch.clamp(new_coeffs, 0.0, 1.0)
                    
                    # Ensure strict monotonicity for the decomposition
                    val_sorted, _ = torch.sort(val_clamped)
                    
                    # Compute increments
                    increments = torch.cat([val_sorted[:1], val_sorted[1:] - val_sorted[:-1]])
                    increments = torch.clamp(increments, min=1e-8) # Avoid log(0)
                    
                    # Store as log increments
                    self.log_increments.data = torch.log(increments)
                else:
                    self.coeffs.data = new_coeffs

            # --- Update internal state ---
            self.in_min, self.in_max = new_min, new_max
            
            # Apply safety margin
            if CONFIG.get('domain_safety_margin', 0.0) > 0:
                margin = CONFIG['domain_safety_margin'] * (self.in_max - self.in_min)
                self.in_min -= margin
                self.in_max += margin
            
            self.knots.data = torch.linspace(self.in_min, self.in_max, self.num_knots, 
                                           device=self.knots.device, dtype=self.knots.dtype)

    def initialize_as_identity(self, domain_min, domain_max):
        """Initialize spline coefficients to approximate identity function."""
        with torch.no_grad():
            if self.monotonic:
                self.log_increments.data = torch.log(torch.ones(self.num_knots) / self.num_knots + 1e-6)
            else:
                # For B-splines, setting coeffs to linear ramp approximates identity (exact for p=1)
                self.coeffs.data = torch.linspace(domain_min, domain_max, self.num_knots, 
                                                device=self.coeffs.device, dtype=self.coeffs.dtype)
    
    def get_coeffs(self):
        """Get actual spline coefficients."""
        if self.monotonic:
            increments = torch.nn.functional.softplus(self.log_increments)
            cumulative = torch.cumsum(increments, 0)
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
                # B-spline convex hull property: curve contained in range of control points
                if self.train_codomain and self.codomain_params is not None:
                    cc = self.codomain_params.cc.item()
                    cr = self.codomain_params.cr.item()
                    # Transform the range of the coefficients
                    y_raw_min = coeffs.min().item()
                    y_raw_max = coeffs.max().item()
                    
                    # The transformation is y_final = cc - cr + 2*cr * (y_raw - coeff_min)/(coeff_max - coeff_min)
                    # Since we normalize using the CURRENT coeff min/max in forward(), 
                    # the output range effectively spans [cc-cr, cc+cr] if the input spans the knots.
                    return TheoreticalRange(cc - cr, cc + cr)
                else:
                    return TheoreticalRange(coeffs.min().item(), coeffs.max().item())

    def _evaluate_spline_logic(self, x, coeffs, in_min, in_max, knots):
        """
        Core evaluation logic separated from parameter access to support resampling.
        Handles both Order 1 (PWL) and Higher Order (B-splines).
        """
        # Handle out-of-domain (same logic for all orders: linear extension)
        below_domain = x < in_min
        above_domain = x > in_max
        x_clamped = torch.clamp(x, in_min, in_max)

        if self.order == 1:
            # --- Piecewise Linear (Original Code) ---
            intervals = torch.searchsorted(knots, x_clamped) - 1
            intervals = torch.clamp(intervals, 0, self.num_knots - 2)
            
            knot_spacing = knots[intervals + 1] - knots[intervals]
            t = torch.where(knot_spacing > 1e-8,
                            (x_clamped - knots[intervals]) / knot_spacing,
                            torch.zeros_like(x_clamped))
            
            interpolated = (1 - t) * coeffs[intervals] + t * coeffs[intervals + 1]
            
            # Slopes for extension
            left_slope = (coeffs[1] - coeffs[0]) / (knots[1] - knots[0] + 1e-8)
            right_slope = (coeffs[-1] - coeffs[-2]) / (knots[-1] - knots[-2] + 1e-8)
            
        elif self.order == 3:
            # --- Cubic B-Spline (Optimized) ---
            M = self.num_knots - 1
            grid_len = in_max - in_min
            delta = grid_len / M if M > 0 else 1.0
            
            # Normalize x to grid coordinates
            u = (x_clamped - in_min) / (delta + 1e-8)
            
            # Indices
            idx = torch.floor(u).long()
            idx = torch.clamp(idx, 0, M - 1)
            
            # Local parameter t in [0, 1]
            t_local = u - idx
            
            # Indices for coefficients (clamped to valid range)
            # For a uniform B-spline centered at idx, we need idx-1, idx, idx+1, idx+2
            i0 = torch.clamp(idx - 1, 0, self.num_knots - 1)
            i1 = torch.clamp(idx,     0, self.num_knots - 1)
            i2 = torch.clamp(idx + 1, 0, self.num_knots - 1)
            i3 = torch.clamp(idx + 2, 0, self.num_knots - 1)
            
            c0, c1, c2, c3 = coeffs[i0], coeffs[i1], coeffs[i2], coeffs[i3]
            
            # Basis calculation
            t2 = t_local * t_local
            t3 = t2 * t_local
            
            w0 = (1.0 - 3.0*t_local + 3.0*t2 - t3) / 6.0
            w1 = (4.0 - 6.0*t2 + 3.0*t3) / 6.0
            w2 = (1.0 + 3.0*t_local + 3.0*t2 - 3.0*t3) / 6.0
            w3 = t3 / 6.0
            
            interpolated = w0*c0 + w1*c1 + w2*c2 + w3*c3
            
            # Derivatives for slope extension
            # Left boundary (idx=0, t=0)
            li0, li1, li2, li3 = 0, 0, 1, 2
            lc0, lc1, lc2, lc3 = coeffs[li0], coeffs[li1], coeffs[li2], coeffs[li3]
            # w' at t=0: -0.5, 0, 0.5, 0
            left_deriv_local = -0.5*lc0 + 0.5*lc2
            left_slope = left_deriv_local / (delta + 1e-8)
            
            # Right boundary (idx=M-1, t=1)
            ri0 = torch.clamp(torch.tensor(M-2), 0, self.num_knots-1)
            ri1 = torch.clamp(torch.tensor(M-1), 0, self.num_knots-1)
            ri2 = torch.clamp(torch.tensor(M),   0, self.num_knots-1)
            ri3 = torch.clamp(torch.tensor(M+1), 0, self.num_knots-1)
            rc0, rc1, rc2, rc3 = coeffs[ri0], coeffs[ri1], coeffs[ri2], coeffs[ri3]
            # w' at t=1: 0, -0.5, 0, 0.5
            right_deriv_local = -0.5*rc1 + 0.5*rc3
            right_slope = right_deriv_local / (delta + 1e-8)

        else:
             # Fallback for other orders
             raise NotImplementedError(f"Spline order {self.order} not implemented (only 1 and 3 supported)")

        # --- Apply extension and Codomain (Common) ---
        
        if self.monotonic:
            # Monotonic splines: constant extension
            result = torch.where(below_domain, torch.zeros_like(x), interpolated)
            result = torch.where(above_domain, torch.ones_like(x), result)
        else:
            # General splines: Linear extension
            if self.train_codomain and self.codomain_params is not None:
                cc = self.codomain_params.cc
                cr = self.codomain_params.cr
                coeff_min = coeffs.min()
                coeff_max = coeffs.max()
                
                # Normalize result
                denom = coeff_max - coeff_min + 1e-8
                normalized = (interpolated - coeff_min) / denom
                result = cc - cr + 2 * cr * normalized
                
                # Scale slopes
                scale_factor = 2 * cr / denom
                left_slope_scaled = left_slope * scale_factor
                right_slope_scaled = right_slope * scale_factor
                
                result = torch.where(below_domain,
                                   result + left_slope_scaled * (x - in_min),
                                   result)
                result = torch.where(above_domain,
                                   result + right_slope_scaled * (x - in_max),
                                   result)
            else:
                # Fixed codomain / raw coeffs
                result = torch.where(below_domain,
                                   interpolated + left_slope * (x - in_min),
                                   interpolated)
                result = torch.where(above_domain,
                                   interpolated + right_slope * (x - in_max),
                                   result)
        
        return result

    def forward(self, x):
        x = x.to(self.knots.device)
        
        # Track domain violations if debugging
        if CONFIG.get('track_domain_violations', False):
            self.total_evaluations += x.numel()
            violations = ((x < self.in_min) | (x > self.in_max)).sum().item()
            self.domain_violations += violations
            if violations > 0 and CONFIG.get('verbose_domain_violations', False):
                print(f"Domain violation: {violations}/{x.numel()} values outside [{self.in_min:.3f}, {self.in_max:.3f}]")
        
        coeffs = self.get_coeffs()
        return self._evaluate_spline_logic(x, coeffs, self.in_min, self.in_max, self.knots)
    
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
            # For monotonic splines, save log_increments
            state['log_increments'] = self.log_increments.data.clone()
        else:
            # For non-monotonic splines, save raw coefficients
            state['raw_coeffs'] = self.coeffs.data.clone()
            
        return state
    
    def set_domain_state(self, state):
        """Restore domain state from checkpoint."""
        self.knots.data = state['knots']
        self.in_min = state['in_min']
        self.in_max = state['in_max']
        
        # Restore raw parameters to ensure exact restoration
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
    A single Sprecher block that transforms d_in -> d_out.
    
    Args:
        spline_order: Order of splines (1=Piecewise Linear, 3=Cubic B-Spline)
    """
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=100, Phi_knots=100,
                 phi_domain=None, Phi_domain=None, spline_order=1):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final
        self.spline_order = spline_order
        
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
        phi_in_range = phi_domain if phi_domain is not None else (0, 1)
        self.phi = SimpleSpline(
            num_knots=phi_knots,
            in_range=phi_in_range,
            out_range=(0, 1),
            monotonic=True,
            order=spline_order  # Pass order
        )
        
        # The outer general spline Φ
        Phi_in_range = Phi_domain if Phi_domain is not None else (0, 1)
        self.Phi = SimpleSpline(
            num_knots=Phi_knots,
            in_range=Phi_in_range,
            out_range=(0, 1),
            train_codomain=CONFIG['train_phi_codomain'],
            codomain_params=self.phi_codomain_params,
            order=spline_order  # Pass order
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
        """Apply sign-aware lateral mixing bounds to per-q intervals."""
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
            self.Phi.initialize_as_identity(domain_min, domain_max)
            
            if CONFIG.get('debug_domains', False):
                print(f"Layer {self.layer_num}: Initialized Phi codomain to match domain [{domain_min:.3f}, {domain_max:.3f}]")
                print(f"  cc = {domain_center:.3f}, cr = {domain_radius:.3f}")
    
    def compute_pooling_residual_bounds_per_output(self, input_interval, a_per_dim=None, b_per_dim=None):
        """
        Compute per-output pooling residual intervals for composition.
        Returns: r_min_per_q, r_max_per_q (tensor shape [d_out])
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
                # Contribution of each input (min/max depending on weight sign)
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
        STANDARD residual bounds for projection matrix W (d_in x d_out).
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
        """Compute theoretical output range of this block using per-q tight composition."""
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
                # Fallback: use global bounds
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
            
            # Lateral mixing
            if self.lateral_scale is not None:
                s_min_per_q, s_max_per_q = self._apply_lateral_mixing_bounds_per_q(s_min_per_q, s_max_per_q)
            
            # Per-q Φ output ranges (evaluate endpoints + interior knots)
            Phi_at_smin = self.Phi(s_min_per_q)
            Phi_at_smax = self.Phi(s_max_per_q)
            y_min_per_q = torch.minimum(Phi_at_smin, Phi_at_smax)
            y_max_per_q = torch.maximum(Phi_at_smin, Phi_at_smax)
            
            # Check interior knots per q (Approximate for B-splines, exact for PWL)
            # For B-splines, extrema are not strictly at knots, but near control points.
            # Use control point convex hull property for safe bounds.
            if self.spline_order == 1:
                # PWL: check knots
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
            else:
                # Higher order: Use convex hull property of control points within domain
                # Simplified: Use global output range of the spline (safe upper bound)
                global_range = self.Phi.get_actual_output_range()
                
                # B-spline Variation Diminishing property:
                # The curve is contained in convex hull of control points.
                # Just using the min/max of ALL coeffs is valid and safe.
                tr = self.Phi.get_actual_output_range()
                
                y_min_per_q = torch.full_like(y_min_per_q, tr.min)
                y_max_per_q = torch.full_like(y_max_per_q, tr.max)
            
            # Residuals per q
            if CONFIG['use_residual_weights'] and self.input_range is not None:
                if self.residual_weight is not None:
                    # Scalar residual per output (d_in == d_out)
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
                        # Fallback
                        a, b = self.input_range.min, self.input_range.max
                        r_lo, r_hi = (w * a, w * b) if w.item() >= 0 else (w * b, w * a)
                        r_min_per_q = torch.full((self.d_out,), r_lo.item(), device=device)
                        r_max_per_q = torch.full((self.d_out,), r_hi.item(), device=device)
                elif self.residual_projection is not None:
                    # STANDARD linear projection
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
                        a, b = self.input_range.min, self.input_range.max
                        r_min_per_q, r_max_per_q = self.compute_pooling_residual_bounds_per_output(Interval(a, b))
                        r_min_per_q = r_min_per_q.to(device)
                        r_max_per_q = r_max_per_q.to(device)
                elif self.residual_broadcast_weights is not None:
                    w = self.residual_broadcast_weights.to(device)
                    src = self.broadcast_sources.to(device)
                    if self.input_min_per_dim is not None and self.input_max_per_dim is not None:
                        a_vec = torch.as_tensor(self.input_min_per_dim, device=device, dtype=torch.float32)
                        b_vec = torch.as_tensor(self.input_max_per_dim, device=device, dtype=torch.float32)
                        if a_vec.numel() != self.d_in or b_vec.numel() != self.d_in:
                            a_vec = torch.full((self.d_in,), self.input_range.min, device=device)
                            b_vec = torch.full((self.d_in,), self.input_range.max, device=device)
                        a_src = a_vec[src]
                        b_src = b_vec[src]
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
            
            # Compose Φ and residual per-q
            final_min_per_q = y_min_per_q + r_min_per_q
            final_max_per_q = y_max_per_q + r_max_per_q

            # Store per-q outputs
            self.output_min_per_q = final_min_per_q.detach().clone()
            self.output_max_per_q = final_max_per_q.detach().clone()

            # Aggregate to layer output_range
            if self.is_final:
                final_min = final_min_per_q.sum().item()
                final_max = final_max_per_q.sum().item()
            else:
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
        ranges['codomain_initialized'] = self.codomain_initialized_from_domain
        return ranges
    
    def set_theoretical_ranges(self, ranges):
        """Restore theoretical ranges from checkpoint."""
        if 'input_range' in ranges:
            self.input_range = TheoreticalRange(ranges['input_range'][0], ranges['input_range'][1])
        if 'output_range' in ranges:
            self.output_range = TheoreticalRange(ranges['output_range'][0], ranges['output_range'][1])
        if 'codomain_initialized' in ranges:
            self.codomain_initialized_from_domain = ranges['codomain_initialized']
    
    def forward(self, x, x_original=None):
        """Forward pass with configurable memory mode."""
        if CONFIG.get('low_memory_mode', False):
            return self._forward_memory_efficient(x, x_original)
        else:
            return self._forward_original(x, x_original)
    
    def _forward_original(self, x, x_original=None):
        """Original forward pass - O(B × d_in × d_out) memory."""
        q_on_device = self.q_values.to(x.device)

        x_expanded = x.unsqueeze(-1)  # (batch_size, d_in, 1)
        q = q_on_device.view(1, 1, -1)  # (1, 1, d_out)
        
        # Apply translation by η * q
        shifted = x_expanded + self.eta * q
        
        # Apply inner monotonic spline φ
        phi_out = self.phi(shifted)  # (batch_size, d_in, d_out)
        
        # Weight by λ and sum over input dimension
        weighted = phi_out * self.lambdas.view(1, -1, 1)
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * q_on_device  # (batch_size, d_out)
        
        # Apply lateral mixing
        if self.lateral_scale is not None:
            s = self._apply_lateral_mixing_to_s(s, x.device)
        
        # Apply Phi
        activated = self.Phi(s)
        
        # Add residual connections
        activated = self._add_residual(activated, x_original)
        
        if self.is_final:
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated
    
    def _forward_memory_efficient(self, x, x_original=None):
        """Memory-efficient forward pass - O(B × max(d_in, d_out)) memory."""
        batch_size = x.shape[0]
        device = x.device
        
        s = torch.zeros(batch_size, self.d_out, device=device, dtype=x.dtype)
        
        for q_idx in range(self.d_out):
            q_val = float(q_idx)
            shifted = x + self.eta * q_val
            phi_out = self.phi(shifted)
            weighted = phi_out * self.lambdas
            s[:, q_idx] = weighted.sum(dim=1) + Q_VALUES_FACTOR * q_val
        
        if self.lateral_scale is not None:
            s = self._apply_lateral_mixing_to_s(s, device)
        
        activated = self.Phi(s)
        
        activated = self._add_residual(activated, x_original)
        
        if self.is_final:
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated
    
    def _apply_lateral_mixing_to_s(self, s, device):
        """Apply lateral mixing to s."""
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
            activated = activated + torch.matmul(x_original, self.residual_projection)
        elif self.residual_weight is not None:
            activated = activated + self.residual_weight * x_original
        elif self.residual_pooling_weights is not None:
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
            residual_contribution = torch.gather(
                x_original, 1,
                self.broadcast_sources.unsqueeze(0).expand(x_original.shape[0], -1)
            )
            residual_contribution = residual_contribution * self.residual_broadcast_weights.view(1, -1)
            activated = activated + residual_contribution
        
        return activated
    
    def get_output_range(self):
        """Get the current output range of this block."""
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
    Builds the Sprecher network with a given hidden-layer architecture.
    
    Args:
        spline_order: Order of splines (1=Piecewise Linear, 3=Cubic B-Spline)
    """
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=100, Phi_knots=100,
                 norm_type='none', norm_position='after', norm_skip_first=True, initialize_domains=True,
                 domain_ranges=None, spline_order=1):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.norm_skip_first = norm_skip_first
        self.spline_order = spline_order
        
        layers = []
        if not architecture:
            is_final = (final_dim == 1)
            phi_domain = domain_ranges.get('layer_0_phi') if domain_ranges else None
            Phi_domain = domain_ranges.get('layer_0_Phi') if domain_ranges else None
            layers.append(SprecherLayerBlock(
                d_in=input_dim, d_out=final_dim,
                layer_num=0, is_final=is_final,
                phi_knots=phi_knots, Phi_knots=Phi_knots,
                phi_domain=phi_domain, Phi_domain=Phi_domain,
                spline_order=spline_order
            ))
        else:
            L = len(architecture)
            d_in = input_dim
            for i in range(L):
                d_out = architecture[i]
                is_final_block = (i == L - 1) and (self.final_dim == 1)
                phi_domain = domain_ranges.get(f'layer_{i}_phi') if domain_ranges else None
                Phi_domain = domain_ranges.get(f'layer_{i}_Phi') if domain_ranges else None
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=d_out,
                    layer_num=i, is_final=is_final_block,
                    phi_knots=phi_knots, Phi_knots=Phi_knots,
                    phi_domain=phi_domain, Phi_domain=Phi_domain,
                    spline_order=spline_order
                ))
                d_in = d_out
            
            if self.final_dim > 1:
                phi_domain = domain_ranges.get(f'layer_{L}_phi') if domain_ranges else None
                Phi_domain = domain_ranges.get(f'layer_{L}_Phi') if domain_ranges else None
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=self.final_dim,
                    layer_num=L, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots,
                    phi_domain=phi_domain, Phi_domain=Phi_domain,
                    spline_order=spline_order
                ))
        
        self.layers = nn.ModuleList(layers)
        
        # Create normalization layers
        self.norm_layers = nn.ModuleList()
        if norm_type != 'none':
            for i, layer in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    if norm_position == 'before':
                        num_features = layer.d_in
                    else:
                        num_features = 1 if layer.is_final else layer.d_out
                    
                    if norm_type == 'batch':
                        self.norm_layers.append(nn.BatchNorm1d(num_features))
                    elif norm_type == 'layer':
                        self.norm_layers.append(nn.LayerNorm(num_features))
                    else:
                        raise ValueError(f"Unknown norm_type: {norm_type}")
        
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
        if initialize_domains and CONFIG.get('use_theoretical_domains', True):
            self.update_all_domains()
    
    def update_all_domains(self, allow_resampling=True, force_resample=False):
        """Update all spline domains based on theoretical bounds."""
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

        current_interval = Interval(0.0, 1.0)
        current_min_per_dim = torch.zeros(self.input_dim, device=device)
        current_max_per_dim = torch.ones(self.input_dim, device=device)
        
        for layer_idx, layer in enumerate(self.layers):
            # Normalization BEFORE
            if self.norm_type != 'none' and self.norm_position == 'before':
                norm_layer = self.norm_layers[layer_idx]
                if not isinstance(norm_layer, nn.Identity):
                    if isinstance(norm_layer, nn.BatchNorm1d):
                        current_interval = compute_batchnorm_bounds(
                            current_interval, norm_layer, training_mode=self.training
                        )
                        current_min_per_dim = torch.full((layer.d_in,), current_interval.min, device=device)
                        current_max_per_dim = torch.full((layer.d_in,), current_interval.max, device=device)
                    elif isinstance(norm_layer, nn.LayerNorm):
                        current_interval = compute_layernorm_bounds(
                            current_interval, norm_layer, layer.d_in
                        )
                        current_min_per_dim = torch.full((layer.d_in,), current_interval.min, device=device)
                        current_max_per_dim = torch.full((layer.d_in,), current_interval.max, device=device)
            
            layer.input_min_per_dim = current_min_per_dim.detach().clone().to(layer.phi.knots.device)
            layer.input_max_per_dim = current_max_per_dim.detach().clone().to(layer.phi.knots.device)

            current_range = TheoreticalRange(current_interval.min, current_interval.max)
            layer.update_phi_domain_theoretical(current_range, allow_resampling, force_resample)
            layer.update_Phi_domain_theoretical(allow_resampling, force_resample)
            layer.compute_output_range_theoretical()
            
            if layer.is_final:
                next_min_per_dim = torch.tensor([layer.output_range.min], device=device)
                next_max_per_dim = torch.tensor([layer.output_range.max], device=device)
            else:
                next_min_per_dim = layer.output_min_per_q.to(device)
                next_max_per_dim = layer.output_max_per_q.to(device)
            
            output_interval = Interval(layer.output_range.min, layer.output_range.max)
            
            # Normalization AFTER
            if self.norm_type != 'none' and self.norm_position == 'after':
                norm_layer = self.norm_layers[layer_idx]
                if not isinstance(norm_layer, nn.Identity):
                    if isinstance(norm_layer, nn.BatchNorm1d):
                        output_interval = compute_batchnorm_bounds(
                            output_interval, norm_layer, training_mode=self.training
                        )
                        if not self.training:
                            eps = norm_layer.eps
                            running_mean = norm_layer.running_mean.to(device)
                            running_var = norm_layer.running_var.to(device)
                            std = torch.sqrt(running_var + eps)
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
                            next_min_per_dim = torch.full_like(next_min_per_dim, output_interval.min)
                            next_max_per_dim = torch.full_like(next_max_per_dim, output_interval.max)
                    elif isinstance(norm_layer, nn.LayerNorm):
                        num_features = 1 if layer.is_final else layer.d_out
                        output_interval = compute_layernorm_bounds(
                            output_interval, norm_layer, num_features
                        )
                        next_min_per_dim = torch.full_like(next_min_per_dim, output_interval.min)
                        next_max_per_dim = torch.full_like(next_max_per_dim, output_interval.max)
            
            current_min_per_dim = next_min_per_dim
            current_max_per_dim = next_max_per_dim
            current_interval = Interval(current_min_per_dim.min().item(), current_max_per_dim.max().item())
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.layers):
            if self.norm_type != 'none' and self.norm_position == 'before':
                x_in = self.norm_layers[i](x_in)
            
            x_out = layer(x_in, x_in) 
            
            if self.norm_type != 'none' and self.norm_position == 'after':
                x_out = self.norm_layers[i](x_out)
            
            x_in = x_out
        
        x = x_in
        x = self.output_scale * x + self.output_bias
        return x
    
    def get_domain_violation_stats(self):
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_domain_violation_stats()
        return stats
    
    def reset_domain_violation_stats(self):
        for layer in self.layers:
            layer.reset_domain_violation_stats()
    
    def print_domain_violation_report(self):
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
        domain_states = {}
        for i, layer in enumerate(self.layers):
            domain_states[f'layer_{i}_phi'] = layer.phi.get_domain_state()
            domain_states[f'layer_{i}_Phi'] = layer.Phi.get_domain_state()
            domain_states[f'layer_{i}_ranges'] = layer.get_theoretical_ranges()
        return domain_states
    
    def get_domain_ranges(self):
        domain_ranges = {}
        for i, layer in enumerate(self.layers):
            domain_ranges[f'layer_{i}_phi'] = (layer.phi.in_min, layer.phi.in_max)
            domain_ranges[f'layer_{i}_Phi'] = (layer.Phi.in_min, layer.Phi.in_max)
        return domain_ranges
    
    def set_all_domain_states(self, domain_states):
        for i, layer in enumerate(self.layers):
            phi_key = f'layer_{i}_phi'
            Phi_key = f'layer_{i}_Phi'
            ranges_key = f'layer_{i}_ranges'
            
            if phi_key in domain_states:
                layer.phi.set_domain_state(domain_states[phi_key])
            if Phi_key in domain_states:
                layer.Phi.set_domain_state(domain_states[Phi_key])
            if ranges_key in domain_states:
                layer.set_theoretical_ranges(domain_states[ranges_key])


def test_domain_tightness(model, dataset, n_samples=10000):
    """Test that computed domains are both safe and tight."""
    device = next(model.parameters()).device
    
    x_test = torch.rand(n_samples, dataset.input_dim, device=device)
    
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
                
                if actual_min < module.in_min - 1e-6 or actual_max > module.in_max + 1e-6:
                    print(f"SAFETY VIOLATION in {name}:")
                    print(f"  Computed: [{module.in_min:.6f}, {module.in_max:.6f}]")
                    print(f"  Actual:   [{actual_min:.6f}, {actual_max:.6f}]")
        return hook
    
    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.phi.register_forward_hook(hook_fn(f"layer_{i}_phi")))
        handles.append(layer.Phi.register_forward_hook(hook_fn(f"layer_{i}_Phi")))
    
    with torch.no_grad():
        for batch_start in range(0, n_samples, 100):
            batch_end = min(batch_start + 100, n_samples)
            _ = model(x_test[batch_start:batch_end])
    
    for handle in handles:
        handle.remove()
    
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