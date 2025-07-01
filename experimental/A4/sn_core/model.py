"""Sprecher Network model components: splines, layers, and networks."""

import torch
import torch.nn as nn
import numpy as np
from .config import CONFIG, Q_VALUES_FACTOR


class TheoreticalRange:
    """Tracks theoretical min/max values for a layer's input/output."""
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def __repr__(self):
        return f"TheoreticalRange({self.min:.3f}, {self.max:.3f})"
    
    def contains(self, value):
        """Check if a value is within this range."""
        return self.min <= value <= self.max


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


class AffineNormalization(nn.Module):
    """
    Affine normalization with exact computable bounds.
    y = weight * x + bias
    
    Key properties:
    - Exact output bounds computable from input bounds
    - Learnable per-feature weights and biases
    - Identity initialization option for stability
    """
    def __init__(self, num_features, eps=1e-8, identity_init=True):
        """
        Args:
            num_features: Number of features to normalize
            eps: Small constant for numerical stability
            identity_init: If True, initialize close to identity
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.identity_init = identity_init
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Track expected input range for smart initialization
        self.register_buffer('expected_input_min', torch.zeros(1))
        self.register_buffer('expected_input_max', torch.ones(1))
        
    def update_expected_input_range(self, input_min, input_max):
        """Update expected input range and optionally reinitialize."""
        self.expected_input_min = torch.tensor(float(input_min))
        self.expected_input_max = torch.tensor(float(input_max))
        
        if self.identity_init and not self.training:
            # Initialize to approximate identity: map [input_min, input_max] → [input_min, input_max]
            with torch.no_grad():
                # For identity: output = weight * input + bias
                # We want: input_min → input_min and input_max → input_max
                # This gives us: weight = 1, bias = 0 (already initialized correctly)
                # But we might want to scale down large ranges
                input_range = input_max - input_min
                if input_range > 10.0:  # Large range, scale down
                    target_range = 2.0  # Map to [-1, 1]
                    self.weight.data.fill_(target_range / (input_range + self.eps))
                    self.bias.data.fill_(-1.0 - self.weight.data.item() * input_min)
    
    def get_exact_output_bounds(self, input_min, input_max):
        """
        Compute EXACT output bounds for affine transformation.
        
        For y = w*x + b:
        - If w > 0: y ∈ [w*input_min + b, w*input_max + b]  
        - If w < 0: y ∈ [w*input_max + b, w*input_min + b]
        - Handle each feature separately, then take global min/max
        """
        with torch.no_grad():
            # Compute outputs at boundary points
            output_at_min = self.weight * input_min + self.bias
            output_at_max = self.weight * input_max + self.bias
            
            # Handle positive and negative weights
            feature_mins = torch.where(self.weight >= 0, output_at_min, output_at_max)
            feature_maxs = torch.where(self.weight >= 0, output_at_max, output_at_min)
            
            # Global bounds across all features
            global_min = feature_mins.min().item()
            global_max = feature_maxs.max().item()
            
            return global_min, global_max
    
    def forward(self, x):
        """Simple affine transformation."""
        return self.weight * x + self.bias


class ExactLayerNorm(nn.Module):
    """
    LayerNorm with EXACT computable output bounds.
    For any input in [a, b], we can compute exact output bounds.
    """
    def __init__(self, num_features, eps=1e-5, identity_init=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.identity_init = identity_init
        
        # Standard LayerNorm parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # For tracking expected input range
        self.register_buffer('expected_input_min', torch.zeros(1))
        self.register_buffer('expected_input_max', torch.ones(1))
        
    def update_expected_input_range(self, input_min, input_max):
        """Update expected input range for identity initialization."""
        self.expected_input_min = torch.tensor(float(input_min))
        self.expected_input_max = torch.tensor(float(input_max))
        
        if self.identity_init and not self.training:
            # Initialize to approximate identity based on expected range
            with torch.no_grad():
                input_center = (input_min + input_max) / 2
                input_scale = (input_max - input_min) / 2
                # Set gamma to preserve scale, beta to preserve center
                self.gamma.data.fill_(input_scale)
                self.beta.data.fill_(input_center)
    
    def get_exact_output_bounds(self, input_min, input_max):
        """
        Compute EXACT output bounds for LayerNorm.
        
        Key insight: For LayerNorm with all features in [input_min, input_max]:
        - When all features are equal: normalized = 0, output = beta
        - When features maximally vary: normalized in [-1, 1], output in [beta - |gamma|, beta + |gamma|]
        """
        with torch.no_grad():
            # Exact bounds based on LayerNorm properties
            output_min = (self.beta - torch.abs(self.gamma)).min().item()
            output_max = (self.beta + torch.abs(self.gamma)).max().item()
            
            return output_min, output_max
    
    def forward(self, x):
        # Standard LayerNorm computation
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline.
    
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
        if monotonic:
            # For monotonic splines, use log-space parameters to ensure monotonicity
            self.log_increments = nn.Parameter(torch.zeros(num_knots))
            with torch.no_grad():
                # Initialize to approximate linear function
                self.log_increments.data = torch.log(torch.ones(num_knots) / num_knots + 1e-6)
        else:
            # For general splines (Φ), initialize to zeros
            # Will be properly initialized when domain is computed
            self.coeffs = nn.Parameter(torch.zeros(num_knots))
    
    def update_domain(self, new_range):
        """
        Update the domain of the spline by adjusting knot positions.
        For non-monotonic splines (Φ), this method resamples the spline to preserve
        its learned functional shape in the new domain, preventing information loss
        during training.
        """
        with torch.no_grad():
            new_min, new_max = new_range

            # Guard clause: If the domain isn't changing significantly, do nothing.
            # This prevents floating point drift and unnecessary computation.
            if abs(self.in_min - new_min) < 1e-6 and abs(self.in_max - new_max) < 1e-6:
                return

            # --- Spline Resampling to Preserve Learned Shape (for Φ splines) ---
            # This is the core fix for training instability.
            # It's applied only to the general (non-monotonic) splines after the
            # first training step has occurred.
            if not self.monotonic and self.total_evaluations > 0:
                # Store the state of the old spline before changing anything
                old_knots = self.knots.clone()
                old_coeffs = self.get_coeffs().clone()
                
                # Define a helper function to evaluate the OLD spline at any point.
                # This function captures the learned shape we want to preserve.
                def old_spline_eval(x_vals):
                    # Clamp inputs to the old domain for robust interpolation.
                    x_clamped = torch.clamp(x_vals, old_knots[0], old_knots[-1])
                    
                    intervals = torch.searchsorted(old_knots, x_clamped) - 1
                    intervals = torch.clamp(intervals, 0, self.num_knots - 2)
                    
                    knot_spacing = old_knots[intervals + 1] - old_knots[intervals]
                    t = torch.where(knot_spacing > 1e-8,
                                    (x_clamped - old_knots[intervals]) / knot_spacing,
                                    torch.zeros_like(x_clamped))
                    
                    interpolated = (1 - t) * old_coeffs[intervals] + t * old_coeffs[intervals + 1]
                    return interpolated

                # Define the new knot positions in the new domain
                new_knots_pos = torch.linspace(new_min, new_max, self.num_knots, 
                                               device=self.knots.device, dtype=self.knots.dtype)
                
                # Resample: Evaluate the OLD spline at the NEW knot positions
                resampled_coeffs = old_spline_eval(new_knots_pos)

                # Update the learnable parameters with the resampled values
                self.coeffs.data = resampled_coeffs

            # --- Update the internal state for the new domain ---
            self.in_min, self.in_max = new_min, new_max

            # Add safety margin only if explicitly configured (default is 0)
            if CONFIG.get('domain_safety_margin', 0.0) > 0:
                margin = CONFIG['domain_safety_margin'] * (self.in_max - self.in_min)
                self.in_min -= margin
                self.in_max += margin
            
            self.knots.data = torch.linspace(self.in_min, self.in_max, self.num_knots, 
                                           device=self.knots.device, dtype=self.knots.dtype)

    def initialize_as_identity(self, domain_min, domain_max):
        """Initialize spline coefficients to approximate identity function on given domain."""
        with torch.no_grad():
            if self.monotonic:
                # For monotonic splines, keep uniform increments
                self.log_increments.data = torch.log(torch.ones(self.num_knots) / self.num_knots + 1e-6)
            else:
                # For general splines, set coefficients to be linear from domain_min to domain_max
                self.coeffs.data = torch.linspace(domain_min, domain_max, self.num_knots, 
                                                device=self.coeffs.device, dtype=self.coeffs.dtype)
    
    def get_coeffs(self):
        """Get the actual spline coefficients (for both monotonic and non-monotonic)"""
        if self.monotonic:
            # Use softplus to ensure positive increments, then cumsum for monotonicity
            increments = torch.nn.functional.softplus(self.log_increments)
            cumulative = torch.cumsum(increments, 0)
            # Normalize to [0, 1] range
            cumulative = cumulative / (cumulative[-1] + 1e-8)
            # For monotonic splines, output range is always [0, 1]
            return cumulative
        else:
            # For non-monotonic splines, return coefficients directly
            # They will be scaled by codomain if trainable
            return self.coeffs
    
    def get_actual_output_range(self):
        """Compute the actual range of outputs this spline can produce."""
        with torch.no_grad():
            coeffs = self.get_coeffs()
            
            if self.monotonic:
                # Monotonic splines always output [0, 1]
                return TheoreticalRange(0.0, 1.0)
            else:
                # For Φ splines, the piecewise linear function achieves its extrema at knots
                # So we just need to check the coefficient values
                if self.train_codomain and self.codomain_params is not None:
                    # Apply codomain transformation to coefficients
                    cc = self.codomain_params.cc.item()
                    cr = self.codomain_params.cr.item()
                    coeff_min = coeffs.min().item()
                    coeff_max = coeffs.max().item()
                    
                    # Transform each coefficient
                    normalized_coeffs = (coeffs - coeff_min) / (coeff_max - coeff_min + 1e-8)
                    transformed_coeffs = cc - cr + 2 * cr * normalized_coeffs
                    
                    # True min/max are among the transformed coefficients
                    actual_min = transformed_coeffs.min().item()
                    actual_max = transformed_coeffs.max().item()
                else:
                    # No transformation needed - min/max are just coefficient extrema
                    actual_min = coeffs.min().item()
                    actual_max = coeffs.max().item()
                
                return TheoreticalRange(actual_min, actual_max)
    
    def forward(self, x):
        x = x.to(self.knots.device)
        
        # Track domain violations if debugging
        if CONFIG.get('track_domain_violations', False):
            self.total_evaluations += x.numel()
            violations = ((x < self.in_min) | (x > self.in_max)).sum().item()
            self.domain_violations += violations
            if violations > 0 and CONFIG.get('verbose_domain_violations', False):
                print(f"Domain violation: {violations}/{x.numel()} values outside [{self.in_min:.3f}, {self.in_max:.3f}]")
        
        # Handle out-of-domain inputs by extending the spline linearly
        # This is more theoretically sound than clamping
        below_domain = x < self.in_min
        above_domain = x > self.in_max
        
        # For in-domain values, proceed as normal
        x_clamped = torch.clamp(x, self.in_min, self.in_max)
        
        # Find which segment each input value belongs to
        intervals = torch.searchsorted(self.knots, x_clamped) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)
        
        # Compute interpolation parameter t in [0,1]
        knot_spacing = self.knots[intervals + 1] - self.knots[intervals]
        # Avoid division by zero for very small domains
        t = torch.where(knot_spacing > 1e-8,
                        (x_clamped - self.knots[intervals]) / knot_spacing,
                        torch.zeros_like(x_clamped))
        
        # Get coefficients
        coeffs = self.get_coeffs()
        
        # Linear interpolation
        interpolated = (1 - t) * coeffs[intervals] + t * coeffs[intervals + 1]
        
        # For monotonic splines, handle out-of-domain linearly based on boundary slopes
        if self.monotonic:
            # Extend with slope 0 outside domain (keeps values at 0 or 1)
            result = torch.where(below_domain, torch.zeros_like(x), interpolated)
            result = torch.where(above_domain, torch.ones_like(x), result)
        else:
            # For general splines, extend linearly based on boundary slopes
            if self.train_codomain and self.codomain_params is not None:
                # Apply trainable codomain scaling
                cc = self.codomain_params.cc
                cr = self.codomain_params.cr
                # The coefficients are in the original scale, map to trainable codomain
                coeff_min = coeffs.min()
                coeff_max = coeffs.max()
                # Normalize and rescale
                normalized = (interpolated - coeff_min) / (coeff_max - coeff_min + 1e-8)
                result = cc - cr + 2 * cr * normalized
                
                # Handle out-of-domain with appropriate linear extension
                left_slope = (coeffs[1] - coeffs[0]) / (self.knots[1] - self.knots[0] + 1e-8)
                right_slope = (coeffs[-1] - coeffs[-2]) / (self.knots[-1] - self.knots[-2] + 1e-8)
                
                # Scale slopes according to codomain transformation
                scale_factor = 2 * cr / (coeff_max - coeff_min + 1e-8)
                left_slope_scaled = left_slope * scale_factor
                right_slope_scaled = right_slope * scale_factor
                
                left_value = cc - cr + 2 * cr * (coeffs[0] - coeff_min) / (coeff_max - coeff_min + 1e-8)
                right_value = cc - cr + 2 * cr * (coeffs[-1] - coeff_min) / (coeff_max - coeff_min + 1e-8)
                
                result = torch.where(below_domain,
                                   left_value + left_slope_scaled * (x - self.in_min),
                                   result)
                result = torch.where(above_domain,
                                   right_value + right_slope_scaled * (x - self.in_max),
                                   result)
            else:
                # Fixed codomain case - extend linearly
                left_slope = (coeffs[1] - coeffs[0]) / (self.knots[1] - self.knots[0] + 1e-8)
                right_slope = (coeffs[-1] - coeffs[-2]) / (self.knots[-1] - self.knots[-2] + 1e-8)
                
                result = torch.where(below_domain,
                                   coeffs[0] + left_slope * (x - self.in_min),
                                   interpolated)
                result = torch.where(above_domain,
                                   coeffs[-1] + right_slope * (x - self.in_max),
                                   result)
        
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


class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ with dynamically computed domain and fixed [0,1] codomain
      - A general spline Φ with dynamically computed domain and optionally trainable codomain
      - A trainable shift (η)
      - A trainable weight VECTOR (λ) - one weight per input dimension
    If is_final is True, the block sums its outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=100, Phi_knots=100):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final
        
        # Theoretical range tracking
        self.input_range = TheoreticalRange(0.0, 1.0) if layer_num == 0 else None
        self.output_range = None
        
        # Track whether codomain has been initialized from domain
        self.codomain_initialized_from_domain = False
        
        # Create codomain parameters for Φ if trainable
        if CONFIG['train_phi_codomain']:
            self.phi_codomain_params = PhiCodomainParams(cc=0.0, cr=10.0)
        else:
            self.phi_codomain_params = None
        
        # The inner monotonic spline φ (with fixed codomain [0,1])
        # Domain will be updated dynamically based on inputs and η
        self.phi = SimpleSpline(
            num_knots=phi_knots,
            in_range=(0, 1),  # Initial domain, will be updated
            out_range=(0, 1),  # Fixed codomain
            monotonic=True
        )
        
        # The outer general spline Φ
        # Both domain and (optionally) codomain are dynamic
        self.Phi = SimpleSpline(
            num_knots=Phi_knots,
            in_range=(0, 1),  # Temporary - will be updated immediately
            out_range=(0, 1),  # Temporary - will be updated immediately
            train_codomain=CONFIG['train_phi_codomain'],
            codomain_params=self.phi_codomain_params
        )
        
        # Weight VECTOR (not matrix!) - TRUE SPRECHER IMPLEMENTATION
        # Only d_in weights, shared across all d_out outputs
        self.lambdas = nn.Parameter(torch.randn(d_in) * np.sqrt(2.0 / d_in))
        
        # Initialize eta to a reasonable value based on d_out
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10)))
        
        # Q-values for indexing
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))
        
        # Residual connection weight (learnable) - only create if enabled in config
        if CONFIG['use_residual_weights']:
            if d_in == d_out:
                # When dimensions match, use a simple scalar weight
                self.residual_weight = nn.Parameter(torch.tensor(0.1))
                self.residual_projection = None
            else:
                # When dimensions don't match, use a projection matrix
                self.residual_weight = None
                self.residual_projection = nn.Linear(d_in, d_out, bias=False)
                # Initialize projection to small values for stability
                with torch.no_grad():
                    self.residual_projection.weight.data *= 0.1
        else:
            self.residual_weight = None
            self.residual_projection = None
    
    def update_phi_domain_theoretical(self, input_range):
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
        
        self.phi.update_domain((phi_min, phi_max))
    
    def update_Phi_domain_theoretical(self):
        """Update Φ domain based on theoretical bounds."""
        # φ outputs in [0,1], weighted by lambdas, plus q term
        with torch.no_grad():
            lambda_sum_pos = torch.sum(torch.clamp(self.lambdas, min=0)).item()
            lambda_sum_neg = torch.sum(torch.clamp(self.lambdas, max=0)).item()
            
            # φ outputs in [0,1], so weighted sum is in [lambda_sum_neg, lambda_sum_pos]
            # Plus q term adds [0, Q_VALUES_FACTOR*(d_out-1)]
            phi_domain_min = lambda_sum_neg
            phi_domain_max = lambda_sum_pos + Q_VALUES_FACTOR * (self.d_out - 1)

            # Ensure domain is valid
            if phi_domain_min >= phi_domain_max:
                phi_domain_max = phi_domain_min + 1e-4
            
            self.Phi.update_domain((phi_domain_min, phi_domain_max))
            
            # Initialize codomain to match domain (one time only)
            if CONFIG['train_phi_codomain'] and not self.codomain_initialized_from_domain:
                self.initialize_codomain_from_domain(phi_domain_min, phi_domain_max)
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
                print(f"Layer {self.layer_num}: Initialized Φ codomain to match domain [{domain_min:.3f}, {domain_max:.3f}]")
                print(f"  cc = {domain_center:.3f}, cr = {domain_radius:.3f}")
    
    def compute_output_range_theoretical(self):
        """Compute theoretical output range of this block."""
        # Get the actual output range of Φ
        phi_range = self.Phi.get_actual_output_range()
        
        # If residual connections are used, we need to account for them
        if CONFIG['use_residual_weights'] and self.input_range is not None:
            if self.residual_weight is not None:
                # Scalar residual weight
                residual_contribution_min = self.residual_weight.item() * self.input_range.min
                residual_contribution_max = self.residual_weight.item() * self.input_range.max
                
                # Account for both positive and negative weights
                if self.residual_weight.item() >= 0:
                    final_min = phi_range.min + residual_contribution_min
                    final_max = phi_range.max + residual_contribution_max
                else:
                    final_min = phi_range.min + residual_contribution_max
                    final_max = phi_range.max + residual_contribution_min
            elif self.residual_projection is not None:
                # For projection, we need to be more conservative
                # Assume worst-case amplification by projection matrix
                weight_norm = torch.norm(self.residual_projection.weight, p=2).item()
                residual_bound = weight_norm * max(abs(self.input_range.min), abs(self.input_range.max))
                final_min = phi_range.min - residual_bound
                final_max = phi_range.max + residual_bound
            else:
                final_min = phi_range.min
                final_max = phi_range.max
        else:
            final_min = phi_range.min
            final_max = phi_range.max
        
        self.output_range = TheoreticalRange(final_min, final_max)
        return self.output_range
    
    def forward(self, x, x_original=None):
        # Regular forward pass (no domain updates here anymore!)
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # shape: (1, 1, d_out)
        
        # Apply translation by η * q (part of Sprecher's construction)
        shifted = x_expanded + self.eta * q
        
        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)
        
        # Weight by λ (now a VECTOR) and sum over input dimension
        weighted = phi_out * self.lambdas.view(1, -1, 1)  # Broadcast: (1, d_in, 1)
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * self.q_values  # shape: (batch_size, d_out)
        
        # Pass through the general outer spline Φ
        activated = self.Phi(s)
        
        # Add residual connection if enabled in config
        if CONFIG['use_residual_weights'] and x_original is not None:
            if self.residual_weight is not None:
                # Dimensions match, use scalar weight
                activated = activated + self.residual_weight * x_original
            elif self.residual_projection is not None:
                # Dimensions don't match, use projection
                activated = activated + self.residual_projection(x_original)
        
        if self.is_final:
            # Sum outputs if this is the final block (produces scalar output)
            return activated.sum(dim=1, keepdim=True)
        else:
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
    """
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=100, 
                 Phi_knots=100, batchnorm=False, norm_type=None, target_stats=None):
        """
        Initialize Sprecher Network with optional normalization.
        
        Args:
            input_dim: Input dimension
            architecture: List of hidden layer sizes
            final_dim: Output dimension
            phi_knots: Number of knots for φ splines
            Phi_knots: Number of knots for Φ splines
            batchnorm: Legacy parameter for Giovanni's batch normalization
            norm_type: Normalization type ('none', 'exact_layer', 'giovanni_batch', 'affine')
            target_stats: Optional dict with 'mean' and 'std' for alternative initialization
        """
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        self.batchnorm = batchnorm
        self.norm_type = norm_type if norm_type is not None else ('giovanni_batch' if batchnorm else 'none')

        # Create layers
        layers = []
        if not architecture: # No hidden layers
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(
                d_in=input_dim, d_out=final_dim,
                layer_num=0, is_final=is_final,
                phi_knots=phi_knots, Phi_knots=Phi_knots
            ))
        else:
            L = len(architecture)
            # Create hidden layers
            d_in = input_dim
            for i in range(L):
                d_out = architecture[i]
                # The last hidden block is summed if the final output is scalar
                is_final_block = (i == L - 1) and (self.final_dim == 1)
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=d_out,
                    layer_num=i, is_final=is_final_block,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
                d_in = d_out
            
            # Add a final output block if output is vector-valued
            if self.final_dim > 1:
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=self.final_dim,
                    layer_num=L, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
        
        self.layers = nn.ModuleList(layers)

        # Create normalization layers based on type
        if self.norm_type == 'giovanni_batch':
            # Giovanni's original implementation
            self.batch_layers = nn.ModuleList([
                torch.nn.BatchNorm1d(i) for i in architecture[:-1]
            ])
            self.norm_layers = None
        elif self.norm_type == 'exact_layer':
            # Exact LayerNorm implementation
            self.batch_layers = None
            self.norm_layers = nn.ModuleList()
            
            # Create LayerNorm for each layer output (except final)
            for i, layer in enumerate(self.layers[:-1]):  # Skip last layer
                self.norm_layers.append(
                    ExactLayerNorm(layer.d_out, eps=1e-5, identity_init=True)
                )
        elif self.norm_type == 'affine':
            # Affine normalization with exact bounds
            self.batch_layers = None
            self.norm_layers = nn.ModuleList()
            
            # Create AffineNormalization for each layer output (except final)
            for i, layer in enumerate(self.layers[:-1]):  # Skip last layer
                self.norm_layers.append(
                    AffineNormalization(layer.d_out, eps=1e-8, identity_init=True)
                )
        else:
            # No normalization
            self.batch_layers = None
            self.norm_layers = None

        # Alternative initialization for final normalization layer based on target stats
        # This is commented out but can be enabled by uncommenting the code below
        '''
        if target_stats is not None and self.norm_layers is not None and len(self.norm_layers) > 0:
            # Initialize the last normalization layer to map to target distribution
            with torch.no_grad():
                last_norm = self.norm_layers[-1]
                target_mean = target_stats.get('mean', 0.0)
                target_std = target_stats.get('std', 1.0)
                
                if isinstance(last_norm, ExactLayerNorm):
                    # For LayerNorm: output ≈ gamma * normalized + beta
                    # We want output to have mean=target_mean, std=target_std
                    # Since normalized has mean≈0, std≈1, we set:
                    last_norm.gamma.data.fill_(target_std)
                    last_norm.beta.data.fill_(target_mean)
                elif isinstance(last_norm, AffineNormalization):
                    # For Affine: output = weight * input + bias
                    # Assuming normalized input has some distribution, we can approximate:
                    # (This is a rough approximation - could be improved with actual input stats)
                    last_norm.weight.data.fill_(target_std)
                    last_norm.bias.data.fill_(target_mean)
        '''
        
        # Initialize domains based on theoretical bounds
        if CONFIG.get('use_theoretical_domains', True):
            self.update_all_domains()
    
    def update_all_domains(self):
        """Update all spline domains based on theoretical bounds."""
        current_range = TheoreticalRange(0.0, 1.0)  # Input is in [0,1]
        
        for i, layer in enumerate(self.layers):
            # For Giovanni's batch norm (pre-normalization)
            if self.norm_type == 'giovanni_batch' and self.batch_layers is not None:
                if i > 0 and i - 1 < len(self.batch_layers):
                    # Cannot compute exact bounds for BatchNorm
                    # Use conservative estimate
                    current_range = TheoreticalRange(-3.0, 3.0)
            
            # Update φ domain based on input range
            layer.update_phi_domain_theoretical(current_range)
            
            # Update Φ domain based on current lambdas
            layer.update_Phi_domain_theoretical()
            
            # Compute this layer's output range for next layer
            layer_output_range = layer.compute_output_range_theoretical()
            
            # Apply normalization effect if using exact LayerNorm or Affine
            if self.norm_type in ['exact_layer', 'affine'] and self.norm_layers is not None:
                if i < len(self.norm_layers):
                    # Update expected input range for identity init
                    self.norm_layers[i].update_expected_input_range(
                        layer_output_range.min, layer_output_range.max
                    )
                    # Get exact bounds after normalization
                    norm_min, norm_max = self.norm_layers[i].get_exact_output_bounds(
                        layer_output_range.min, layer_output_range.max
                    )
                    current_range = TheoreticalRange(norm_min, norm_max)
                else:
                    current_range = layer_output_range
            else:
                current_range = layer_output_range
            
            # Debug output if configured
            if CONFIG.get('debug_domains', False):
                print(f"Layer {layer.layer_num}: input_range={layer.input_range}, output_range={layer.output_range}")
                print(f"  φ domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
                print(f"  Φ domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
                if self.norm_type in ['exact_layer', 'affine'] and i < len(self.norm_layers):
                    print(f"  After normalization: {current_range}")
    
    def forward(self, x):
        x_in = x
        
        for i, layer in enumerate(self.layers):
            # Giovanni's pre-block normalization
            if self.norm_type == 'giovanni_batch' and self.batch_layers is not None:
                if i > 0 and i - 1 < len(self.batch_layers):
                    x_in = self.batch_layers[i - 1](x_in)
            
            # For residual connections, pass the input of the *current* layer
            x_out = layer(x_in, x_in)
            
            # Exact LayerNorm or Affine post-block normalization (skip last layer)
            if self.norm_type in ['exact_layer', 'affine'] and self.norm_layers is not None:
                if i < len(self.norm_layers):
                    x_out = self.norm_layers[i](x_out)
            
            x_in = x_out
        
        # Final output after all layers (no more scaling/bias)
        return x_in
    
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
            print(f"  φ violations: {layer_stats['phi']:.2%}")
            print(f"  Φ violations: {layer_stats['Phi']:.2%}")