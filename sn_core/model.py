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
    
    def update_domain(self, new_range, allow_resampling=True, force_resample=False):
        """
        Update the domain of the spline by adjusting knot positions.
        For non-monotonic splines (Φ), this method resamples the spline to preserve
        its learned functional shape in the new domain, preventing information loss
        during training.
        
        Args:
            new_range: Tuple of (new_min, new_max) for the domain
            allow_resampling: If False, skip resampling even if conditions are met
            force_resample: If True, force resampling regardless of total_evaluations
        """
        with torch.no_grad():
            new_min, new_max = new_range
            
            # Debug logging for checkpoint operations
            if CONFIG.get('debug_checkpoint_loading', False):
                print(f"\n[CHECKPOINT DEBUG] SimpleSpline.update_domain called:")
                print(f"  - Monotonic: {self.monotonic}")
                print(f"  - Old domain: [{self.in_min:.6f}, {self.in_max:.6f}]")
                print(f"  - New domain: [{new_min:.6f}, {new_max:.6f}]")
                print(f"  - Allow resampling: {allow_resampling}")
                print(f"  - Total evaluations: {self.total_evaluations}")

            # Guard clause: If the domain isn't changing significantly, do nothing.
            # This prevents floating point drift and unnecessary computation.
            if abs(self.in_min - new_min) < 1e-6 and abs(self.in_max - new_max) < 1e-6:
                if CONFIG.get('debug_checkpoint_loading', False):
                    print("  - Domain unchanged, skipping update")
                return

            # --- Spline Resampling to Preserve Learned Shape (for Φ splines) ---
            # This is the core fix for training instability.
            # It's applied only to the general (non-monotonic) splines after the
            # first training step has occurred.
            if not self.monotonic and (self.total_evaluations > 0 or force_resample) and allow_resampling:
                if CONFIG.get('debug_checkpoint_loading', False):
                    print(f"  - RESAMPLING WILL OCCUR (non-monotonic, evaluations={self.total_evaluations}, force={force_resample}, resampling allowed)")
                    
                # Store the state of the old spline before changing anything
                old_knots = self.knots.clone()
                old_coeffs = self.get_coeffs().clone()
                
                if CONFIG.get('debug_checkpoint_loading', False):
                    print(f"  - Old coeffs min/max: {old_coeffs.min():.6f}/{old_coeffs.max():.6f}")
                
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
                
                if CONFIG.get('debug_checkpoint_loading', False):
                    print(f"  - New coeffs min/max after resampling: {resampled_coeffs.min():.6f}/{resampled_coeffs.max():.6f}")
                    print(f"  - Coeffs changed: {not torch.allclose(old_coeffs, resampled_coeffs)}")
            else:
                if CONFIG.get('debug_checkpoint_loading', False) and not self.monotonic:
                    print(f"  - NO RESAMPLING (evaluations={self.total_evaluations}, allow={allow_resampling})")

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
        
        if CONFIG.get('debug_checkpoint_loading', False):
            print(f"\n[CHECKPOINT DEBUG] SimpleSpline.get_domain_state:")
            print(f"  - Monotonic: {self.monotonic}")
            print(f"  - Domain: [{state['in_min']:.6f}, {state['in_max']:.6f}]")
            print(f"  - Knots shape: {state['knots'].shape}")
            print(f"  - Coeffs min/max: {state['coeffs'].min():.6f}/{state['coeffs'].max():.6f}")
            print(f"  - Total evaluations: {state['total_evaluations']}")
            if 'raw_coeffs' in state:
                print(f"  - Raw coeffs min/max: {state['raw_coeffs'].min():.6f}/{state['raw_coeffs'].max():.6f}")
            if 'log_increments' in state:
                print(f"  - Log increments min/max: {state['log_increments'].min():.6f}/{state['log_increments'].max():.6f}")
            
        return state
    
    def set_domain_state(self, state):
        """Restore domain state from checkpoint."""
        if CONFIG.get('debug_checkpoint_loading', False):
            print(f"\n[CHECKPOINT DEBUG] SimpleSpline.set_domain_state:")
            print(f"  - Monotonic: {self.monotonic}")
            print(f"  - Old domain: [{self.in_min:.6f}, {self.in_max:.6f}]")
            print(f"  - New domain: [{state['in_min']:.6f}, {state['in_max']:.6f}]")
            print(f"  - New knots shape: {state['knots'].shape}")
            print(f"  - New total evaluations: {state.get('total_evaluations', 0)}")
            
        self.knots.data = state['knots']
        self.in_min = state['in_min']
        self.in_max = state['in_max']
        
        # Restore raw parameters to ensure exact restoration
        if self.monotonic and 'log_increments' in state:
            # For monotonic splines, restore log_increments
            self.log_increments.data = state['log_increments']
            if CONFIG.get('debug_checkpoint_loading', False):
                print(f"  - Log increments restored: min={state['log_increments'].min():.6f}, max={state['log_increments'].max():.6f}")
        elif not self.monotonic and 'raw_coeffs' in state:
            # For non-monotonic splines, restore raw coefficients
            self.coeffs.data = state['raw_coeffs']
            if CONFIG.get('debug_checkpoint_loading', False):
                print(f"  - Raw coeffs restored: min={state['raw_coeffs'].min():.6f}, max={state['raw_coeffs'].max():.6f}")
        
        self.domain_violations = state.get('domain_violations', 0)
        self.total_evaluations = state.get('total_evaluations', 0)
        
        if CONFIG.get('debug_checkpoint_loading', False):
            print(f"  - Domain restored successfully")
            print(f"  - Knots updated: {torch.allclose(self.knots, state['knots'])}")


class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ with dynamically computed domain and fixed [0,1] codomain
      - A general spline Φ with dynamically computed domain and optionally trainable codomain
      - A trainable shift (η)
      - A trainable weight VECTOR (λ) - one weight per input dimension
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
    
    def update_Phi_domain_theoretical(self, allow_resampling=True, force_resample=False):
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
            
            self.Phi.update_domain((phi_domain_min, phi_domain_max), allow_resampling=allow_resampling, force_resample=force_resample)
            
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
                print(f"Layer {self.layer_num}: Initialized Phi codomain to match domain [{domain_min:.3f}, {domain_max:.3f}]")
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
        # Regular forward pass (no domain updates here anymore!)

        # --- FIX STARTS HERE ---
        # Ensure the q_values buffer is on the same device as the input tensor.
        # This is a robust fix for device mismatches that can occur after `copy.deepcopy`.
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
        # --- FIX ENDS HERE ---
        
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
        Update all spline domains based on theoretical bounds.
        
        Args:
            allow_resampling: If False, prevent spline coefficient resampling during domain updates
            force_resample: If True, force resampling regardless of total_evaluations
        """
        current_range = TheoreticalRange(0.0, 1.0)  # Input is in [0,1]
        
        for layer in self.layers:
            # Update φ domain based on input range
            layer.update_phi_domain_theoretical(current_range, allow_resampling=allow_resampling, force_resample=force_resample)
            
            # Update Φ domain based on current lambdas
            layer.update_Phi_domain_theoretical(allow_resampling=allow_resampling, force_resample=force_resample)
            
            # Compute this layer's output range for next layer
            current_range = layer.compute_output_range_theoretical()
            
            # Debug output if configured
            if CONFIG.get('debug_domains', False):
                print(f"Layer {layer.layer_num}: input_range={layer.input_range}, output_range={layer.output_range}")
                print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
                print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
    
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