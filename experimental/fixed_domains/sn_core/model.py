"""Sprecher Network model components: splines, layers, and networks."""

import torch
import torch.nn as nn
import numpy as np
from .config import CONFIG, PHI_RANGE, Q_VALUES_FACTOR


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
        
        # Initialize spline coefficients
        torch.manual_seed(CONFIG['seed'])
        if monotonic:
            # For monotonic splines, use log-space parameters to ensure monotonicity
            self.log_increments = nn.Parameter(torch.zeros(num_knots))
            with torch.no_grad():
                # Initialize to approximate linear function
                self.log_increments.data = torch.log(torch.ones(num_knots) / num_knots + 1e-6)
        else:
            # For general splines (Φ), initialize closer to identity/linear
            if self.train_codomain and self.codomain_params is not None:
                cc = self.codomain_params.cc.item()
                cr = self.codomain_params.cr.item()
                out_min = cc - cr
                out_max = cc + cr
            else:
                out_min, out_max = out_range
            # Initialize to approximate linear function with small random perturbations
            linear_coeffs = torch.linspace(out_min, out_max, num_knots)
            self.coeffs = nn.Parameter(linear_coeffs + torch.randn(num_knots) * 0.01)
    
    def update_domain(self, new_range):
        """Update the domain of the spline by adjusting knot positions."""
        with torch.no_grad():
            self.in_min, self.in_max = new_range
            self.knots.data = torch.linspace(self.in_min, self.in_max, self.num_knots, 
                                           device=self.knots.device, dtype=self.knots.dtype)
    
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
    
    def forward(self, x):
        x = x.to(self.knots.device)
        
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
                left_slope = (coeffs[1] - coeffs[0]) / (self.knots[1] - self.knots[0])
                right_slope = (coeffs[-1] - coeffs[-2]) / (self.knots[-1] - self.knots[-2])
                
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
                left_slope = (coeffs[1] - coeffs[0]) / (self.knots[1] - self.knots[0])
                right_slope = (coeffs[-1] - coeffs[-2]) / (self.knots[-1] - self.knots[-2])
                
                result = torch.where(below_domain,
                                   coeffs[0] + left_slope * (x - self.in_min),
                                   interpolated)
                result = torch.where(above_domain,
                                   coeffs[-1] + right_slope * (x - self.in_max),
                                   result)
        
        return result


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
            in_range=PHI_RANGE,  # Initial domain, will be updated
            out_range=PHI_RANGE,  # Initial codomain, may be trainable
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
    
    def forward(self, x, x_original=None):
        # Update φ domain based on current η and input range
        with torch.no_grad():
            if self.layer_num == 0:
                # First layer: inputs assumed to be in [0,1]
                # Domain needs to handle x + η*q for q in [0, d_out-1]
                phi_max = 1.0 + self.eta.item() * (self.d_out - 1)
                self.phi.update_domain((0, phi_max))
            else:
                # Deeper layers: based on actual input range
                x_min = x.min().item()
                x_max = x.max().item()
                # Domain needs to handle [x_min, x_max] + η*q
                phi_min = x_min
                phi_max = x_max + self.eta.item() * (self.d_out - 1)
                self.phi.update_domain((phi_min, phi_max))
        
        # Regular forward pass
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # shape: (1, 1, d_out)
        
        # Apply translation by η * q (part of Sprecher's construction)
        shifted = x_expanded + self.eta * q
        
        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)
        
        # Weight by λ (now a VECTOR) and sum over input dimension
        weighted = phi_out * self.lambdas.view(1, -1, 1)  # Broadcast: (1, d_in, 1)
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * self.q_values  # shape: (batch_size, d_out)
        
        # Update Φ domain based on current λ values
        with torch.no_grad():
            # Compute the range of possible inputs to Φ
            lambda_sum_pos = torch.sum(torch.clamp(self.lambdas, min=0)).item()
            lambda_sum_neg = torch.sum(torch.clamp(self.lambdas, max=0)).item()
            # φ outputs in [0,1], so weighted sum is in [lambda_sum_neg, lambda_sum_pos]
            # Plus q term adds [0, Q_VALUES_FACTOR*(d_out-1)]
            phi_domain_min = lambda_sum_neg
            phi_domain_max = lambda_sum_pos + Q_VALUES_FACTOR * (self.d_out - 1)
            self.Phi.update_domain((phi_domain_min, phi_domain_max))
        
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
        if CONFIG['train_phi_codomain'] and self.phi_codomain_params is not None:
            cc = self.phi_codomain_params.cc.item()
            cr = self.phi_codomain_params.cr.item()
            return (cc - cr, cc + cr)
        else:
            # Use the fixed range
            return (-10.0, 10.0)  # Or whatever fixed range is appropriate


class SprecherMultiLayerNetwork(nn.Module):
    """
    Builds the Sprecher network with a given hidden-layer architecture and final output dimension.
    """
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=100, Phi_knots=100):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        
        layers = []
        if len(architecture) == 0:
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(
                d_in=input_dim, d_out=final_dim,
                layer_num=0, is_final=is_final,
                phi_knots=phi_knots, Phi_knots=Phi_knots
            ))
        else:
            L = len(architecture)
            for i in range(L - 1):
                d_in = input_dim if i == 0 else architecture[i - 1]
                d_out = architecture[i]
                layers.append(SprecherLayerBlock(
                    d_in=d_in, d_out=d_out,
                    layer_num=i, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
            d_in_last = input_dim if L == 1 else architecture[L - 2]
            d_out_last = architecture[L - 1]
            if self.final_dim == 1:
                layers.append(SprecherLayerBlock(
                    d_in=d_in_last, d_out=d_out_last,
                    layer_num=L-1, is_final=True,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
            else:
                layers.append(SprecherLayerBlock(
                    d_in=d_in_last, d_out=d_out_last,
                    layer_num=L-1, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
                layers.append(SprecherLayerBlock(
                    d_in=d_out_last, d_out=self.final_dim,
                    layer_num=L, is_final=False,
                    phi_knots=phi_knots, Phi_knots=Phi_knots
                ))
        
        self.layers = nn.ModuleList(layers)
        
        # Output scaling parameters for better initialization
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        x_original = x
        for i, layer in enumerate(self.layers):
            # Pass original input for residual connections
            if CONFIG['use_residual_weights']:
                x = layer(x, x_original)
                x_original = x  # Update x_original for next layer
            else:
                x = layer(x, None)
        # Apply output scaling and bias
        x = self.output_scale * x + self.output_bias
        return x