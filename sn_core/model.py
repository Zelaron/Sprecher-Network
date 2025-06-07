"""Sprecher Network model components: splines, layers, and networks."""

import torch
import torch.nn as nn
import numpy as np

# Global configuration parameters
PHI_IN_RANGE = (-10.0, 10.0)
PHI_OUT_RANGE = (-10.0, 10.0)
TRAIN_PHI_RANGE = True
Q_VALUES_FACTOR = 1.0
USE_RESIDUAL_WEIGHTS = True  # Set to False to disable residual connections for testing
SEED = 45


class PhiRangeParams(nn.Module):
    """
    Global, trainable range parameters for all Φ splines.
    The domain and codomain are defined via a center and a radius:
      domain: (dc - dr, dc + dr)
      codomain: (cc - cr, cc + cr)
    """
    def __init__(self, dc=0.0, dr=1.0, cc=0.0, cr=1.0):
        super().__init__()
        self.dc = nn.Parameter(torch.tensor(dc, dtype=torch.float32))  # dc: domain center for Φ
        self.dr = nn.Parameter(torch.tensor(dr, dtype=torch.float32))  # dr: domain radius for Φ
        self.cc = nn.Parameter(torch.tensor(cc, dtype=torch.float32))  # cc: codomain center for Φ
        self.cr = nn.Parameter(torch.tensor(cr, dtype=torch.float32))  # cr: codomain radius for Φ


class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline.
    
    If train_range is False, the domains and codomains of the splines Φ are not trained.
    If train_range is True and a valid range_params (of type PhiRangeParams)
    is provided, then:
      - The input range (domain) is computed on the fly from normalized knots
        and the current dc and dr.
      - After linear interpolation, the raw output is normalized using the
        initial (stored) min and max and then re-scaled to the current output range.
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False,
                 train_range=False, range_params=None):
        super().__init__()
        self.num_knots = num_knots
        self.monotonic = monotonic
        self.train_range = train_range
        self.fixed_in_range = in_range
        self.fixed_out_range = out_range
        self.range_params = range_params

        if self.train_range and (self.range_params is not None):
            # Create normalized knots in [0,1]; these will be scaled at runtime.
            self.register_buffer('normalized_knots', torch.linspace(0, 1, num_knots))
        else:
            self.in_min, self.in_max = in_range
            self.out_min, self.out_max = out_range
            self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        # Initialize spline coefficients
        torch.manual_seed(SEED)
        if monotonic:
            # For monotonic splines, use log-space parameters to ensure monotonicity
            # This avoids the need for sorting and improves gradient flow
            self.log_increments = nn.Parameter(torch.zeros(num_knots))
            with torch.no_grad():
                # Initialize to approximate linear function
                self.log_increments.data = torch.log(torch.ones(num_knots) / num_knots + 1e-6)
        else:
            # For general splines (Φ), initialize closer to identity/linear
            if self.train_range and (self.range_params is not None):
                cc = self.range_params.cc.item()
                cr = self.range_params.cr.item()
                out_min = cc - cr
                out_max = cc + cr
            else:
                out_min, out_max = out_range
            # Initialize to approximate linear function with small random perturbations
            linear_coeffs = torch.linspace(out_min, out_max, num_knots)
            self.coeffs = nn.Parameter(linear_coeffs + torch.randn(num_knots) * 0.01)
    
    def get_coeffs(self):
        """Get the actual spline coefficients (for both monotonic and non-monotonic)"""
        if self.monotonic:
            # Use softplus to ensure positive increments, then cumsum for monotonicity
            increments = torch.nn.functional.softplus(self.log_increments)
            cumulative = torch.cumsum(increments, 0)
            # Normalize to [0, 1] range
            cumulative = cumulative / (cumulative[-1] + 1e-8)
            # Scale to output range
            if self.train_range and (self.range_params is not None):
                cc = self.range_params.cc
                cr = self.range_params.cr
                out_min = cc - cr
                out_max = cc + cr
            else:
                out_min, out_max = self.fixed_out_range
            return out_min + (out_max - out_min) * cumulative
        else:
            return self.coeffs
    
    def forward(self, x):
        if self.train_range and (self.range_params is not None):
            # Compute dynamic in_range from current trainable parameters.
            dc = self.range_params.dc
            dr = self.range_params.dr
            in_min = dc - dr
            in_max = dc + dr
            knots = self.normalized_knots * (in_max - in_min) + in_min
            # Also compute dynamic out_range.
            cc = self.range_params.cc
            cr = self.range_params.cr
            out_min = cc - cr
            out_max = cc + cr
        else:
            knots = self.knots
            in_min = self.in_min
            in_max = self.in_max
            out_min = self.out_min
            out_max = self.out_max
        
        x = x.to(knots.device)
        x = torch.clamp(x, in_min, in_max)
        
        # Find which segment each input value belongs to
        intervals = torch.searchsorted(knots, x) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)
        
        # Compute interpolation parameter t in [0,1]
        t = (x - knots[intervals]) / (knots[intervals + 1] - knots[intervals])
        
        # Get coefficients
        coeffs = self.get_coeffs()
        
        # Linear interpolation
        raw = (1 - t) * coeffs[intervals] + t * coeffs[intervals + 1]
        
        # If train_range is enabled, re-scale the raw output.
        eps = 1e-8
        if self.train_range and (self.range_params is not None) and not self.monotonic:
            if not hasattr(self, 'init_coeff_min') or not hasattr(self, 'init_coeff_max'):
                # Store initial min and max of coeffs.
                self.register_buffer('init_coeff_min', self.coeffs.data.min().clone())
                self.register_buffer('init_coeff_max', self.coeffs.data.max().clone())
            coeff_min = self.init_coeff_min
            coeff_max = self.init_coeff_max
            # Add small epsilon to avoid division by zero
            raw_normalized = (raw - coeff_min) / (coeff_max - coeff_min + eps)
            raw_normalized = torch.clamp(raw_normalized, 0, 1)  # Ensure normalized values are in [0, 1]
            final = out_min + (out_max - out_min) * raw_normalized
        else:
            final = raw
        return final


class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ (with fixed (0,1) domain and range)
      - A general spline Φ (with trainable domain and codomain)
      - A trainable shift (η)
      - A trainable weight VECTOR (λ) - one weight per input dimension
    If is_final is True, the block sums its outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, is_final=False, phi_range_params=None, phi_knots=100, Phi_knots=100):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        # The inner monotonic spline φ (with fixed range (0,1))
        # This represents the 'inner function' in Sprecher's theorem
        self.phi = SimpleSpline(num_knots=phi_knots, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        
        # The outer general spline Φ with trainable range parameters
        # This represents the 'outer function' in Sprecher's theorem
        self.Phi = SimpleSpline(num_knots=Phi_knots, in_range=PHI_IN_RANGE, out_range=PHI_OUT_RANGE,
                                train_range=TRAIN_PHI_RANGE, range_params=phi_range_params)
        
        # Weight VECTOR (not matrix!) - TRUE SPRECHER IMPLEMENTATION
        # Only d_in weights, shared across all d_out outputs
        self.lambdas = nn.Parameter(torch.randn(d_in) * np.sqrt(2.0 / d_in))
        
        # Initialize eta to a reasonable value based on d_out
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10)))
        
        # Q-values for indexing.
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))
        
        # Residual connection weight (learnable) - only create if USE_RESIDUAL_WEIGHTS is True
        if USE_RESIDUAL_WEIGHTS:
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
        # x: (batch_size, d_in)
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # shape: (1, 1, d_out)
        
        # Apply translation by η * q (part of Sprecher's construction)
        shifted = x_expanded + self.eta * q
        shifted = torch.clamp(shifted, 0, 1)
        
        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)
        
        # Weight by λ (now a VECTOR) and sum over input dimension
        # lambdas has shape (d_in), we need to broadcast it to (batch_size, d_in, d_out)
        weighted = phi_out * self.lambdas.view(1, -1, 1)  # Broadcast: (1, d_in, 1)
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * self.q_values  # shape: (batch_size, d_out)
        
        # Pass through the general outer spline Φ
        activated = self.Phi(s)
        
        # Add residual connection if USE_RESIDUAL_WEIGHTS is True
        if USE_RESIDUAL_WEIGHTS and x_original is not None:
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


class SprecherMultiLayerNetwork(nn.Module):
    """
    Builds the Sprecher network with a given hidden-layer architecture and final output dimension.
    A single global PhiRangeParams instance is created and shared among all Φ splines.
    """
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=100, Phi_knots=100):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        
        # Compute domain and codomain centers and radii (for Φ) from the configured outer ranges.
        dc = (PHI_IN_RANGE[0] + PHI_IN_RANGE[1]) / 2.0
        dr = (PHI_IN_RANGE[1] - PHI_IN_RANGE[0]) / 2.0
        cc = (PHI_OUT_RANGE[0] + PHI_OUT_RANGE[1]) / 2.0
        cr = (PHI_OUT_RANGE[1] - PHI_OUT_RANGE[0]) / 2.0

        # Create global trainable parameters for all Φ splines
        self.phi_range_params = PhiRangeParams(dc=dc, dr=dr, cc=cc, cr=cr)
        
        layers = []
        if len(architecture) == 0:
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=final_dim,
                                             is_final=is_final, phi_range_params=self.phi_range_params,
                                             phi_knots=phi_knots, Phi_knots=Phi_knots))
        else:
            L = len(architecture)
            for i in range(L - 1):
                d_in = input_dim if i == 0 else architecture[i - 1]
                d_out = architecture[i]
                layers.append(SprecherLayerBlock(d_in=d_in, d_out=d_out,
                                                 is_final=False, phi_range_params=self.phi_range_params,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
            d_in_last = input_dim if L == 1 else architecture[L - 2]
            d_out_last = architecture[L - 1]
            if self.final_dim == 1:
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last,
                                                 is_final=True, phi_range_params=self.phi_range_params,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
            else:
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last,
                                                 is_final=False, phi_range_params=self.phi_range_params,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
                layers.append(SprecherLayerBlock(d_in=d_out_last, d_out=self.final_dim,
                                                 is_final=False, phi_range_params=self.phi_range_params,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
        
        self.layers = nn.ModuleList(layers)
        
        # Output scaling parameters for better initialization
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        x_original = x
        for i, layer in enumerate(self.layers):
            # Pass original input for residual connections
            if USE_RESIDUAL_WEIGHTS:
                x = layer(x, x_original)
                x_original = x  # Update x_original for next layer
            else:
                x = layer(x, None)
        # Apply output scaling and bias
        x = self.output_scale * x + self.output_bias
        return x