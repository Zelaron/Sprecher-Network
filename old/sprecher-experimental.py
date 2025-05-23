import os
import torch
import torch.nn as nn
import numpy as np

# --- Force a non-interactive backend if Tcl/Tk is not properly installed ---
# import matplotlib
# matplotlib.use('Agg') # Renders plots to files instead of interactive windows

import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import math
import copy
# Import the scheduler
from torch.optim.lr_scheduler import CyclicLR

##############################################################################
#                          CONFIGURABLE SETTINGS                             #
##############################################################################

# 'architecture' defines the hidden-layer sizes.
architecture = [17, 17, 17, 17] # Works decently for a function of two variables
# architecture = [10] # Good for a function of one variable

# Total training epochs
TOTAL_EPOCHS = 50000

# Whether to save the final plot
SAVE_FINAL_PLOT = True

# Image size in pixels (when saving)
FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT = 3840, 2160

# Fixed seed for reproducibility
SEED = 45

# Variance penalty weight.
LAMBDA_VAR = 1.0

# Global outer spline (𝛷) configuration.
# You can specify different ranges for the default input (domain) and output (codomain) of 𝛷, which will be trained if TRAIN_PHI_RANGE is set to true.
PHI_IN_RANGE = (-10.0, 10.0)
PHI_OUT_RANGE = (-10.0, 10.0)

# Determines whether the spline ranges for 𝛷 are trainable
TRAIN_PHI_RANGE = True

# Number of knots (control points defining spline segments) for the inner monotonic splines (φ)
PHI_KNOTS = 300

# Number of knots for the outer general splines (𝛷)
PHI_CAPITAL_KNOTS = 200

# --- Settings for Cyclic Learning Rate (CLR) ---
# Base LR for parameters that will be cycled (coeffs, eta, lambdas)
CLR_BASE_LR_BOOSTABLE = 0.0003
# Max LR for parameters that will be cycled
CLR_MAX_LR_BOOSTABLE = 0.003 # 10x base_lr - adjust as needed
# Constant LR for global range parameters (dc, dr, cc, cr)
CLR_CONST_LR_RANGE = 0.001
# Number of epochs for half a cycle (increasing phase)
CLR_STEP_SIZE_UP = 5000
# CLR mode ('triangular', 'triangular2', 'exp_range')
# 'triangular2' decreases the peak LR after each cycle
CLR_MODE = 'triangular2'
CLR_GAMMA = None # only used for exp_range


##############################################################################
#                          TARGET FUNCTION DEFINITION                        #
##############################################################################

def target_function(x):
    # 1D scalar function:
    # return (x[:, [0]] - 3/10)**5 - (x[:, [0]] - 1/3)**3 + (1/5)*(x[:, [0]] - 1/10)**2

    # 2D scalar function example:
    return torch.exp(torch.sin(11 * x[:, [0]])) + 3 * x[:, [1]] + 4 * torch.sin(8 * x[:, [1]])

    # 2D vector function example
    # Try setting e.g. architecture = [20, 20, 20, 20, 20], TOTAL_EPOCHS = 100000, PHI_KNOTS = PHI_CAPITAL_KNOTS = 500 for this example:
    # return torch.cat([
    #     (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7,
    #     (1/4)*x[:, [1]] + (1/5)*x[:, [1]]**2 - x[:, [0]]**3 + (1/5)*torch.sin(7*x[:, [0]]),
    #     torch.exp(torch.sin(11 * x[:, [0]])) + 3 * x[:, [1]] + 4 * torch.sin(8 * x[:, [1]])
    # ], dim=1)

##############################################################################
#                     UTILITY: DETERMINE INPUT DIMENSION                     #
##############################################################################

def get_input_dimension(target_function):
    """
    Tries dimensions 1 to 3 until target_function runs without error.
    Returns the dimension that works.
    """
    for dim in range(1, 4):
        test_x = torch.zeros(1, dim)
        try:
            result = target_function(test_x)
            if torch.isfinite(result).all():
                return dim
        except Exception:
            continue
    raise ValueError("Could not determine input dimension (or dimension > 3 not supported)")

##############################################################################
#                     UTILITY: DETERMINE OUTPUT DIMENSION                    #
##############################################################################

def get_output_dimension(target_function, input_dim):
    """
    Determines the output dimension of the target function.
    """
    test_x = torch.zeros(1, input_dim)
    result = target_function(test_x)

    # Handle scalar outputs (1D tensors)
    if result.dim() == 1:
        return 1

    # For 2D tensors, return the size of the second dimension
    return result.shape[1]

##############################################################################
#                             SET SEEDS FOR REPRODUCIBILITY                  #
##############################################################################

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############################################################################
#                        GLOBAL PHI RANGE PARAMETERS                         #
##############################################################################

class PhiRangeParams(nn.Module):
    """
    Global, trainable range parameters for all 𝛷 splines.
    The domain and codomain are defined via a center and a radius:
      domain: (dc - dr, dc + dr)
      codomain: (cc - cr, cc + cr)
    """
    def __init__(self, dc=0.0, dr=1.0, cc=0.0, cr=1.0):
        super().__init__()
        self.dc = nn.Parameter(torch.tensor(dc, dtype=torch.float32))  # dc: domain center for 𝛷
        self.dr = nn.Parameter(torch.tensor(dr, dtype=torch.float32))  # dr: domain radius for 𝛷
        self.cc = nn.Parameter(torch.tensor(cc, dtype=torch.float32))  # cc: codomain center for 𝛷
        self.cr = nn.Parameter(torch.tensor(cr, dtype=torch.float32))  # cr: codomain radius for 𝛷

##############################################################################
#                             SIMPLE SPLINE MODULE                           #
##############################################################################

class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline.

    If train_range is False, the domains and codomains of the splines 𝛷 are not trained.
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
        # Ensure consistent seeding for initialization across runs
        current_rng_state = torch.get_rng_state()
        torch.manual_seed(SEED + sum(ord(c) for c in self.__class__.__name__)) # Use a slightly different seed for each module type
        if monotonic:
            # For monotonic splines (φ), initialize with increasing values
            increments = torch.rand(num_knots) * 0.1
            # Ensure Parameter name is 'coeffs' for easy identification later
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                if self.train_range and (self.range_params is not None):
                    # This case shouldn't happen for monotonic splines per design, but handle defensively
                    cc = self.range_params.cc.item()
                    cr = self.range_params.cr.item()
                    out_min = cc - cr
                    out_max = cc + cr
                else:
                    out_min, out_max = out_range
                # Normalize initial coeffs to the specified output range
                min_val = self.coeffs.data.min()
                max_val = self.coeffs.data.max()
                if max_val > min_val: # Avoid division by zero if all increments were zero
                    self.coeffs.data = (self.coeffs.data - min_val) / (max_val - min_val)
                    self.coeffs.data = self.coeffs.data * (out_max - out_min) + out_min
                else: # Handle flat case
                    self.coeffs.data.fill_((out_max + out_min) / 2.0)

        else:
            # For general splines (𝛷), initialize with a balanced distribution
            if self.train_range and (self.range_params is not None):
                cc = self.range_params.cc.item()
                cr = self.range_params.cr.item()
                out_min = cc - cr
                out_max = cc + cr
            else:
                out_min, out_max = out_range
            # Ensure Parameter name is 'coeffs'
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_min, out_max, num_knots)
            )
        torch.set_rng_state(current_rng_state) # Restore RNG state

    def forward(self, x):
        if self.train_range and (self.range_params is not None):
            # Compute dynamic in_range from current trainable parameters.
            dc = self.range_params.dc
            dr = torch.abs(self.range_params.dr) # Ensure radius is non-negative
            in_min = dc - dr
            in_max = dc + dr
            knots = self.normalized_knots * (in_max - in_min) + in_min
            # Also compute dynamic out_range.
            cc = self.range_params.cc
            cr = torch.abs(self.range_params.cr) # Ensure radius is non-negative
            out_min = cc - cr
            out_max = cc + cr
        else:
            knots = self.knots
            in_min = self.in_min
            in_max = self.in_max
            out_min = self.out_min
            out_max = self.out_max

        x = x.to(knots.device)
        # Ensure clamping uses computed min/max
        x_clamped = torch.clamp(x, min=in_min, max=in_max)

        # Find which segment each input value belongs to
        # Use searchsorted on the clamped value to avoid out-of-bounds indices
        intervals = torch.searchsorted(knots, x_clamped, right=False) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)

        # Compute interpolation parameter t in [0,1]
        # Avoid division by zero if knots are identical (shouldn't happen with linspace)
        knot_diff = knots[intervals + 1] - knots[intervals]
        # Use clamped x for t calculation relative to the segment
        t = (x_clamped - knots[intervals]) / (knot_diff + 1e-8)
        t = torch.clamp(t, 0.0, 1.0) # Clamp t explicitly for numerical stability

        coeffs_to_use = self.coeffs
        if self.monotonic:
            # For φ splines, ensure coefficients are sorted to maintain monotonicity
            # Note: This sort is done *every forward pass*. As noted, alternatives
            # exist but are not the focus here per the instructions.
            coeffs_to_use = torch.sort(self.coeffs)[0]

        # Perform linear interpolation
        raw = (1 - t) * coeffs_to_use[intervals] + t * coeffs_to_use[intervals + 1]

        # If train_range is enabled, re-scale the raw output.
        # We normalize raw using initial coeff min/max (stored during first call)
        # and then map to the current out_range.
        eps = 1e-8
        if self.train_range and (self.range_params is not None):
            # Use the currently sorted coeffs if monotonic for initial range calculation
            current_coeffs_for_range = coeffs_to_use if self.monotonic else self.coeffs
            if not hasattr(self, 'init_coeff_min') or not hasattr(self, 'init_coeff_max'):
                 # Store initial min and max based on the possibly sorted coefficients
                 with torch.no_grad():
                    # Ensure buffers are created on the correct device
                    self.register_buffer('init_coeff_min', current_coeffs_for_range.data.min().clone().detach())
                    self.register_buffer('init_coeff_max', current_coeffs_for_range.data.max().clone().detach())


            coeff_min = self.init_coeff_min
            coeff_max = self.init_coeff_max

            # Check if initial range is valid
            if coeff_max - coeff_min > eps:
                 raw_normalized = (raw - coeff_min) / (coeff_max - coeff_min + eps)
                 final = out_min + (out_max - out_min) * raw_normalized
            else: # Handle case where initial coefficients were flat
                 final = torch.full_like(raw, (out_min + out_max) / 2.0)

        else:
            final = raw

        return final


    def get_flatness_penalty(self):
        coeffs_to_use = self.coeffs
        if self.monotonic:
             # Use sorted coeffs for penalty calculation to reflect the actual function shape
            coeffs_to_use = torch.sort(self.coeffs)[0]

        if len(coeffs_to_use) < 3:
            return torch.tensor(0.0, device=coeffs_to_use.device)

        second_diff = coeffs_to_use[2:] - 2 * coeffs_to_use[1:-1] + coeffs_to_use[:-2]
        return torch.mean(second_diff ** 2)

##############################################################################
#                          SPRECHER LAYER BLOCK                              #
##############################################################################

class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ (with fixed (0,1) domain and range)
      - A general spline 𝛷 (with trainable domain and codomain)
      - A trainable shift (η)
      - A trainable weight matrix (λ)
    If is_final is True, the block sums its outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, is_final=False, phi_range_params=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final

        # The inner monotonic spline φ (with fixed range (0,1))
        # This represents the 'inner function' in Sprecher's theorem
        self.phi = SimpleSpline(num_knots=PHI_KNOTS, in_range=(0, 1), out_range=(0, 1), monotonic=True)

        # The outer general spline 𝛷 with trainable range parameters
        # This represents the 'outer function' in Sprecher's theorem
        self.Phi = SimpleSpline(num_knots=PHI_CAPITAL_KNOTS, in_range=PHI_IN_RANGE, out_range=PHI_OUT_RANGE,
                                train_range=TRAIN_PHI_RANGE, range_params=phi_range_params)

        # Explicitly initialize Phi coeffs linearly within the initial range
        with torch.no_grad():
            if TRAIN_PHI_RANGE and phi_range_params is not None:
                cc = phi_range_params.cc.item()
                cr = abs(phi_range_params.cr.item()) # Use abs
                out_min_init = cc - cr
                out_max_init = cc + cr
            else:
                out_min_init, out_max_init = PHI_OUT_RANGE
            # Ensure coeffs are initialized linearly within the *initial* dynamic range
            self.Phi.coeffs.data = torch.linspace(out_min_init, out_max_init, self.Phi.num_knots, device=self.Phi.coeffs.device)
            # Store initial min/max based on this linear initialization
            if TRAIN_PHI_RANGE and phi_range_params is not None:
                 if not hasattr(self.Phi, 'init_coeff_min'):
                    self.Phi.register_buffer('init_coeff_min', self.Phi.coeffs.data.min().clone().detach())
                 if not hasattr(self.Phi, 'init_coeff_max'):
                    self.Phi.register_buffer('init_coeff_max', self.Phi.coeffs.data.max().clone().detach())


        # Weight matrix and shift parameter. Ensure Parameter names match for grouping.
        self.lambdas = nn.Parameter(torch.ones(d_in, d_out))
        self.eta = nn.Parameter(torch.tensor(1.0 / 100.0)) # Ensure name is 'eta'

        # Q-values for indexing.
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, d_in)
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # shape: (1, 1, d_out)

        # Apply translation by eta * q (part of Sprecher's construction)
        # Ensure eta is non-negative if desired, or handle potential sign flips
        # For simplicity here, we don't constrain eta, but ensure it's used consistently.
        shifted = x_expanded + self.eta * q
        # Clamp to the fixed domain [0, 1] of the inner spline phi
        shifted = torch.sigmoid(5.0 * (shifted - 0.5))   # factor 5 ≈ hard clamp; tune if you like

        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)

        # Weight by λ and sum over input dimension (weighted superposition)
        weighted = phi_out * self.lambdas.unsqueeze(0)
        # Adding q serves as a learnable bias/offset for each dimension before Phi
        s = weighted.sum(dim=1) + self.q_values  # shape: (batch_size, d_out)

        # Pass through the general outer spline 𝛷
        activated = self.Phi(s)

        if self.is_final:
            # Sum outputs if this is the final block (produces scalar output)
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated

##############################################################################
#                     MULTI-LAYER SPRECHER NETWORK                           #
##############################################################################

class SprecherMultiLayerNetwork(nn.Module):
    """
    Builds the Sprecher network with a given hidden-layer architecture and final output dimension.
    A single global PhiRangeParams instance is created and shared among all 𝛷 splines.
    """
    def __init__(self, input_dim, architecture, final_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim

        # Compute domain and codomain centers and radii (for 𝛷) from the configured outer ranges.
        dc = (PHI_IN_RANGE[0] + PHI_IN_RANGE[1]) / 2.0
        dr = (PHI_IN_RANGE[1] - PHI_IN_RANGE[0]) / 2.0
        cc = (PHI_OUT_RANGE[0] + PHI_OUT_RANGE[1]) / 2.0
        cr = (PHI_OUT_RANGE[1] - PHI_OUT_RANGE[0]) / 2.0

        # Create global trainable parameters for all 𝛷 splines
        self.phi_range_params = PhiRangeParams(dc=dc, dr=dr, cc=cc, cr=cr)

        layers = []
        current_dim = input_dim
        # Handle case with no hidden layers
        if not architecture:
             # Determine if the single block is final based on final_dim
             is_final = (final_dim == 1)
             # Output dimension of the single block is final_dim
             layers.append(SprecherLayerBlock(d_in=current_dim, d_out=final_dim,
                                              is_final=is_final, phi_range_params=self.phi_range_params))
        else:
            # Iterate through hidden layers defined in architecture
            for i, hidden_dim in enumerate(architecture):
                is_final_layer = (i == len(architecture) - 1) and (final_dim == 1)
                output_dim = hidden_dim
                # If it's the last hidden layer AND the final output is scalar,
                # this block should perform the final summation.
                if is_final_layer:
                     # The output dimension of this block IS the last hidden dim,
                     # but it performs summation if final_dim is 1.
                     layers.append(SprecherLayerBlock(d_in=current_dim, d_out=output_dim,
                                                     is_final=True, phi_range_params=self.phi_range_params))
                     current_dim = output_dim # Update current_dim for potential next layer
                else:
                    # Not the final summing layer
                     layers.append(SprecherLayerBlock(d_in=current_dim, d_out=output_dim,
                                                     is_final=False, phi_range_params=self.phi_range_params))
                     current_dim = output_dim

            # Add a final block ONLY if the desired final_dim is different from the last hidden layer's dim
            # AND the last block wasn't already the final summing block (i.e., final_dim > 1).
            if final_dim > 1 and current_dim != final_dim:
                 layers.append(SprecherLayerBlock(d_in=current_dim, d_out=final_dim,
                                                  is_final=False, # This block doesn't sum, just transforms dims
                                                  phi_range_params=self.phi_range_params))


        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

##############################################################################
#                           DATA GENERATION                                  #
##############################################################################

def generate_data(n_samples_per_dim=32):
    """
    Generates a grid of points in [0,1]^d and computes the target function.
    n_samples_per_dim determines the resolution along each axis.
    """
    input_dim = get_input_dimension(target_function)
    print(f"Generating data with input dimension: {input_dim}") # Debug print
    if input_dim == 1:
        x = torch.linspace(0, 1, n_samples_per_dim).unsqueeze(1)
        print(f"Generated {x.shape[0]} data points for 1D input.") # Debug print
    elif input_dim == 2:
        # Use n_samples_per_dim for each dimension of the grid
        x_coords = torch.linspace(0, 1, n_samples_per_dim)
        y_coords = torch.linspace(0, 1, n_samples_per_dim)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        print(f"Generated {x.shape[0]} data points ({n_samples_per_dim}x{n_samples_per_dim}) for 2D input.") # Debug print
    elif input_dim == 3:
         # Extend to 3D if needed (though target function is currently 2D)
         x_coords = torch.linspace(0, 1, n_samples_per_dim)
         y_coords = torch.linspace(0, 1, n_samples_per_dim)
         z_coords = torch.linspace(0, 1, n_samples_per_dim)
         X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
         x = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
         print(f"Generated {x.shape[0]} data points ({n_samples_per_dim}^3) for 3D input.") # Debug print
    else:
         raise ValueError(f"Data generation not implemented for input dimension: {input_dim}")


    y = target_function(x)
    if y.dim() == 1:
        y = y.unsqueeze(1) # Ensure output is always [N, D_out]
    return x, y, input_dim


##############################################################################
#                           TRAINING FUNCTION                                #
##############################################################################

def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu", final_dim=1):
    # Determine n_samples based on input dim to keep total points roughly constant (~1024)
    input_dim = get_input_dimension(target_function)
    if input_dim == 1:
        n_samples_per_dim = 1024
    elif input_dim == 2:
        n_samples_per_dim = 32 # 32*32 = 1024
    elif input_dim == 3:
        n_samples_per_dim = 10 # 10*10*10 = 1000
    else:
        n_samples_per_dim = 32 # Default fallback

    x_train, y_train, _ = generate_data(n_samples_per_dim=n_samples_per_dim) # Use determined sample size
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = SprecherMultiLayerNetwork(input_dim=input_dim,
                                      architecture=architecture,
                                      final_dim=final_dim).to(device)

    # --- Define Parameter Groups for Optimizer ---
    # Group parameters carefully to apply different LR strategies
    phi_range_group = []
    cycled_lr_group = [] # Parameters whose LR will be cycled
    other_group = []     # Fallback for any missed parameters

    # Store indices for easy access later (e.g., for logging LR)
    group_indices = {"phi_range": -1, "cycled": -1, "other": -1}
    current_group_index = 0

    # Separate phi_range parameters first if they exist and require grad
    if TRAIN_PHI_RANGE and hasattr(model, 'phi_range_params'):
        params = [p for p in model.phi_range_params.parameters() if p.requires_grad]
        if params:
            phi_range_group.extend(params)
            group_indices["phi_range"] = current_group_index
            current_group_index += 1

    # Group other parameters
    phi_range_param_ids = {id(p) for p in phi_range_group} # Get IDs for quick lookup
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in phi_range_param_ids:
            continue # Skip non-trainable or already grouped range params

        # Group coeffs, eta, lambdas into the cycled group
        if "coeffs" in name or "eta" in name or "lambdas" in name:
            cycled_lr_group.append(param)
        else:
            print(f"Warning: Parameter '{name}' assigned to 'other' group.")
            other_group.append(param)

    # Define base learning rates for the optimizer configuration
    params_for_optimizer = []
    base_lrs_for_scheduler = []
    max_lrs_for_scheduler = []

    # Add phi_range group if it exists
    if phi_range_group:
        params_for_optimizer.append({"params": phi_range_group, "lr": CLR_CONST_LR_RANGE, "name": "phi_range"})
        base_lrs_for_scheduler.append(CLR_CONST_LR_RANGE) # Constant LR
        max_lrs_for_scheduler.append(CLR_CONST_LR_RANGE)  # Constant LR
        print(f"Parameter group 'phi_range' (constant LR={CLR_CONST_LR_RANGE}) assigned index {group_indices['phi_range']}.")

    # Add cycled group if it exists
    if cycled_lr_group:
        group_indices["cycled"] = current_group_index
        params_for_optimizer.append({"params": cycled_lr_group, "lr": CLR_BASE_LR_BOOSTABLE, "name": "cycled"}) # Use base LR for optimizer init
        base_lrs_for_scheduler.append(CLR_BASE_LR_BOOSTABLE) # Base LR for cycle
        max_lrs_for_scheduler.append(CLR_MAX_LR_BOOSTABLE)  # Max LR for cycle
        print(f"Parameter group 'cycled' (LR cycle: {CLR_BASE_LR_BOOSTABLE}-{CLR_MAX_LR_BOOSTABLE}) assigned index {group_indices['cycled']}.")
        current_group_index += 1
    else:
        print("Warning: No parameters found for the 'cycled' LR group.")


    # Add other group if it exists
    if other_group:
        group_indices["other"] = current_group_index
        # Assign a default constant LR for 'other' params, e.g., same as cycled base LR
        other_lr = CLR_BASE_LR_BOOSTABLE
        params_for_optimizer.append({"params": other_group, "lr": other_lr, "name": "other"})
        base_lrs_for_scheduler.append(other_lr) # Constant LR
        max_lrs_for_scheduler.append(other_lr)  # Constant LR
        print(f"Parameter group 'other' (constant LR={other_lr}) assigned index {group_indices['other']}.")
        current_group_index += 1

    optimizer = torch.optim.Adam(params_for_optimizer, weight_decay=1e-7)

    # --- Setup CyclicLR Scheduler ---
    scheduler = None
    if cycled_lr_group: # Only setup scheduler if there are params to cycle
        try:
            scheduler = CyclicLR(optimizer,
                                 base_lr=base_lrs_for_scheduler,
                                 max_lr=max_lrs_for_scheduler,
                                 step_size_up=CLR_STEP_SIZE_UP,
                                 mode=CLR_MODE,
                                 gamma=CLR_GAMMA if CLR_MODE=='exp_range' else 1.0,
                                 cycle_momentum=False) # Adam handles momentum internally
            print(f"CyclicLR scheduler initialized: mode='{CLR_MODE}', step_size_up={CLR_STEP_SIZE_UP} epochs.")
            print(f"  Base LRs per group: {base_lrs_for_scheduler}")
            print(f"  Max LRs per group: {max_lrs_for_scheduler}")
        except Exception as e:
             print(f"Error initializing CyclicLR scheduler: {e}")
             print("Training will proceed without CyclicLR.")
             scheduler = None # Ensure scheduler is None if init fails
    else:
        print("Skipping CyclicLR initialization as 'cycled' group is empty.")


    # Calculate total trainable parameters
    total_params = sum(p.numel() for group in params_for_optimizer for p in group['params'])
    phi_params_count = sum(p.numel() for p in phi_range_group)
    print(f"Total number of trainable parameters: {total_params}")
    if TRAIN_PHI_RANGE and phi_params_count > 0:
        print(f"Including {phi_params_count} global range parameters (dc, dr, cc, cr) in the total count (using constant LR).")
    elif TRAIN_PHI_RANGE:
         print(f"Global range parameters set to be trainable, but none found in the model.")
    else:
         print(f"Global range parameters are not trained.")


    # --- Training Loop Initialization ---
    losses = []
    learning_rates_cycled = [] # To store LR history for plotting
    # Overall best tracking
    best_global_loss = float("inf")
    best_global_state = None

    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:

        # --- Standard Training Step ---
        model.train() # Ensure model is in training mode
        optimizer.zero_grad()
        output = model(x_train)

        mse_loss = torch.mean((output - y_train) ** 2)

        # Calculate smoothness penalty (regularization term)
        flatness_penalty = 0.0
        for layer in model.layers:
            # Apply penalty to both phi and Phi splines
            flatness_penalty += 0.01 * layer.phi.get_flatness_penalty() + 0.1 * layer.Phi.get_flatness_penalty()

        # Variance penalty: encourage network output variability to match that of the target
        std_target = torch.std(y_train)
        std_output = torch.std(output)
        var_loss = LAMBDA_VAR * (std_output - std_target) ** 2

        # Combine all loss components
        loss = mse_loss + flatness_penalty + var_loss
        loss_item = loss.item() # Store loss value before backward potentially modifies things
        loss.backward()

        # Capture gradients for the global range parameters for debugging (if they exist and have grads)
        dc_val, dr_val, cc_val, cr_val = float('nan'), float('nan'), float('nan'), float('nan')
        dc_grad, dr_grad, cc_grad, cr_grad = float('nan'), float('nan'), float('nan'), float('nan')
        if TRAIN_PHI_RANGE and group_indices["phi_range"] != -1: # Check if group exists
            try:
                # Access params through the model attribute directly if possible
                if hasattr(model, 'phi_range_params'):
                    prp = model.phi_range_params
                    if prp.dc.requires_grad:
                        dc_val = prp.dc.item()
                        dr_val = prp.dr.item()
                        cc_val = prp.cc.item()
                        cr_val = prp.cr.item()
                        dc_grad = prp.dc.grad.item() if prp.dc.grad is not None else 0.0
                        dr_grad = prp.dr.grad.item() if prp.dr.grad is not None else 0.0
                        cc_grad = prp.cc.grad.item() if prp.cc.grad is not None else 0.0
                        cr_grad = prp.cr.grad.item() if prp.cr.grad is not None else 0.0
            except AttributeError:
                 pass # Silently ignore if phi_range_params not found or params missing


        optimizer.step()

        # --- Step the LR scheduler ---
        # Store LR *before* stepping for logging the LR used in the current epoch's update
        current_lr_cycled = float('nan')
        if scheduler and group_indices["cycled"] != -1:
             # Get the LR for the 'cycled' group (assuming it exists)
             current_lr_cycled = optimizer.param_groups[group_indices["cycled"]]['lr']
             learning_rates_cycled.append(current_lr_cycled)
             scheduler.step() # Step the scheduler AFTER the optimizer step
        elif group_indices["cycled"] != -1:
             # If no scheduler but group exists, log the constant base LR
             current_lr_cycled = optimizer.param_groups[group_indices["cycled"]]['lr']
             learning_rates_cycled.append(current_lr_cycled)
        else:
             learning_rates_cycled.append(float('nan')) # Append NaN if cycled group doesn't exist


        losses.append(loss_item)

        # --- Update Best Global State ---
        if loss_item < best_global_loss:
            best_global_loss = loss_item
            best_global_state = copy.deepcopy(model.state_dict()) # Use deepcopy

        # --- Update Progress Bar ---
        postfix_dict = {
            'loss': f'{loss_item:.2e}',
            'LR_cyc': f'{current_lr_cycled:.2e}' if not math.isnan(current_lr_cycled) else 'N/A', # Show current cycled LR
            'best_glob': f'{best_global_loss:.2e}', # Show global best loss
            'std_out': f'{std_output.item():.3f}',
            'std_tar': f'{std_target.item():.3f}',
            'var_loss': f'{var_loss.item():.2e}'
        }

        if TRAIN_PHI_RANGE and group_indices["phi_range"] != -1:
             postfix_dict.update({
                 'dc': f'{dc_val:.3f}',
                 'dr': f'{dr_val:.3f}',
                 'cc': f'{cc_val:.3f}',
                 'cr': f'{cr_val:.3f}',
             })
        pbar.set_postfix(postfix_dict)

        # --- Periodic Printing ---
        if (epoch + 1) % print_every == 0:
            print(f"\nEpoch {epoch+1}/{total_epochs}: Loss = {loss_item:.4e}, Cycled LR = {current_lr_cycled:.4e}, Best Global Loss: {best_global_loss:.4e}")

    # --- End of Training ---
    print(f"\nFinished training. Overall best loss achieved: {best_global_loss:.4e}")
    if best_global_state is not None:
        print("Loading best global model state found during training.")
        model.load_state_dict(best_global_state)
    else:
        print("Warning: No best global state recorded.")


    # Print final eta and lambda parameters for each block
    for idx, layer in enumerate(model.layers, start=1):
         print("-" * 20)
         print(f"Block {idx}: Final Parameters (from best global state)")
         print(f"  eta = {layer.eta.item():.6f}")
         if hasattr(layer, 'lambdas'):
              print(f"  lambdas shape = {tuple(layer.lambdas.shape)}")
              # Avoid printing huge matrices if they are large
              if layer.lambdas.numel() > 100: # Increased limit slightly
                   print(f"  lambdas (first 5x5 elems or fewer) = \n{layer.lambdas.detach().cpu().numpy()[:5, :5]}")
              elif layer.lambdas.numel() > 0 :
                   print(f"  lambdas = \n{layer.lambdas.detach().cpu().numpy()}")
              else:
                   print(f"  lambdas = (empty tensor)")
         if hasattr(layer, 'phi') and hasattr(layer.phi, 'coeffs'):
             phi_coeffs = layer.phi.coeffs.detach().cpu().numpy()
             print(f"  phi coeffs range: [{phi_coeffs.min():.4f}, {phi_coeffs.max():.4f}]")
         if hasattr(layer, 'Phi') and hasattr(layer.Phi, 'coeffs'):
             Phi_coeffs = layer.Phi.coeffs.detach().cpu().numpy()
             print(f"  Phi coeffs range: [{Phi_coeffs.min():.4f}, {Phi_coeffs.max():.4f}]")

    print("-" * 20)

    # Return the LR history along with other results
    return model, losses, model.layers, learning_rates_cycled


##############################################################################
#                    PLOTTING: NETWORK SCHEMATIC + SPLINES                   #
##############################################################################

def plot_network_structure_ax(ax, layers, input_dim, final_dim=1):
    num_blocks = len(layers)
    if num_blocks == 0:
        ax.text(0.5, 0.5, "No network layers", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return

    # Determine the number of node columns needed (input + hidden + output)
    # If final_dim=1 and last block is_final, output is conceptual after last block.
    # Otherwise, the output is represented by the nodes of the last block.
    last_block_is_final_summing = layers[-1].is_final if layers else False
    num_node_layers = num_blocks + 1 # Input layer + one layer per block output
    if final_dim == 1 and last_block_is_final_summing:
        num_node_layers += 1 # Add extra column for the final summed output node

    layer_x = np.linspace(0.1, 0.9, num_node_layers)

    # --- Input Layer ---
    input_y = np.linspace(0.2, 0.8, max(1, input_dim))
    first_hidden_layer_d_out = layers[0].d_out if layers else 1
    first_hidden_y = np.linspace(0.2, 0.8, max(1, first_hidden_layer_d_out))
    input_node_y_center = np.mean(first_hidden_y) if input_dim == 1 else 0.5 # Center input if 1D

    if input_dim == 1:
         # Draw single input node centered vertically w.r.t first hidden layer
         ax.scatter(layer_x[0], input_node_y_center, c='black', s=80, zorder=5)
         ax.text(layer_x[0] - 0.03, input_node_y_center, '$x_1$', ha='right', va='center', fontsize=8)
         prev_y = [input_node_y_center] # Use the calculated center y for connections
    else:
         # Draw multiple input nodes
         for i, y in enumerate(input_y):
             ax.scatter(layer_x[0], y, c='black', s=80, zorder=5)
             ax.text(layer_x[0] - 0.03, y, f'$x_{{{i+1}}}$', ha='right', va='center', fontsize=8)
         prev_y = input_y

    prev_x = layer_x[0]
    node_ys = [prev_y] # Store y-coords for each layer

    # --- Hidden Layers ---
    for block_index, block in enumerate(layers):
        current_block_output_x = layer_x[block_index + 1]
        d_out = block.d_out
        # Determine color: Blue for hidden, Red if it's the *effective* output layer
        is_effective_output_layer = (block_index == num_blocks - 1) and not block.is_final
        node_color = 'red' if is_effective_output_layer else 'blue'

        # If this block performs the final sum, its *nodes* are still conceptually hidden (blue)
        if block.is_final:
             node_color = 'blue'

        current_y = np.linspace(0.2, 0.8, max(1, d_out))
        node_ys.append(current_y)

        # Draw nodes for this block's output
        for y in current_y:
            ax.scatter(current_block_output_x, y, c=node_color, s=80, zorder=5)

        # Draw connections from previous layer
        for py in prev_y:
            for cy in current_y:
                ax.plot([prev_x, current_block_output_x], [py, cy], 'k-', alpha=0.2, zorder=1) # Reduced alpha

        # Add block labels and spline symbols
        block_label_x = (prev_x + current_block_output_x) / 2.0
        ax.text(block_label_x, 0.95, f"Block {block_index+1}", ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(block_label_x, 0.10, f"$\\phi^{{({block_index+1})}}$", ha='center', va='center', fontsize=9, color='darkcyan')
        ax.text(block_label_x, 0.05, f"$\\Phi^{{({block_index+1})}}$", ha='center', va='center', fontsize=9, color='purple')

        prev_x = current_block_output_x
        prev_y = current_y

    # --- Output Layer ---
    # Check if a final summing node needs to be drawn explicitly
    if final_dim == 1 and last_block_is_final_summing:
        output_x = layer_x[-1] # Use the last allocated x position
        output_y = np.mean(prev_y) # Position vertically centered w.r.t previous layer
        ax.scatter(output_x, output_y, c='red', s=80, zorder=5) # Final output node is red
        # Draw connections from last hidden layer to the final summed node
        for py in prev_y:
            ax.plot([prev_x, output_x], [py, output_y], 'k-', alpha=0.2, zorder=1)
        # Add output label
        ax.text(output_x + 0.03, output_y, "$\\hat{y}$", ha='left', va='center', fontsize=10)

    # Label the output if it was represented by the nodes of the last block (vector output case)
    elif not last_block_is_final_summing and node_ys:
         last_node_y = node_ys[-1]
         last_node_x = layer_x[num_blocks] # x-position of the last block's output nodes
         output_label = f"$\\hat{{y}}_{{1..{final_dim}}}$"
         ax.text(last_node_x + 0.03, np.mean(last_node_y), output_label, ha='left', va='center', fontsize=10)


    ax.set_title("Network Structure", fontsize=11)
    ax.axis('off')
    ax.set_ylim(0, 1)


def plot_results(model, layers):
    if layers:
        input_dim = layers[0].d_in
    else:
        input_dim = getattr(model, 'input_dim', get_input_dimension(target_function))

    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', get_output_dimension(target_function, input_dim))

    top_cols = 1 + 2 * num_blocks
    if num_blocks == 0: top_cols = 1

    plt.rcParams.update({'xtick.labelsize': 7, 'ytick.labelsize': 7})

    fig_width = max(12, 2 * top_cols)
    fig_height = 10

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.4)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, top_cols, subplot_spec=gs_main[0],
                                              width_ratios=[1.5] + [1.0] * (top_cols - 1) if top_cols > 1 else [1],
                                              wspace=0.4)

    # Plot Network Structure
    ax_net = fig.add_subplot(gs_top[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim, final_dim)

    # Plot Splines
    for i, layer in enumerate(layers):
        j = i + 1
        col_phi = 1 + 2 * i
        col_Phi = 1 + 2 * i + 1

        # Plot inner spline phi
        if col_phi < top_cols:
            ax_phi = fig.add_subplot(gs_top[0, col_phi])
            block_label = f"Block {j}"
            ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$ (Monotonic)", fontsize=9)
            with torch.no_grad():
                 if hasattr(layer, 'phi') and hasattr(layer.phi, 'coeffs'):
                     x_vals = torch.linspace(layer.phi.fixed_in_range[0], layer.phi.fixed_in_range[1], 200).to(layer.phi.coeffs.device)
                     y_vals = layer.phi(x_vals)
                     ax_phi.plot(x_vals.cpu(), y_vals.cpu(), color='darkcyan', linewidth=1.5)
                 else:
                     ax_phi.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax_phi.grid(True, linestyle=':', alpha=0.6)
            ax_phi.tick_params(axis='both', which='major', labelsize=7)

        # Plot outer spline Phi
        if col_Phi < top_cols:
            ax_Phi = fig.add_subplot(gs_top[0, col_Phi])
            ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$ (General)", fontsize=9)
            with torch.no_grad():
                 if hasattr(layer, 'Phi') and hasattr(layer.Phi, 'coeffs'):
                     if layer.Phi.train_range and (layer.Phi.range_params is not None):
                         dc = layer.Phi.range_params.dc.item()
                         dr = abs(layer.Phi.range_params.dr.item())
                         in_min, in_max = dc - dr, dc + dr
                         # Shortened x-label format
                         ax_Phi.set_xlabel(f"Input([{in_min:.2f}, {in_max:.2f}])", fontsize=7)
                     else:
                         in_min, in_max = layer.Phi.fixed_in_range
                         ax_Phi.set_xlabel(f"Input (Fixed [{in_min:.2f}, {in_max:.2f}])", fontsize=7)

                     x_vals = torch.linspace(in_min, in_max, 200).to(layer.Phi.coeffs.device)
                     y_vals = layer.Phi(x_vals)
                     ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), color='purple', linewidth=1.5)
                 else:
                     ax_Phi.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax_Phi.grid(True, linestyle=':', alpha=0.6)
            ax_Phi.tick_params(axis='both', which='major', labelsize=7)

    device = next(model.parameters()).device
    model.eval() # Set model to evaluation mode for plotting

    # Plot Function Comparison / Surfaces
    if input_dim == 1:
        n_test = 400
        x_test = torch.linspace(0, 1, n_test).reshape(-1, 1).to(device)
        y_true = target_function(x_test)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        with torch.no_grad():
            y_pred = model(x_test)

        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, final_dim, subplot_spec=gs_main[1], wspace=0.3)

        for d in range(final_dim):
            ax_func = fig.add_subplot(gs_bottom[0, d])
            if final_dim > 1:
                ax_func.set_title(f"Function Comparison (Output Dim {d+1})", fontsize=10)
            else:
                 ax_func.set_title("Function Comparison", fontsize=10)

            ax_func.plot(x_test.cpu(), y_true[:, d].cpu(), 'k--', linewidth=1.5, label='Target f(x)')
            ax_func.plot(x_test.cpu(), y_pred[:, d].cpu(), 'r-', linewidth=1.5, alpha=0.8, label='Network Output')
            ax_func.grid(True, linestyle=':', alpha=0.6)
            ax_func.legend(fontsize=8)
            ax_func.set_xlabel("Input x", fontsize=8)
            ax_func.set_ylabel("Output", fontsize=8)
            ax_func.tick_params(axis='both', which='major', labelsize=7)

    elif input_dim == 2:
        n_surf = 50
        x_surf = torch.linspace(0, 1, n_surf)
        y_surf = torch.linspace(0, 1, n_surf)
        X_surf, Y_surf = torch.meshgrid(x_surf, y_surf, indexing='ij')
        points_surf = torch.stack([X_surf.flatten(), Y_surf.flatten()], dim=1).to(device)

        y_true_surf = target_function(points_surf)
        if y_true_surf.dim() == 1: y_true_surf = y_true_surf.unsqueeze(1)

        with torch.no_grad():
            y_pred_surf = model(points_surf)

        num_bottom_cols = final_dim if final_dim > 1 else 2
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, num_bottom_cols, subplot_spec=gs_main[1], wspace=0.15)

        X_np, Y_np = X_surf.cpu().numpy(), Y_surf.cpu().numpy()
        axis_label_pad = 5 # Increased padding for 3D axis labels

        if final_dim == 1:
             Z_true_np = y_true_surf.reshape(n_surf, n_surf).cpu().numpy()
             Z_pred_np = y_pred_surf.reshape(n_surf, n_surf).cpu().numpy()

             ax_target = fig.add_subplot(gs_bottom[0, 0], projection='3d')
             ax_target.set_title("Target Function Surface", fontsize=10)
             ax_target.plot_surface(X_np, Y_np, Z_true_np, cmap='viridis', alpha=0.9)
             ax_target.tick_params(axis='both', which='major', labelsize=6)
             ax_target.set_xlabel('x1', fontsize=7, labelpad=axis_label_pad) # Increased padding
             ax_target.set_ylabel('x2', fontsize=7, labelpad=axis_label_pad) # Increased padding
             ax_target.set_zlabel('f(x1, x2)', fontsize=7, labelpad=axis_label_pad) # Increased padding


             ax_output = fig.add_subplot(gs_bottom[0, 1], projection='3d')
             ax_output.set_title("Network Output Surface", fontsize=10)
             ax_output.plot_surface(X_np, Y_np, Z_pred_np, cmap='plasma', alpha=0.9)
             ax_output.tick_params(axis='both', which='major', labelsize=6)
             ax_output.set_xlabel('x1', fontsize=7, labelpad=axis_label_pad) # Increased padding
             ax_output.set_ylabel('x2', fontsize=7, labelpad=axis_label_pad) # Increased padding
             ax_output.set_zlabel('ŷ(x1, x2)', fontsize=7, labelpad=axis_label_pad) # Increased padding

             z_min = min(Z_true_np.min(), Z_pred_np.min())
             z_max = max(Z_true_np.max(), Z_pred_np.max())
             # Add small margin to zlim to prevent clipping at exact min/max
             z_range = z_max - z_min
             z_margin = z_range * 0.05 if z_range > 1e-6 else 0.1
             ax_target.set_zlim(z_min - z_margin, z_max + z_margin)
             ax_output.set_zlim(z_min - z_margin, z_max + z_margin)


        else: # Vector output (final_dim > 1)
            from matplotlib.lines import Line2D # Import only if needed
            for d in range(final_dim):
                ax_out = fig.add_subplot(gs_bottom[0, d], projection='3d')
                ax_out.set_title(f"Output Dim {d+1}", fontsize=10)

                Z_true_d = y_true_surf[:, d].reshape(n_surf, n_surf).cpu().numpy()
                Z_pred_d = y_pred_surf[:, d].reshape(n_surf, n_surf).cpu().numpy()

                ax_out.plot_surface(X_np, Y_np, Z_true_d, cmap='viridis', alpha=0.6, label='Target')
                ax_out.plot_surface(X_np, Y_np, Z_pred_d, cmap='plasma', alpha=0.6, label='Output')

                ax_out.tick_params(axis='both', which='major', labelsize=6)
                ax_out.set_xlabel('x1', fontsize=7, labelpad=axis_label_pad) # Increased padding
                ax_out.set_ylabel('x2', fontsize=7, labelpad=axis_label_pad) # Increased padding
                ax_out.set_zlabel(f'Dim {d+1}', fontsize=7, labelpad=axis_label_pad) # Increased padding

                legend_elements = [Line2D([0], [0], color='k', markerfacecolor=plt.cm.viridis(0.5), marker='s', linestyle='None', markersize=8, label='Target'),
                                Line2D([0], [0], color='k', markerfacecolor=plt.cm.plasma(0.5), marker='s', linestyle='None', markersize=8, label='Output')]
                ax_out.legend(handles=legend_elements, fontsize=7, loc='upper left')

                z_min_d = min(Z_true_d.min(), Z_pred_d.min())
                z_max_d = max(Z_true_d.max(), Z_pred_d.max())
                # Add small margin to zlim
                z_range_d = z_max_d - z_min_d
                z_margin_d = z_range_d * 0.05 if z_range_d > 1e-6 else 0.1
                ax_out.set_zlim(z_min_d - z_margin_d, z_max_d + z_margin_d)


    else: # Input dim > 2
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[1])
        ax_func = fig.add_subplot(gs_bottom[0, 0])
        ax_func.text(0.5, 0.5, f"Plotting not implemented for input dimension {input_dim}",
                     ha='center', va='center', fontsize=12)
        ax_func.axis('off')

    # Update title to reflect CLR usage
    fig.suptitle(f"Sprecher Network Results ({TOTAL_EPOCHS} Epochs, Arch: {architecture}, CyclicLR)", fontsize=14, y=0.99)
    # Use tight_layout carefully, suppress warning if known incompatibility exists (like 3D plots)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    except UserWarning:
        print("Note: tight_layout may not be fully compatible with 3D axes.")
    except Exception as e:
        print(f"Error during tight_layout: {e}")


    return fig


##############################################################################
#                              MAIN EXECUTION                                #
##############################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Automatically determine input and output dimensions
    input_dim = get_input_dimension(target_function)
    final_dim = get_output_dimension(target_function, input_dim)
    print(f"Target function - Input Dim: {input_dim}, Output Dim: {final_dim}")

    # --- Start Training ---
    # train_network now returns learning_rates_cycled as the 4th item
    model, losses, layers, learning_rates_cycled = train_network(
        target_function,
        architecture=architecture,
        total_epochs=TOTAL_EPOCHS,
        print_every=max(1, TOTAL_EPOCHS // 100), # Print more often
        device=device,
        final_dim=final_dim
    )

    # --- Plotting Results ---
    print("Generating results plots...")
    fig_results = plot_results(model, layers)

    if SAVE_FINAL_PLOT:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        prefix = f"{input_dim}D_Input"
        arch_str = "-".join(map(str, architecture)) if architecture else "NoHidden"
        # Update filename to reflect CLR usage
        filename = f"{prefix}_{arch_str}_Out{final_dim}_{TOTAL_EPOCHS}epochs_CyclicLR.png"
        filepath = os.path.join(plots_dir, filename)
        dpi = fig_results.get_dpi()
        # Ensure calculated size is positive
        save_width = max(1, FINAL_IMAGE_WIDTH / dpi)
        save_height = max(1, FINAL_IMAGE_HEIGHT / dpi)
        fig_results.set_size_inches(save_width, save_height)
        try:
             fig_results.savefig(filepath, dpi=dpi, bbox_inches='tight')
             print(f"Final plot saved as {filepath}")
        except Exception as e:
             print(f"Error saving final plot: {e}")


    plt.show() # Display the results plot

    # --- Plotting Loss Curve and Learning Rate ---
    print("Generating loss curve and LR plot...")
    fig_loss_lr, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Loss on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Log Scale)", color=color)
    ax1.semilogy(losses, label="Training Loss", alpha=0.8, linewidth=1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.set_title("Training Loss Curve and Cycled Learning Rate")

    # Plot Learning Rate on secondary y-axis
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Cycled Learning Rate', color=color)
    # Filter out potential NaNs if cycled group didn't exist
    valid_lr_indices = [i for i, lr in enumerate(learning_rates_cycled) if not math.isnan(lr)]
    valid_lrs = [lr for lr in learning_rates_cycled if not math.isnan(lr)]
    if valid_lrs:
        ax2.plot(valid_lr_indices, valid_lrs, label='Cycled LR', color=color, alpha=0.7, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color)
    # Optional: Use log scale for LR if it varies a lot, but linear is often clearer for cycles
    # ax2.set_yscale('log')

    fig_loss_lr.tight_layout() # otherwise the right y-label is slightly clipped

    # Add combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines2: # Only add LR legend if it was plotted
        ax2.legend(lines + lines2, labels + labels2, loc='upper center')
    else:
        ax1.legend(lines, labels, loc='upper center')


    if SAVE_FINAL_PLOT:
        # Update filename
        loss_filename = f"{prefix}_{arch_str}_Out{final_dim}_{TOTAL_EPOCHS}epochs_CyclicLR_LossCurve.png"
        loss_filepath = os.path.join(plots_dir, loss_filename)
        try:
            fig_loss_lr.savefig(loss_filepath, dpi=150, bbox_inches='tight')
            print(f"Loss curve saved as {loss_filepath}")
        except Exception as e:
            print(f"Error saving loss curve plot: {e}")

    plt.show() # Display the loss curve plot
    print("Script finished.")
