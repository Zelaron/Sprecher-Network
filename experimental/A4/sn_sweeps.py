"""Batch sweep runner for Sprecher Network experiments."""

import subprocess
import sys


# Sweep configurations
SETUPS = [
    # (dataset,         arch,        phi_knots, Phi_knots, epochs)
    ("toy_1d_poly",     "15,15",       100,       100,       4000),
    ("toy_1d_complex",  "15,15",       100,       100,       4000),
    ("toy_2d",          "10,10,10,10", 100,       100,       4000),
    ("toy_2d_vector",   "15,15",       100,       100,       500),
    ("toy_100d",        "100",         15,        15,        100),
    ("special_bessel",  "2,2,2",       50,        50,        12000),
    ("feynman_uv",      "2,2,2",       100,       100,       12000),
    ("poisson",         "15,15",       25,        25,        4000),
    ("toy_4d_to_5d",    "20,20,20",    50,        50,        2000),
]


def main():
    """Run all sweep configurations."""
    print("Starting Sprecher Network sweep experiments...")
    print(f"Total configurations: {len(SETUPS)}\n")
    
    for i, (dataset, arch, phi_knots, Phi_knots, epochs) in enumerate(SETUPS, 1):
        print(f"Configuration {i}/{len(SETUPS)}")
        print("=" * 60)
        
        cmd = [
            sys.executable, "sn_experiments.py",
            "--dataset", dataset,
            "--arch", arch,
            "--phi_knots", str(phi_knots),
            "--Phi_knots", str(Phi_knots),
            "--epochs", str(epochs),
            "--seed", "0",
            "--save_plots",
            "--no_show"  # Don't show plots in batch mode
        ]
        
        print("▶", " ".join(cmd))
        print()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running configuration: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            # Continue with next configuration
        
        print("\n")
    
    print("All sweeps completed!")
    print("Check the 'plots' directory for results.")


if __name__ == "__main__":
    main()