# Sprecher Network - Nintendo DS Demo

A real-time MNIST digit classifier running on the Nintendo DS, demonstrating the extreme parameter efficiency of Sprecher Networks. This implementation performs neural network inference on 2004-era hardware using only **fixed-point arithmetic** - no floating-point unit required.

## Why This Exists

The Nintendo DS has:
- **67 MHz ARM9** processor
- **4 MB RAM**
- **No FPU** (floating-point unit)

Consider a `784→[100,100]→10` network for MNIST classification:

| Architecture | Parameters | Memory (Q16.16) |
|--------------|------------|-----------------|
| **MLP** | ~89,000 | ~350 KB |
| **Sprecher Network** | ~2,600 | ~10 KB |

At this modest size, an MLP would technically fit. But scale up to `784→[512,512]→10`:

| Architecture | Parameters | Memory (Q16.16) |
|--------------|------------|-----------------|
| **MLP** | ~670,000 | **~2.6 MB** |
| **Sprecher Network** | ~5,200 | ~20 KB |

Or try `784→[1024,1024]→10`:

| Architecture | Parameters | Memory (Q16.16) |
|--------------|------------|-----------------|
| **MLP** | ~1.9M | **~7.4 MB** (exceeds DS RAM!) |
| **Sprecher Network** | ~8,300 | ~32 KB |

**The MLP cannot run. The Sprecher Network fits with room to spare.**

This is the O(N) vs O(N²) scaling difference in action. Sprecher Networks enable architectures on resource-constrained hardware that would otherwise be completely infeasible.

## Prerequisites

### For Emulator (melonDS)

| Software | Purpose |
|----------|---------|
| Python 3.x + PyTorch | Training and weight export |
| [devkitPro](https://devkitpro.org/wiki/Getting_Started) | Nintendo DS toolchain (devkitARM + libnds) |
| [melonDS](https://melonds.kuribo64.net/) | DS emulator (recommended over DeSmuME) |
| MSYS2 | Build environment (included with devkitPro on Windows) |

### For Real Hardware

| Hardware | Purpose |
|----------|---------|
| Nintendo DS / DS Lite / DSi / 3DS | Target device |
| Flashcart (R4, R4i Gold, Ace3DS+, etc.) | Runs homebrew from microSD |
| microSD card | Stores ROM and weights file |
| microSD card reader | For transferring files |

---

## Quick Start (Emulator)

### Step 1: Train and Export

From the main project directory:

```bash
# Train the model
python sn_mnist.py --mode train --arch 100,100 --epochs 10

# Export weights for NDS (creates sprecher_ds/sn_weights.bin)
python sn_mnist.py --mode export_nds
```

### Step 2: Build the ROM

Open the **MSYS2/devkitPro terminal**, then:

```bash
cd /path/to/sprecher-network/sprecher_ds
make clean
make
```

This produces `sprecher_ds.nds` (~250 KB).

### Step 3: Create SD Card Image

```bash
python make_sd.py
```

This creates `sd.img` (~16 MB) containing the weights file with FAT16 filesystem.

### Step 4: Configure melonDS

1. Open melonDS
2. Go to **Config → Emu settings → DLDI**
3. Set **SD card image** to your `sd.img` file
4. **Enable** the "Enable DLDI" checkbox
5. Click OK

### Step 5: Run

1. **File → Open ROM** → select `sprecher_ds.nds`
2. Draw a digit on the bottom (touch) screen
3. Press **A** to classify
4. Press **B** to clear

The top screen displays network info and predictions with confidence scores.

---

## Real Hardware Setup

### Step 1: Prepare Files

After building (Steps 1-2 above), you need two files:
- `sprecher_ds.nds` - the ROM
- `sn_weights.bin` - the trained weights

### Step 2: Set Up Flashcart

#### For R4-style cards (R4, R4i Gold, Ace3DS+, DSTT, etc.)

1. Format microSD as **FAT32** (or FAT16 for cards <2GB)
2. Install the flashcart's kernel/firmware if required (varies by card)
3. Copy `sprecher_ds.nds` to the microSD root (or a folder)
4. Copy `sn_weights.bin` to the microSD root
   - **Important:** Some filesystems convert this to `SN_WEIGH.BIN` (8.3 format) - the code handles both names

#### For DSi/3DS with CFW

If running via TWiLight Menu++ or similar:
1. Place `sprecher_ds.nds` in your ROMs folder
2. Place `sn_weights.bin` on the SD card root
3. Launch via your homebrew launcher

### Step 3: Run

1. Insert microSD into flashcart
2. Insert flashcart into DS
3. Power on and select `sprecher_ds.nds`
4. Draw and classify!

### Hardware Notes

- **DS Lite** has the best screen quality for the touch interface
- **DSi/3DS** work but may require specific DLDI patches (usually auto-patched by TWiLight Menu++)
- **Original DS (fat)** works fine, touchscreen is slightly less precise
- **Touchscreen calibration** matters - if drawing feels offset, recalibrate in DS system settings

---

## Controls

| Button | Action |
|--------|--------|
| **Stylus** | Draw on bottom screen |
| **A** | Classify drawn digit |
| **B** | Clear canvas |
| **START** | Quit |

---

## File Reference

| File | Description |
|------|-------------|
| `source/sprecher_ds.cpp` | Main inference engine (Q16.16 fixed-point) |
| `Makefile` | devkitPro build configuration |
| `make_sd.py` | Creates FAT16 SD image for melonDS |
| `sn_weights.bin` | Exported network weights (generated) |
| `sd.img` | SD card image for emulator (generated) |
| `sprecher_ds.nds` | Compiled ROM (generated) |

---

## Troubleshooting

### White Screen on Startup

The VBlank interrupt handler may be missing. Ensure `sprecher_ds.cpp` contains:

```cpp
void VblankHandler() { }
// ... in main():
irqSet(IRQ_VBLANK, VblankHandler);
irqEnable(IRQ_VBLANK);
```

### "FAT init FAILED!" (Emulator)

- Verify DLDI is enabled in melonDS settings
- Check that `sd.img` path is correct
- Regenerate with `python make_sd.py`

### "FAT init FAILED!" (Real Hardware)

- Reformat SD card as FAT32
- Try a different SD card (some old cards have compatibility issues)
- Ensure flashcart firmware is properly installed

### "Load FAILED: Cannot open..."

- Verify `sn_weights.bin` exists before running `make_sd.py`
- On real hardware, ensure the file is in the SD card root
- Check filename - some systems convert to `SN_WEIGH.BIN`

### Compilation Errors

- Use the MSYS2 terminal from devkitPro (not regular cmd/PowerShell)
- Verify `DEVKITARM` environment variable is set
- Run `pacman -S nds-dev` if libnds is missing

### Poor Recognition Accuracy

- Train for more epochs (10+ recommended)
- Draw digits **centered** and **large** on the screen
- The active drawing area is the center 192×192 pixels
- Clear strokes work better than sketchy ones

---

## Technical Details

### Fixed-Point Arithmetic

All computation uses **Q16.16 format** (16 integer bits, 16 fractional bits):

```cpp
typedef int32_t fixed;
#define FIX_SHIFT 16
#define FLOAT_TO_FIX(f) ((fixed)((f) * 65536.0f))
#define FIX_MUL(a, b) ((fixed)(((int64_t)(a) * (b)) >> 16))
```

### Saturating Operations

Overflow protection prevents wrap-around errors:

```cpp
static inline fixed fix_add_sat(fixed a, fixed b) {
    int64_t sum = (int64_t)a + b;
    if (sum > INT32_MAX) return INT32_MAX;
    if (sum < INT32_MIN) return INT32_MIN;
    return (fixed)sum;
}
```

### Fast Exp Approximation

Softmax requires exponentiation, implemented via lookup table:

```cpp
// 257-entry table for 2^(k/256), k=0..256
// exp(x) ≈ 2^(x / ln(2))
```

### SNDS v3 Binary Format

The weight file uses a compact binary format:

```
Header: "SNDS" magic + version (3) + architecture info
Per-block: φ knots, φ coeffs, Φ knots, Φ coeffs, λ weights, η, 
           lateral mixing params, residual weights, BatchNorm params
All values: Q16.16 fixed-point (4 bytes each)
```

---

## Building from Scratch

Full rebuild sequence:

```bash
# From project root
python sn_mnist.py --mode train --arch 100,100 --epochs 10
python sn_mnist.py --mode export_nds

# From sprecher_ds/
cd sprecher_ds
python make_sd.py
make clean && make

# Result: sprecher_ds.nds + sd.img ready for melonDS
#    or: sprecher_ds.nds + sn_weights.bin ready for real hardware
```

---

## License

Same as main project - MIT License.