// sprecher_ds.cpp
// Nintendo DS / DS Lite homebrew demo: MNIST digit classification with Sprecher Networks
// Build environment: devkitPro (libnds + libfat)

#include <nds.h>
#include <fat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

// libnds API compatibility: newer libnds removed irqInit().
// Declare it as an optional weak symbol so this code builds against both old and new libnds.
extern "C" void irqInit(void) __attribute__((weak));

// Top screen text console
static PrintConsole topScreen;

// ============================================================
// CRITICAL: VBlank interrupt handler prevents white screen freeze
// ============================================================
void VblankHandler() { }

// ============================================================
// Fixed-point (Q16.16)
// ============================================================

typedef int32_t fix16;
static const int FIX_SHIFT = 16;
static const fix16 FIX_ONE = (fix16)(1 << FIX_SHIFT);

static inline fix16 sat_i64(int64_t v) {
    if (v > (int64_t)INT32_MAX) return (fix16)INT32_MAX;
    if (v < (int64_t)INT32_MIN) return (fix16)INT32_MIN;
    return (fix16)v;
}

static inline fix16 sat_shl32(int32_t v, int n) {
    if (n <= 0) return (fix16)v;
    if (n >= 31) return (v < 0) ? (fix16)INT32_MIN : (fix16)INT32_MAX;
    return sat_i64(((int64_t)v) << n);
}

static inline fix16 fix_add(fix16 a, fix16 b) {
    return sat_i64((int64_t)a + (int64_t)b);
}

static inline fix16 fix_sub(fix16 a, fix16 b) {
    return sat_i64((int64_t)a - (int64_t)b);
}

static inline fix16 fix_mul(fix16 a, fix16 b) {
    int64_t prod = (int64_t)a * (int64_t)b;
    return sat_i64(prod >> FIX_SHIFT);
}

static inline fix16 fix_div(fix16 a, fix16 b) {
    if (b == 0) return (a >= 0) ? (fix16)INT32_MAX : (fix16)INT32_MIN;
    // Avoid left-shifting negative values (undefined behavior). Multiply instead.
    int64_t num = (int64_t)a * (int64_t)FIX_ONE;
    return sat_i64(num / (int64_t)b);
}
// -----------------------------------------------------------------------------
// Fast exp() + softmax helpers (Q16.16).
// We use this only for displaying probabilities on-screen.
// -----------------------------------------------------------------------------

// LUT for 2^(k/256) in Q16.16, k=0..256 (inclusive).
static const uint32_t EXP2_LUT_Q16[257] = {
    65536u, 65714u, 65892u, 66071u, 66250u, 66429u, 66609u, 66790u,
    66971u, 67153u, 67335u, 67517u, 67700u, 67884u, 68068u, 68252u,
    68438u, 68623u, 68809u, 68996u, 69183u, 69370u, 69558u, 69747u,
    69936u, 70126u, 70316u, 70507u, 70698u, 70889u, 71082u, 71274u,
    71468u, 71661u, 71856u, 72050u, 72246u, 72442u, 72638u, 72835u,
    73032u, 73230u, 73429u, 73628u, 73828u, 74028u, 74229u, 74430u,
    74632u, 74834u, 75037u, 75240u, 75444u, 75649u, 75854u, 76060u,
    76266u, 76473u, 76680u, 76888u, 77096u, 77305u, 77515u, 77725u,
    77936u, 78147u, 78359u, 78572u, 78785u, 78998u, 79212u, 79427u,
    79642u, 79858u, 80075u, 80292u, 80510u, 80728u, 80947u, 81166u,
    81386u, 81607u, 81828u, 82050u, 82273u, 82496u, 82719u, 82944u,
    83169u, 83394u, 83620u, 83847u, 84074u, 84302u, 84531u, 84760u,
    84990u, 85220u, 85451u, 85683u, 85915u, 86148u, 86382u, 86616u,
    86851u, 87086u, 87322u, 87559u, 87796u, 88034u, 88273u, 88513u,
    88752u, 88993u, 89234u, 89476u, 89719u, 89962u, 90206u, 90451u,
    90696u, 90942u, 91188u, 91436u, 91684u, 91932u, 92181u, 92431u,
    92682u, 92933u, 93185u, 93438u, 93691u, 93945u, 94200u, 94455u,
    94711u, 94968u, 95226u, 95484u, 95743u, 96002u, 96263u, 96524u,
    96785u, 97048u, 97311u, 97575u, 97839u, 98104u, 98370u, 98637u,
    98905u, 99173u, 99442u, 99711u, 99982u, 100253u, 100524u, 100797u,
    101070u, 101344u, 101619u, 101895u, 102171u, 102448u, 102726u, 103004u,
    103283u, 103564u, 103844u, 104126u, 104408u, 104691u, 104975u, 105260u,
    105545u, 105831u, 106118u, 106406u, 106694u, 106984u, 107274u, 107565u,
    107856u, 108149u, 108442u, 108736u, 109031u, 109326u, 109623u, 109920u,
    110218u, 110517u, 110816u, 111117u, 111418u, 111720u, 112023u, 112327u,
    112631u, 112937u, 113243u, 113550u, 113858u, 114167u, 114476u, 114787u,
    115098u, 115410u, 115723u, 116036u, 116351u, 116667u, 116983u, 117300u,
    117618u, 117937u, 118257u, 118577u, 118899u, 119221u, 119544u, 119869u,
    120194u, 120519u, 120846u, 121174u, 121502u, 121832u, 122162u, 122493u,
    122825u, 123158u, 123492u, 123827u, 124163u, 124500u, 124837u, 125176u,
    125515u, 125855u, 126197u, 126539u, 126882u, 127226u, 127571u, 127917u,
    128263u, 128611u, 128960u, 129310u, 129660u, 130012u, 130364u, 130718u,
    131072u
};

// exp(x) in Q16.16, using exp(x)=2^(x/ln2) with an exp2 LUT.
static inline fix16 exp_fix(fix16 x) {
    // round(1/ln(2) * 65536) = 94548
    const int32_t INV_LN2_Q16 = 94548;
    // y = x / ln(2) in Q16.16
    fix16 y = sat_i64(((int64_t)x * (int64_t)INV_LN2_Q16) >> FIX_SHIFT);
    int32_t n = (int32_t)(y >> FIX_SHIFT);
    uint32_t frac = (uint32_t)(y & (FIX_ONE - 1)); // 0..65535
    uint32_t idx = frac >> 8;                      // 0..255
    uint32_t t   = frac & 0xFF;                    // 0..255

    int32_t v0 = (int32_t)EXP2_LUT_Q16[idx];
    int32_t v1 = (int32_t)EXP2_LUT_Q16[idx + 1];
    int32_t v  = v0 + (int32_t)(((int64_t)(v1 - v0) * (int64_t)t) >> 8);

    // Apply 2^n (n can be negative). Note: v is always positive.
    if (n >= 0) {
        if (n >= 15) return INT32_MAX; // avoid undefined shift / overflow
        return sat_shl32(v, n);
    } else {
        int sh = -n;
        if (sh >= 31) return 0;
        // Round while shifting right.
        return (fix16)((v + (1 << (sh - 1))) >> sh);
    }
}

// softmax(logits) -> probs, all Q16.16. 'tmp_exp' must have length >= n.
static void softmax_fix(const fix16* logits, int n, fix16* probs, fix16* tmp_exp) {
    if (n <= 0) return;
    fix16 maxv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > maxv) maxv = logits[i];
    }

    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        fix16 shifted = logits[i] - maxv; // <= 0
        fix16 e = exp_fix(shifted);
        tmp_exp[i] = e;
        sum += (int64_t)e;
    }
    if (sum <= 0) {
        // Should never happen (max element yields exp(0)=1), but be defensive.
        fix16 inv_n = fix_div(FIX_ONE, (fix16)(n << FIX_SHIFT));
        for (int i = 0; i < n; i++) probs[i] = inv_n;
        return;
    }

    fix16 sum_fix = sat_i64(sum);
    for (int i = 0; i < n; i++) {
        probs[i] = fix_div(tmp_exp[i], sum_fix);
    }
}


static void __attribute__((unused)) iprint_fix16_4(fix16 v) {
    bool neg = (v < 0);
    uint32_t uv = neg ? (uint32_t)(-v) : (uint32_t)v;
    uint32_t ip = (uv >> FIX_SHIFT);

    // frac in [0, 65535]. Convert to 4 decimals with rounding.
    uint32_t frac = (uv & 0xFFFF);
    uint32_t frac4 = (frac * 10000U + 32768U) >> FIX_SHIFT;  // rounded

    if (frac4 >= 10000U) { frac4 = 0; ip++; } // handle carry from rounding

    iprintf("%s%lu.%04lu", neg ? "-" : "", (unsigned long)ip, (unsigned long)frac4);
}


static void fix16_to_str_4(fix16 v, char* out, size_t out_len) {
    // Convert Q16.16 to a human-readable decimal string with 4 fractional digits.
    // Note: Uses integer math only (no floats).
    int64_t vv = (int64_t)v;
    bool neg = (vv < 0);
    uint64_t uv = neg ? (uint64_t)(-vv) : (uint64_t)vv;

    uint32_t ip = (uint32_t)(uv >> FIX_SHIFT);
    uint32_t frac = (uint32_t)(uv & 0xFFFFu);
    uint32_t frac4 = (frac * 10000u + 0x8000u) >> FIX_SHIFT;

    // Carry if rounding pushed us to 10000.
    if (frac4 >= 10000u) {
        frac4 = 0;
        ip++;
    }

    if (neg) snprintf(out, out_len, "-%lu.%04lu", (unsigned long)ip, (unsigned long)frac4);
    else     snprintf(out, out_len,  "%lu.%04lu", (unsigned long)ip, (unsigned long)frac4);
}

static void clear_row(int row) {
    // DS console is 32 cols wide, but printing exactly 32 chars can wrap/scroll.
    // Clear 31 columns to avoid accidental wrap on the last row.
    iprintf("\x1b[%d;0H                               ", row);
}
// ============================================================
// Spline (piecewise-linear)
// ============================================================

typedef struct {
    uint32_t n_knots;
    fix16* knots;
    fix16* coeffs;
    bool monotonic;
    fix16 x0, xN, dx, inv_dx, slope0, slopeN;
    bool uniform;
} Spline;

static void prepare_spline(Spline* sp) {
    if (!sp) return;

    sp->uniform = false;
    sp->x0 = sp->xN = sp->dx = sp->inv_dx = sp->slope0 = sp->slopeN = 0;

    if (sp->n_knots < 2 || !sp->knots || !sp->coeffs) return;

    sp->x0 = sp->knots[0];
    sp->xN = sp->knots[sp->n_knots - 1];

    // Boundary slopes for Φ's linear extrapolation (φ uses constant extension).
    fix16 dx0 = fix_sub(sp->knots[1], sp->knots[0]);
    fix16 dxN = fix_sub(sp->knots[sp->n_knots - 1], sp->knots[sp->n_knots - 2]);

    if (dx0 != 0) {
        sp->slope0 = fix_div(fix_sub(sp->coeffs[1], sp->coeffs[0]), dx0);
    }
    if (dxN != 0) {
        sp->slopeN = fix_div(
            fix_sub(sp->coeffs[sp->n_knots - 1], sp->coeffs[sp->n_knots - 2]),
            dxN
        );
    }

    // Detect uniform knot spacing *after* Q16.16 quantization.
    // The Python exporter uses linspace, but Q16.16 rounding can make a few steps differ by 1 LSB.
    sp->dx = dx0;
    if (sp->dx == 0) return;

    bool uniform = true;
    for (uint32_t k = 1; k + 1 < sp->n_knots; k++) {
        if (fix_sub(sp->knots[k + 1], sp->knots[k]) != sp->dx) {
            uniform = false;
            break;
        }
    }

    if (uniform) {
        sp->inv_dx = fix_div(FIX_ONE, sp->dx);
        sp->uniform = (sp->inv_dx != 0);
    }
}

static fix16 spline_eval(const Spline* sp, fix16 x) {
    if (!sp || sp->n_knots < 2) return 0;

    if (x < sp->x0) {
        if (sp->monotonic) return 0;
        return fix_add(sp->coeffs[0], fix_mul(sp->slope0, fix_sub(x, sp->x0)));
    }

    if (x > sp->xN) {
        if (sp->monotonic) return FIX_ONE;
        return fix_add(sp->coeffs[sp->n_knots - 1], fix_mul(sp->slopeN, fix_sub(x, sp->xN)));
    }

    if (sp->uniform) {
        fix16 t_all = fix_mul(fix_sub(x, sp->x0), sp->inv_dx);
        int idx = (int)(t_all >> FIX_SHIFT);
        if (idx < 0) idx = 0;
        if (idx >= (int)sp->n_knots - 1) idx = (int)sp->n_knots - 2;

        int64_t x_base = (int64_t)sp->x0 + (int64_t)idx * (int64_t)sp->dx;
        fix16 local_t = fix_mul(fix_sub(x, (fix16)x_base), sp->inv_dx);
        fix16 y0 = sp->coeffs[idx];
        fix16 y1 = sp->coeffs[idx + 1];
        return fix_add(y0, fix_mul(local_t, fix_sub(y1, y0)));
    }

    // Binary search fallback
    int lo = 0, hi = (int)sp->n_knots - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (x < sp->knots[mid]) hi = mid;
        else lo = mid;
    }
    fix16 denom = fix_sub(sp->knots[hi], sp->knots[lo]);
    fix16 t = (denom != 0) ? fix_div(fix_sub(x, sp->knots[lo]), denom) : 0;
    return fix_add(sp->coeffs[lo], fix_mul(t, fix_sub(sp->coeffs[hi], sp->coeffs[lo])));
}

// ============================================================
// Sprecher Network structures
// ============================================================

typedef enum { RES_NONE=0, RES_SCALAR=1, RES_POOL=2, RES_BROADCAST=3, RES_PROJ=4 } ResidualType;

typedef struct {
    uint32_t d_in, d_out;
    fix16 eta;
    Spline phi, Phi;
    fix16* lambdas;
    bool has_lateral;
    fix16 lateral_scale;
    fix16* lateral_w_fwd;
    fix16* lateral_w_bwd;
    ResidualType residual_type;
    fix16 residual_scalar;
    fix16* residual_pool;
    fix16* residual_bcast;
    uint32_t norm_dim;
    fix16* norm_scale;
    fix16* norm_bias;
} Block;

typedef struct {
    uint32_t version, flags, num_blocks, input_dim, output_dim, total_params, max_dim;
    fix16 q_factor;
    bool lateral_bidir, norm_before;
    uint32_t norm_type;
    bool norm_skip_first;
    Block* blocks;
    fix16 output_scale, output_bias;
    fix16* all_memory;
    uint32_t all_memory_count;
    fix16* work_a;
    fix16* work_b;
    fix16* scratch;
} SprecherNet;

static const uint32_t FLAG_USE_LATERAL     = 1u << 0;
static const uint32_t FLAG_LATERAL_BIDIR   = 1u << 1;
static const uint32_t FLAG_USE_RESIDUAL    = 1u << 2;
static const uint32_t FLAG_RESIDUAL_LINEAR = 1u << 3;
static const uint32_t FLAG_NORM_BEFORE     = 1u << 6;
static const uint32_t FLAG_NORM_SKIP_FIRST = 1u << 7;

static const uint32_t BF_HAS_LATERAL = 1u << 0;
static const uint32_t BF_HAS_NORM    = 1u << 1;

// ============================================================
// File IO helpers
// ============================================================

static bool read_u32(FILE* f, uint32_t* out) { return fread(out, 4, 1, f) == 1; }
static bool read_i32(FILE* f, int32_t* out) { return fread(out, 4, 1, f) == 1; }

static void free_net(SprecherNet* net) {
    if (!net) return;
    if (net->blocks) free(net->blocks);
    if (net->all_memory) free(net->all_memory);
    memset(net, 0, sizeof(SprecherNet));
}

// ============================================================
// Normalization
// ============================================================

static void apply_norm_inplace(const Block* blk, fix16* vec) {
    if (!blk || blk->norm_dim == 0) return;
    for (uint32_t i = 0; i < blk->norm_dim; i++) {
        vec[i] = fix_add(fix_mul(vec[i], blk->norm_scale[i]), blk->norm_bias[i]);
    }
}

// ============================================================
// Forward pass
// ============================================================

static void forward_block(const SprecherNet* net, const Block* blk, const fix16* x, fix16* y) {
    const uint32_t d_in = blk->d_in;
    const uint32_t d_out = blk->d_out;
    fix16* scratch = net->scratch;

    // 1) Compute unmixed pre-activations s_q sequentially (memory-efficient; no d_in*d_out tensor)
    for (uint32_t q = 0; q < d_out; q++) {
        fix16 q_fix = (fix16)((int32_t)q << FIX_SHIFT);

        // eta_q = eta * q
        fix16 eta_q = sat_i64(((int64_t)blk->eta * (int64_t)q_fix) >> FIX_SHIFT);

        int64_t acc = 0;
        for (uint32_t i = 0; i < d_in; i++) {
            fix16 phi_val = spline_eval(&blk->phi, fix_add(x[i], eta_q));

            // Accumulate in 64-bit to avoid per-term saturation artifacts:
            // term = (lambda_i * phi_val) >> 16  (still Q16.16)
            acc += ((int64_t)blk->lambdas[i] * (int64_t)phi_val) >> FIX_SHIFT;
        }
        // + q_factor * q
        acc += ((int64_t)net->q_factor * (int64_t)q_fix) >> FIX_SHIFT;

        y[q] = sat_i64(acc);
    }

    // 2) Optional lateral mixing on s (cyclic neighbors; O(d_out) params)
    if (blk->has_lateral && d_out > 1) {
        if (net->lateral_bidir) {
            for (uint32_t q = 0; q < d_out; q++) {
                uint32_t q_next = (q + 1u == d_out) ? 0u : (q + 1u);
                uint32_t q_prev = (q == 0u) ? (d_out - 1u) : (q - 1u);

                fix16 sum = fix_add(
                    fix_mul(blk->lateral_w_fwd[q], y[q_next]),
                    fix_mul(blk->lateral_w_bwd[q], y[q_prev])
                );
                scratch[q] = fix_add(y[q], fix_mul(blk->lateral_scale, sum));
            }
        } else {
            for (uint32_t q = 0; q < d_out; q++) {
                uint32_t q_next = (q + 1u == d_out) ? 0u : (q + 1u);
                scratch[q] = fix_add(
                    y[q],
                    fix_mul(blk->lateral_scale, fix_mul(blk->lateral_w_fwd[q], y[q_next]))
                );
            }
        }
        memcpy(y, scratch, d_out * sizeof(fix16));
    }

    // 3) Apply Φ spline
    for (uint32_t q = 0; q < d_out; q++) {
        y[q] = spline_eval(&blk->Phi, y[q]);
    }

    // 4) Add node-wise (cyclic) residuals (O(max(d_in,d_out)) params)
    switch (blk->residual_type) {
        case RES_SCALAR:
            for (uint32_t q = 0; q < d_out; q++)
                y[q] = fix_add(y[q], fix_mul(blk->residual_scalar, x[q]));
            break;
        case RES_POOL:
            for (uint32_t i = 0; i < d_in; i++) {
                uint32_t q = (d_out != 0) ? (i % d_out) : 0;
                y[q] = fix_add(y[q], fix_mul(blk->residual_pool[i], x[i]));
            }
            break;
        case RES_BROADCAST:
            for (uint32_t q = 0; q < d_out; q++) {
                uint32_t src = (d_in != 0) ? (q % d_in) : 0;
                y[q] = fix_add(y[q], fix_mul(blk->residual_bcast[q], x[src]));
            }
            break;
        default: break;
    }
}

static void net_forward(const SprecherNet* net, const fix16* input, fix16* output, bool show_progress) {
    if (!net || !input || !output) return;
    fix16* a = net->work_a;
    fix16* b = net->work_b;
    uint32_t cur_dim = net->input_dim;
    memcpy(a, input, cur_dim * sizeof(fix16));

    for (uint32_t bi = 0; bi < net->num_blocks; bi++) {
        const Block* blk = &net->blocks[bi];
        if (show_progress) {
            clear_row(16);
            iprintf("\x1b[16;0HLayer %lu/%lu", (unsigned long)(bi+1), (unsigned long)net->num_blocks);
        }
        if (blk->d_in != cur_dim) break;
        bool do_norm = (blk->norm_dim > 0) && !(net->norm_skip_first && bi == 0);
        if (net->norm_before && do_norm) apply_norm_inplace(blk, a);
        forward_block(net, blk, a, b);
        if (!net->norm_before && do_norm) apply_norm_inplace(blk, b);
        fix16* tmp = a; a = b; b = tmp;
        cur_dim = blk->d_out;
    }

    for (uint32_t i = 0; i < net->output_dim; i++)
        output[i] = fix_add(fix_mul(a[i], net->output_scale), net->output_bias);
}

// ============================================================
// Loading SNDS v3 weights
// ============================================================

static bool load_net_v3(SprecherNet* net, const char* filename, char* err, size_t errlen) {
    FILE* f = fopen(filename, "rb");
    if (!f) { snprintf(err, errlen, "Cannot open %s", filename); return false; }

    // Ensure net starts clean in case caller skipped load_net()
    free_net(net);

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SNDS", 4) != 0) {
        snprintf(err, errlen, "Bad magic"); fclose(f); return false;
    }

    uint32_t version = 0, fix_shift = 0;
    if (!read_u32(f, &version) || !read_u32(f, &fix_shift) || version != 3 || fix_shift != 16) {
        snprintf(err, errlen, "Bad version/shift"); fclose(f); return false;
    }

    uint32_t flags = 0, num_blocks = 0, input_dim = 0, output_dim = 0, total_params = 0, max_dim = 0;
    int32_t q_factor_i32 = 0;
    if (!read_u32(f, &flags) || !read_u32(f, &num_blocks) || !read_u32(f, &input_dim) ||
        !read_u32(f, &output_dim) || !read_u32(f, &total_params) || !read_u32(f, &max_dim) ||
        !read_i32(f, &q_factor_i32)) {
        snprintf(err, errlen, "Header failed"); fclose(f); return false;
    }

    if (num_blocks == 0 || num_blocks > 128) {
        snprintf(err, errlen, "Bad num_blocks"); fclose(f); return false;
    }
    if (input_dim == 0 || output_dim == 0 || max_dim == 0 || max_dim > 65535) {
        snprintf(err, errlen, "Bad dims"); fclose(f); return false;
    }

    const uint32_t norm_type = (flags >> 4) & 3u;
    const bool use_lateral = (flags & FLAG_USE_LATERAL) != 0;
    const bool lateral_bidir = (flags & FLAG_LATERAL_BIDIR) != 0;
    const bool norm_before = (flags & FLAG_NORM_BEFORE) != 0;
    const bool norm_skip_first = (flags & FLAG_NORM_SKIP_FIRST) != 0;

    (void)norm_type; // exported as folded affine; stored for debugging only

    net->blocks = (Block*)calloc(num_blocks, sizeof(Block));
    if (!net->blocks) { snprintf(err, errlen, "OOM blocks"); fclose(f); return false; }

    // ------------------------------------------------------------------
    // First pass: count required weight memory, and validate structure.
    // ------------------------------------------------------------------
    const long blocks_start = ftell(f);
    uint32_t weight_fix16s = 0;

    for (uint32_t bi = 0; bi < num_blocks; bi++) {
        uint32_t d_in = 0, d_out = 0, phi_nk = 0, Phi_nk = 0, block_flags = 0, residual_type = 0, norm_dim = 0;
        int32_t eta_i32 = 0;

        if (!read_u32(f, &d_in) || !read_u32(f, &d_out) || !read_u32(f, &phi_nk) || !read_u32(f, &Phi_nk) ||
            !read_u32(f, &block_flags) || !read_u32(f, &residual_type) || !read_u32(f, &norm_dim) || !read_i32(f, &eta_i32)) {
            snprintf(err, errlen, "Unexpected EOF in block header"); fclose(f); free_net(net); return false;
        }

        if (d_in == 0 || d_out == 0 || d_in > max_dim || d_out > max_dim) {
            snprintf(err, errlen, "Bad block dims"); fclose(f); free_net(net); return false;
        }
        if (phi_nk < 2 || Phi_nk < 2 || phi_nk > 4096 || Phi_nk > 4096) {
            snprintf(err, errlen, "Bad knot counts"); fclose(f); free_net(net); return false;
        }

        const bool has_lateral = (block_flags & BF_HAS_LATERAL) != 0;
        if (has_lateral && !use_lateral) {
            snprintf(err, errlen, "File inconsistent: block has lateral but header says lateral off"); fclose(f); free_net(net); return false;
        }

        // Core arrays: phi knots+coeffs, Phi knots+coeffs, lambdas (vector!)
        uint32_t skip_words = phi_nk * 2u + Phi_nk * 2u + d_in;
        weight_fix16s += skip_words;

        // Lateral arrays
        if (has_lateral) {
            uint32_t lat_words = 1u + (lateral_bidir ? (d_out * 2u) : d_out); // scale + weights
            skip_words += lat_words;
            weight_fix16s += lat_words;
        }

        // Residual arrays
        if (residual_type == RES_NONE) {
            // nothing
        } else if (residual_type == RES_SCALAR) {
            skip_words += 1u;
            weight_fix16s += 1u;
        } else if (residual_type == RES_POOL) {
            skip_words += d_in;
            weight_fix16s += d_in;
        } else if (residual_type == RES_BROADCAST) {
            skip_words += d_out;
            weight_fix16s += d_out;
        } else {
            snprintf(err, errlen, "Unsupported residual type %lu", (unsigned long)residual_type);
            fclose(f); free_net(net); return false;
        }

        // Norm arrays (pre-folded BN affine): scale + bias
        if (norm_dim > 0) {
            if (norm_dim > max_dim) {
                snprintf(err, errlen, "Bad norm_dim"); fclose(f); free_net(net); return false;
            }
            skip_words += norm_dim * 2u;
            weight_fix16s += norm_dim * 2u;
        }

        if (fseek(f, (long)(skip_words * 4u), SEEK_CUR) != 0) {
            snprintf(err, errlen, "Seek failed"); fclose(f); free_net(net); return false;
        }
    }

    // Allocate a single contiguous buffer for all weights + working buffers.
    const uint32_t total_fix16s = weight_fix16s + 3u * max_dim;
    net->all_memory = (fix16*)malloc((size_t)total_fix16s * sizeof(fix16));
    if (!net->all_memory) { snprintf(err, errlen, "OOM weights"); fclose(f); free_net(net); return false; }
    net->all_memory_count = total_fix16s;

    // ------------------------------------------------------------------
    // Second pass: load weights into the contiguous buffer
    // ------------------------------------------------------------------
    if (fseek(f, blocks_start, SEEK_SET) != 0) {
        snprintf(err, errlen, "Seek failed"); fclose(f); free_net(net); return false;
    }

    fix16* mem = net->all_memory;

    for (uint32_t bi = 0; bi < num_blocks; bi++) {
        uint32_t d_in = 0, d_out = 0, phi_nk = 0, Phi_nk = 0, block_flags = 0, res_type = 0, norm_dim = 0;
        int32_t eta_i32 = 0;

        if (!read_u32(f, &d_in) || !read_u32(f, &d_out) || !read_u32(f, &phi_nk) || !read_u32(f, &Phi_nk) ||
            !read_u32(f, &block_flags) || !read_u32(f, &res_type) || !read_u32(f, &norm_dim) || !read_i32(f, &eta_i32)) {
            snprintf(err, errlen, "Unexpected EOF in block header"); fclose(f); free_net(net); return false;
        }

        Block* blk = &net->blocks[bi];
        blk->d_in = d_in;
        blk->d_out = d_out;
        blk->eta = (fix16)eta_i32;

        blk->has_lateral = (block_flags & BF_HAS_LATERAL) != 0;
        blk->residual_type = (ResidualType)res_type;
        blk->norm_dim = norm_dim;

        // φ (monotone, constant extension outside domain)
        blk->phi.n_knots = phi_nk;
        blk->phi.monotonic = true;
        blk->phi.knots = mem;
        if (fread(mem, 4, phi_nk, f) != phi_nk) { snprintf(err, errlen, "EOF reading phi knots"); fclose(f); free_net(net); return false; }
        mem += phi_nk;
        blk->phi.coeffs = mem;
        if (fread(mem, 4, phi_nk, f) != phi_nk) { snprintf(err, errlen, "EOF reading phi coeffs"); fclose(f); free_net(net); return false; }
        mem += phi_nk;
        prepare_spline(&blk->phi);

        // Φ (general, linear extrapolation outside domain)
        blk->Phi.n_knots = Phi_nk;
        blk->Phi.monotonic = false;
        blk->Phi.knots = mem;
        if (fread(mem, 4, Phi_nk, f) != Phi_nk) { snprintf(err, errlen, "EOF reading Phi knots"); fclose(f); free_net(net); return false; }
        mem += Phi_nk;
        blk->Phi.coeffs = mem;
        if (fread(mem, 4, Phi_nk, f) != Phi_nk) { snprintf(err, errlen, "EOF reading Phi coeffs"); fclose(f); free_net(net); return false; }
        mem += Phi_nk;
        prepare_spline(&blk->Phi);

        // λ vector (one weight per input dim; shared across all q)
        blk->lambdas = mem;
        if (fread(mem, 4, d_in, f) != d_in) { snprintf(err, errlen, "EOF reading lambdas"); fclose(f); free_net(net); return false; }
        mem += d_in;

        // Lateral mixing (cyclic neighbors)
        if (blk->has_lateral) {
            if (d_out <= 1) {
                // Shouldn't happen (exporter only adds lateral when d_out>1), but keep robust.
                blk->has_lateral = false;
            } else {
                int32_t v = 0;
                if (!read_i32(f, &v)) { snprintf(err, errlen, "EOF reading lateral scale"); fclose(f); free_net(net); return false; }
                blk->lateral_scale = (fix16)v;

                blk->lateral_w_fwd = mem;
                if (fread(mem, 4, d_out, f) != d_out) { snprintf(err, errlen, "EOF reading lateral weights"); fclose(f); free_net(net); return false; }
                mem += d_out;

                if (lateral_bidir) {
                    blk->lateral_w_bwd = mem;
                    if (fread(mem, 4, d_out, f) != d_out) { snprintf(err, errlen, "EOF reading lateral weights (bwd)"); fclose(f); free_net(net); return false; }
                    mem += d_out;
                }
            }
        }

        // Residual params
        if (blk->residual_type == RES_SCALAR) {
            int32_t v = 0;
            if (!read_i32(f, &v)) { snprintf(err, errlen, "EOF reading residual scalar"); fclose(f); free_net(net); return false; }
            blk->residual_scalar = (fix16)v;
        } else if (blk->residual_type == RES_POOL) {
            blk->residual_pool = mem;
            if (fread(mem, 4, d_in, f) != d_in) { snprintf(err, errlen, "EOF reading residual pool"); fclose(f); free_net(net); return false; }
            mem += d_in;
        } else if (blk->residual_type == RES_BROADCAST) {
            blk->residual_bcast = mem;
            if (fread(mem, 4, d_out, f) != d_out) { snprintf(err, errlen, "EOF reading residual bcast"); fclose(f); free_net(net); return false; }
            mem += d_out;
        }

        // Norm params (folded affine)
        if (norm_dim > 0) {
            blk->norm_scale = mem;
            if (fread(mem, 4, norm_dim, f) != norm_dim) { snprintf(err, errlen, "EOF reading norm scale"); fclose(f); free_net(net); return false; }
            mem += norm_dim;

            blk->norm_bias = mem;
            if (fread(mem, 4, norm_dim, f) != norm_dim) { snprintf(err, errlen, "EOF reading norm bias"); fclose(f); free_net(net); return false; }
            mem += norm_dim;
        }
    }

    int32_t out_scale_i32 = 0, out_bias_i32 = 0;
    if (!read_i32(f, &out_scale_i32) || !read_i32(f, &out_bias_i32)) {
        snprintf(err, errlen, "EOF reading footer"); fclose(f); free_net(net); return false;
    }
    fclose(f);

    // Populate net header fields
    net->version = version;
    net->flags = flags;
    net->num_blocks = num_blocks;
    net->input_dim = input_dim;
    net->output_dim = output_dim;
    net->total_params = total_params;
    net->max_dim = max_dim;
    net->q_factor = (fix16)q_factor_i32;
    net->lateral_bidir = lateral_bidir;
    net->norm_before = norm_before;
    net->norm_type = norm_type;
    net->norm_skip_first = norm_skip_first;
    net->output_scale = (fix16)out_scale_i32;
    net->output_bias = (fix16)out_bias_i32;

    // Working buffers (3 * max_dim fix16s)
    net->work_a = mem; mem += max_dim;
    net->work_b = mem; mem += max_dim;
    net->scratch = mem;

    return true;
}

static bool load_net(SprecherNet* net, const char* filename, char* err, size_t errlen) {
    memset(net, 0, sizeof(SprecherNet));
    FILE* f = fopen(filename, "rb");
    if (!f) { snprintf(err, errlen, "Cannot open %s", filename); return false; }
    char magic[4]; uint32_t version;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "SNDS", 4) == 0 && read_u32(f, &version);
    fclose(f);
    if (!ok) { snprintf(err, errlen, "Invalid file"); return false; }
    if (version == 3) return load_net_v3(net, filename, err, errlen);
    snprintf(err, errlen, "Need v3, got v%lu", (unsigned long)version);
    return false;
}

// ============================================================
// Drawing canvas
// ============================================================

static const int CANVAS_W = 256;
static const int CANVAS_H = 192;
static uint8_t g_canvas[CANVAS_W * CANVAS_H];

static inline uint16_t gray555(uint8_t v) {
    int c = v >> 3;
    return (uint16_t)(RGB15(c, c, c) | BIT(15));
}

static void clear_canvas(u16* gfx) {
    memset(g_canvas, 0, sizeof(g_canvas));
    for (int i = 0; i < 256 * 192; i++) gfx[i] = gray555(0);
}

static void draw_dot(u16* gfx, int x, int y, int radius, uint8_t intensity) {
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int xx = x + dx, yy = y + dy;
            if (xx >= 0 && xx < CANVAS_W && yy >= 0 && yy < CANVAS_H && dx*dx + dy*dy <= radius*radius) {
                g_canvas[yy * CANVAS_W + xx] = intensity;
                gfx[yy * 256 + xx] = gray555(intensity);
            }
        }
    }
}

static void draw_line(u16* gfx, int x0, int y0, int x1, int y1, int radius, uint8_t intensity) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        draw_dot(gfx, x0, y0, radius, intensity);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ============================================================
// Downsample to 28x28
// ============================================================

static void downsample_28x28(uint8_t out[28*28]) {
    const int ROI_X0 = 32, ROI_Y0 = 0, ROI_SIZE = 192;
    for (int y = 0; y < 28; y++) {
        int y0 = ROI_Y0 + (y * ROI_SIZE) / 28;
        int y1 = ROI_Y0 + ((y + 1) * ROI_SIZE) / 28;
        for (int x = 0; x < 28; x++) {
            int x0 = ROI_X0 + (x * ROI_SIZE) / 28;
            int x1 = ROI_X0 + ((x + 1) * ROI_SIZE) / 28;
            int sum = 0, count = 0;
            for (int yy = y0; yy < y1; yy++)
                for (int xx = x0; xx < x1; xx++) { sum += g_canvas[yy * CANVAS_W + xx]; count++; }
            out[y * 28 + x] = count > 0 ? (uint8_t)((sum + (count>>1)) / count) : 0;
        }
    }
}
// MNIST-like preprocessing: crop to bounding box, scale to 20x20 preserving aspect,
// center in 28x28, then shift by center-of-mass (similar to common MNIST drawing preprocess).
static bool preprocess_mnist_like_28(const uint8_t in28[28*28], uint8_t out28[28*28]) {
    const int THR = 20;          // ignore very faint pixels
    const int TARGET = 20;       // MNIST uses ~20x20 glyph in 28x28 canvas

    int min_x = 28, min_y = 28, max_x = -1, max_y = -1;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            uint8_t v = in28[y*28 + x];
            if (v > THR) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < 0) {
        // No ink.
        memset(out28, 0, 28*28);
        return false;
    }

    int w = (max_x - min_x + 1);
    int h = (max_y - min_y + 1);
    int max_side = (w > h) ? w : h;

    int new_w = (w * TARGET + max_side/2) / max_side;
    int new_h = (h * TARGET + max_side/2) / max_side;
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;
    if (new_w > TARGET) new_w = TARGET;
    if (new_h > TARGET) new_h = TARGET;

    // Resample cropped box -> new_w x new_h using area averaging.
    uint8_t tmp[TARGET * TARGET];
    memset(tmp, 0, sizeof(tmp));

    for (int dy = 0; dy < new_h; dy++) {
        int sy0 = min_y + (dy * h) / new_h;
        int sy1 = min_y + ((dy + 1) * h) / new_h;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        if (sy1 > (max_y + 1)) sy1 = (max_y + 1);

        for (int dx = 0; dx < new_w; dx++) {
            int sx0 = min_x + (dx * w) / new_w;
            int sx1 = min_x + ((dx + 1) * w) / new_w;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > (max_x + 1)) sx1 = (max_x + 1);

            int sum = 0;
            int cnt = 0;
            for (int sy = sy0; sy < sy1; sy++) {
                for (int sx = sx0; sx < sx1; sx++) {
                    sum += in28[sy*28 + sx];
                    cnt++;
                }
            }
            tmp[dy*TARGET + dx] = (uint8_t)(sum / (cnt ? cnt : 1));
        }
    }

    // Center the resized glyph inside 28x28.
    memset(out28, 0, 28*28);
    int off_x = (28 - new_w) / 2;
    int off_y = (28 - new_h) / 2;
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            out28[(off_y + y)*28 + (off_x + x)] = tmp[y*TARGET + x];
        }
    }

    // Center-of-mass shift to (14,14).
    int64_t sum = 0;
    int64_t mx = 0;
    int64_t my = 0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int v = (int)out28[y*28 + x];
            sum += v;
            mx += (int64_t)x * v;
            my += (int64_t)y * v;
        }
    }
    if (sum > 0) {
        int cx = (int)(mx / sum);
        int cy = (int)(my / sum);
        int dx = 14 - cx;
        int dy = 14 - cy;

        uint8_t shifted[28*28];
        memset(shifted, 0, 28*28);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int nx = x + dx;
                int ny = y + dy;
                if ((unsigned)nx < 28U && (unsigned)ny < 28U) {
                    shifted[ny*28 + nx] = out28[y*28 + x];
                }
            }
        }
        memcpy(out28, shifted, 28*28);
    }

    return true;
}



static void pixels_to_input(const uint8_t pix[28*28], fix16 out[28*28]) {
    // Convert 0..255 grayscale into Q16.16 in [0,1] with rounding.
    for (int i = 0; i < 28*28; i++) {
        uint32_t num = (uint32_t)pix[i] * (uint32_t)FIX_ONE + 127u;
        out[i] = (fix16)(num / 255u);
    }
}

// ============================================================
// Main
// ============================================================

int main(void) {
    // IRQ init (compat): call irqInit() if the symbol exists (older libnds),
    // otherwise skip it (newer libnds initializes IRQs before main).
    if (irqInit) irqInit();
    irqSet(IRQ_VBLANK, VblankHandler);
    irqEnable(IRQ_VBLANK);

    powerOn(POWER_ALL_2D);

    // Make sure the main 2D engine (A) is on the top screen and the sub engine (B) is on the bottom.
    lcdMainOnTop();

    // ----------------------------
    // Top screen: text console (MAIN)
    // ----------------------------
    videoSetMode(MODE_0_2D);
    vramSetBankA(VRAM_A_MAIN_BG);

    consoleInit(&topScreen, 0, BgType_Text4bpp, BgSize_T_256x256, 31, 0, true, true);
    consoleSelect(&topScreen);
    consoleClear();

    // ----------------------------
    // Bottom screen: 16-bit bitmap for drawing (SUB)
    // ----------------------------
    videoSetModeSub(MODE_5_2D);
    vramSetBankC(VRAM_C_SUB_BG);
    int bg = bgInitSub(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0);
    u16* gfx = (u16*)bgGetGfxPtr(bg);

    clear_canvas(gfx);

    iprintf("\nSprecher Network on Nintendo DS\n");
    iprintf("===============================\n\n");
    iprintf(" Stylus: draw on bottom screen\n");
    iprintf(" A: classify digit\n");
    iprintf(" B: clear canvas\n");
    iprintf(" START: quit\n\n");

    // Init FAT for SD card access
    iprintf(" Initializing FAT...\n");
    if (!fatInitDefault()) {
        iprintf("\x1b[31m FAT init FAILED!\x1b[39m\n");
        iprintf(" Check DLDI / SD card setup.\n");
        iprintf("\n Press START to exit.\n");
        while (1) {
            swiWaitForVBlank();
            scanKeys();
            if (keysDown() & KEY_START) return 0;
        }
    }
    iprintf("\x1b[32m FAT OK.\x1b[39m\n");

    // Load weights - try both filenames for compatibility
    SprecherNet net;
    char err[128];
    bool net_ok = load_net(&net, "sn_weights.bin", err, sizeof(err));
    if (!net_ok) {
        // Try 8.3 filename for FAT16 compatibility
        net_ok = load_net(&net, "SN_WEIGH.BIN", err, sizeof(err));
    }

    fix16* logits_buf = NULL;
    fix16* baseline_logits = NULL;
    fix16* probs_buf = NULL;
    bool have_baseline = false;

    if (!net_ok) {
        iprintf("\x1b[31m Load FAILED:\x1b[39m %s\n", err);
        iprintf("\n Put weights file on SD root.\n");
    } else {
        uint64_t mlp_params = 0;
        for (uint32_t bi = 0; bi < net.num_blocks; bi++) {
            mlp_params += (uint64_t)net.blocks[bi].d_in * net.blocks[bi].d_out + net.blocks[bi].d_out;
        }

        iprintf("\n\x1b[32m Loaded SNDS v%lu\x1b[39m\n", (unsigned long)net.version);
        iprintf(" Blocks: %lu\n", (unsigned long)net.num_blocks);

        // Print architecture with run-length encoding for consecutive identical dimensions
        iprintf(" Arch: %lu", (unsigned long)net.input_dim);
        {
            uint32_t prev_dim = 0;
            int run_count = 0;
            for (uint32_t bi = 0; bi < net.num_blocks; bi++) {
                uint32_t d_out = net.blocks[bi].d_out;
                if (bi == 0) {
                    prev_dim = d_out;
                    run_count = 1;
                } else if (d_out == prev_dim) {
                    run_count++;
                } else {
                    // Output the previous run
                    if (run_count > 1) {
                        iprintf("->%lu^%d", (unsigned long)prev_dim, run_count);
                    } else {
                        iprintf("->%lu", (unsigned long)prev_dim);
                    }
                    prev_dim = d_out;
                    run_count = 1;
                }
            }
            // Output the final run
            if (run_count > 0) {
                if (run_count > 1) {
                    iprintf("->%lu^%d", (unsigned long)prev_dim, run_count);
                } else {
                    iprintf("->%lu", (unsigned long)prev_dim);
                }
            }
        }
        iprintf("\n");

        iprintf(" SN params: %lu  (MLP: %llu)\n", (unsigned long)net.total_params, (unsigned long long)mlp_params);

        iprintf("\n Ready! Draw a digit, press A.\n");

        if (net.output_dim < 1 || net.output_dim > 64) {
            iprintf("\x1b[31m Bad output_dim: %lu\x1b[39m\n", (unsigned long)net.output_dim);
            net_ok = false;
        } else {
            const unsigned long need = (unsigned long)(net.output_dim * 3 * sizeof(fix16));
            logits_buf = (fix16*)malloc(net.output_dim * sizeof(fix16));
            baseline_logits = (fix16*)malloc(net.output_dim * sizeof(fix16));
            probs_buf = (fix16*)malloc(net.output_dim * sizeof(fix16));
            if (!logits_buf || !baseline_logits || !probs_buf) {
                iprintf("\x1b[31m OOM\x1b[39m (need %lu B)\n", need);
                net_ok = false;
            } else {
                // Compute blank baseline logits (all-zero input) once.
                memset(net.work_a, 0, net.input_dim * sizeof(fix16));
                net_forward(&net, net.work_a, baseline_logits, true);
                have_baseline = true;
                iprintf(" Calib: blank-baseline ON\n");
            }
        }
    }

    touchPosition touch;
    bool drawing = false;
    int last_x = 0, last_y = 0;

    while (1) {
        swiWaitForVBlank();
        scanKeys();
        int down = keysDown();
        int held = keysHeld();

        if (down & KEY_START) break;

        if (down & KEY_B) {
            clear_canvas(gfx);
            iprintf("\x1b[18;0H Canvas cleared.             \n");
        }

        if (held & KEY_TOUCH) {
            touchRead(&touch);
            int x = touch.px, y = touch.py;
            if (!drawing) { drawing = true; last_x = x; last_y = y; }
            draw_line(gfx, last_x, last_y, x, y, 6, 255);
            last_x = x; last_y = y;
        } else {
            drawing = false;
        }


        if ((down & KEY_A) && net_ok && logits_buf) {
            // Clear result area (rows 16..23) so old characters don't linger.
            for (int r = 16; r <= 23; r++) clear_row(r);

            iprintf("\x1b[17;0HRunning inference...");

            uint8_t pix28[28*28];
            uint8_t proc28[28*28];
            fix16 input[28*28];

            downsample_28x28(pix28);
            preprocess_mnist_like_28(pix28, proc28);
            pixels_to_input(proc28, input);

            memset(logits_buf, 0, net.output_dim * sizeof(fix16));
            // Forward pass (raw logits).
            net_forward(&net, input, logits_buf, true);

            // Raw argmax (useful for debugging).
            int raw_pred = 0;
            fix16 raw_best = logits_buf[0];
            for (uint32_t i = 1; i < net.output_dim; i++) {
                if (logits_buf[i] > raw_best) {
                    raw_best = logits_buf[i];
                    raw_pred = (int)i;
                }
            }

            // Optional: blank-baseline calibration (removes constant class bias).
            bool used_calib = have_baseline && (baseline_logits != NULL);
            if (used_calib) {
                for (uint32_t i = 0; i < net.output_dim; i++) {
                    logits_buf[i] = fix_sub(logits_buf[i], baseline_logits[i]);
                }
            }

            // Argmax on (possibly calibrated) logits.
            int pred = 0;
            fix16 best = logits_buf[0];
            for (uint32_t i = 1; i < net.output_dim; i++) {
                if (logits_buf[i] > best) {
                    best = logits_buf[i];
                    pred = (int)i;
                }
            }

            // Convert logits -> probabilities for display.
            if (probs_buf) {
                softmax_fix(logits_buf, (int)net.output_dim, probs_buf, net.scratch);
            }
            // Clear status + progress lines.
            clear_row(16);
            clear_row(17);

            clear_row(18);
            if (used_calib) {
                iprintf("\x1b[18;0HPREDICTED: %d (raw %d)", pred, raw_pred);
            } else {
                iprintf("\x1b[18;0HPREDICTED: %d", pred);
            }

            if (net.output_dim <= 10) {
                // Two-column display (fits 32-col console without wrapping).
                // By default we show softmax probabilities (over the calibrated logits).
                const fix16* disp = probs_buf ? probs_buf : logits_buf;
                for (int row = 0; row < 5; row++) {
                    uint32_t i = (uint32_t)row;
                    uint32_t j = (uint32_t)(row + 5);

                    char left[48]  = "";
                    char right[48] = "";

                    if (i < net.output_dim) {
                        char vbuf[32];
                        fix16_to_str_4(disp[i], vbuf, sizeof(vbuf));
                        snprintf(left, sizeof(left), "%lu: %s", (unsigned long)i, vbuf);
                    }
                    if (j < net.output_dim) {
                        char vbuf[32];
                        fix16_to_str_4(disp[j], vbuf, sizeof(vbuf));
                        snprintf(right, sizeof(right), "%lu: %s", (unsigned long)j, vbuf);
                    }

                    int screen_row = 19 + row;
                    clear_row(screen_row);
                    iprintf("\x1b[%d;0H%-15s %-15s", screen_row, left, right);
                }
            } else {
                // Fallback: print the first few values (one per row).
                const fix16* disp = probs_buf ? probs_buf : logits_buf;
                for (uint32_t i = 0; (i < 5u) && (i < net.output_dim); i++) {
                    int screen_row = 19 + (int)i;
                    clear_row(screen_row);

                    char vbuf[32];
                    fix16_to_str_4(disp[i], vbuf, sizeof(vbuf));
                    iprintf("\x1b[%d;0H%lu: %s", screen_row, (unsigned long)i, vbuf);
                }
            }
} else if ((down & KEY_A) && !net_ok) {
            clear_row(17);
            iprintf("\x1b[17;0H No weights loaded!          ");
        }

    }

    if (probs_buf) free(probs_buf);
    if (baseline_logits) free(baseline_logits);
    if (logits_buf) free(logits_buf);
    free_net(&net);
    return 0;
}
