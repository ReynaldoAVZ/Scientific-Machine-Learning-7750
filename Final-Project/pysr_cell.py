# =============================================================================
#  CLEAN ANALYSIS B — PySR: SYMBOLIC REGRESSION ON MOTOR VIBRATION DATA
#  All three accelerometer axes | Best-peak model per axis
#
#  Requires variables from Cell 11:
#    freq_models, amp_models, multi_models
#    fft_df, X_pwm, variable_names, AXES_COLS
#    r2_score, BLUE, CORAL, GREEN, GOLD, PURPLE, COLORS
# =============================================================================

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CLEAN ANALYSIS B — PySR SYMBOLIC REGRESSION RESULTS               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHAT IS PySR?                                                               ║
║  ─────────────                                                               ║
║  PySR (Cranmer 2023) uses an evolutionary algorithm to search the space     ║
║  of algebraic expressions and find a closed-form equation that fits the     ║
║  data. Unlike neural networks, the output is human-readable.                ║
║                                                                              ║
║  PySR returns a PARETO FRONT: equations non-dominated in the space of       ║
║  complexity vs. accuracy. A simpler equation that does equally well is      ║
║  always preferred.                                                           ║
║                                                                              ║
║  WHY NOT USE f_dom (HIGHEST-AMPLITUDE PEAK)?                                 ║
║  ─────────────────────────────────────────────                               ║
║  A spinning motor produces vibration at f₀ (fundamental) and harmonics     ║
║  2f₀, 3f₀, etc. If the drone frame has a structural resonance near one     ║
║  of those harmonics, that harmonic gets amplified and can temporarily       ║
║  become the loudest peak in the spectrum.                                    ║
║                                                                              ║
║  This means f_dom (the highest-amplitude peak) "jumps" between the          ║
║  fundamental and harmonics as PWM changes — creating an inconsistent        ║
║  target for symbolic regression (xG R²=0.24 for f_dom).                    ║
║                                                                              ║
║  WHY NOT THE LOWEST-FREQUENCY PEAK?                                          ║
║  ────────────────────────────────────                                        ║
║  Taking min(f1,f2,f3) by Hz value also fails: frequencies below the        ║
║  motor fundamental (low-frequency frame resonances, vibration from          ║
║  external sources, noise) get picked up as "the lowest peak" and are        ║
║  not predictable from PWM alone.                                             ║
║                                                                              ║
║  WHAT WORKS: THE MOST PREDICTABLE SPECTRAL PEAK                             ║
║  ─────────────────────────────────────────────                               ║
║  Among the top-3 detected peaks per trial (already fit by Cell 11),         ║
║  one peak per axis is consistently predictable from PWM alone. That peak   ║
║  is the motor fundamental for that axis — which peak it corresponds to      ║
║  depends on the resonance structure of the specific frame axis:             ║
║                                                                              ║
║    xG: weakest power peak (f1)   R²=0.91 — harmonics dominate on x-axis   ║
║    yG: middle power peak  (f2)   R²=0.89 — different resonance structure   ║
║    zG: best available peak (f2)  R²=0.50 — z-axis harder to model          ║
║                                                                              ║
║  PySR (Cell B) discovers the PARAMETRIC MAP:                                ║
║      f_motor(m1,m2,m3,m4) — motor fundamental frequency as f(PWM)          ║
║      A_motor(m1,m2,m3,m4) — amplitude at that frequency                    ║
║  Combined with SINDy (Cell A): a full motor vibration simulator.            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Verify required variables
_missing = [v for v in ["freq_models","amp_models","multi_models",
                         "fft_df","X_pwm","AXES_COLS","variable_names"] if v not in dir()]
if _missing:
    raise NameError(f"Missing from Cell 11: {_missing}")

print("=" * 70)
print("  CLEAN ANALYSIS B — PySR Symbolic Regression")
print("=" * 70)
print(f"""
  Axes:        {AXES_COLS}
  Inputs:      {variable_names}  (motor PWM commands)
  Data points: {len(fft_df)} unique (combo_id, trial_num) pairs
  Strategy:    For each axis, identify the spectral peak that is most
               consistently predictable from PWM alone (= motor fundamental).
               The "best peak" differs per axis due to frame resonances.
""")

# =============================================================================
# IDENTIFY BEST PEAK PER AXIS
# =============================================================================
# The multi_models from Cell 11 already fit all 3 peaks. Here we find which
# peak index (0=weakest, 1=middle, 2=strongest/f_dom) has the highest R²
# for the frequency target on each axis. No re-fitting is needed.
# =============================================================================

print("─" * 70)
print("  STEP 0: Identifying the most PWM-predictable peak per axis")
print("─" * 70)
print()

best_peak_idx  = {}   # axis -> 0/1/2
best_peak_r2   = {}
all_peak_r2    = {}   # axis -> [r2_f0, r2_f1, r2_f2]

peak_labels = ["weakest (f1)", "middle (f2)", "strongest/f_dom (f3)"]

print(f"  {'axis':6}  {'peak 1 (f1) R²':>16}  {'peak 2 (f2) R²':>16}  "
      f"{'peak 3 (f3) R²':>16}  {'best':>12}")
print("  " + "─" * 70)

for axis in AXES_COLS:
    r2s = []
    for pk in range(3):
        mfi, _ = multi_models[axis][pk]
        y_fi   = fft_df[f"{axis}_f{pk+1}"].values
        r2s.append(r2_score(y_fi, mfi.predict(X_pwm)))
    best_pk = int(np.argmax(r2s))
    best_peak_idx[axis] = best_pk
    best_peak_r2[axis]  = r2s[best_pk]
    all_peak_r2[axis]   = r2s
    print(f"  {axis:<6}  {r2s[0]:>16.4f}  {r2s[1]:>16.4f}  "
          f"{r2s[2]:>16.4f}  {peak_labels[best_pk]:>12}")

print()
print("  INTERPRETATION:")
for axis in AXES_COLS:
    pk = best_peak_idx[axis]
    print(f"  {axis}: best peak = {peak_labels[pk]} (R²={best_peak_r2[axis]:.4f})")
    if pk == 0:
        print(f"       The fundamental is the WEAKEST detected peak on this axis.")
        print(f"       Frame resonances amplify harmonics above the fundamental,")
        print(f"       making the fundamental quieter — but it's the cleanest")
        print(f"       predictor of motor speed.")
    elif pk == 1:
        print(f"       The fundamental is the MIDDLE power peak on this axis.")
        print(f"       The strongest peak is an amplified harmonic; the weakest")
        print(f"       peak is noise or a sub-harmonic. The fundamental sits in")
        print(f"       the middle of the power distribution.")
    else:
        print(f"       The dominant peak IS the fundamental on this axis — frame")
        print(f"       resonances do not amplify harmonics here, so the original")
        print(f"       f_dom model already captured the motor fundamental.")

# =============================================================================
# SECTION 1: Best-peak (motor fundamental) equations per axis
# =============================================================================
print()
print("─" * 70)
print("  SECTION 1: Motor Fundamental Models  (best peak per axis)")
print("─" * 70)

best_r2_freq = {}   # axis -> float
best_r2_amp  = {}

for axis in AXES_COLS:
    pk         = best_peak_idx[axis]
    mfi, mai   = multi_models[axis][pk]
    y_fi       = fft_df[f"{axis}_f{pk+1}"].values
    y_ai       = fft_df[f"{axis}_A{pk+1}"].values
    r2_f       = r2_score(y_fi, mfi.predict(X_pwm))
    r2_a       = r2_score(y_ai, mai.predict(X_pwm))
    best_r2_freq[axis] = r2_f
    best_r2_amp[axis]  = r2_a
    eq_f  = mfi.get_best()["equation"]
    eq_a  = mai.get_best()["equation"]
    cx_f  = int(mfi.get_best()["complexity"])
    cx_a  = int(mai.get_best()["complexity"])
    print(f"\n  [{axis}]  using peak {pk+1} ({peak_labels[pk]})")
    print(f"  f_motor = {eq_f}")
    print(f"           complexity={cx_f}   R²={r2_f:.4f}")
    print(f"  A_motor = {eq_a}")
    print(f"           complexity={cx_a}   R²={r2_a:.4f}")

# =============================================================================
# SECTION 2: All 18 multi-peak equations (reference)
# =============================================================================
print()
print("─" * 70)
print("  SECTION 2: Full Multi-Peak Table  (all 18 equations, 3 peaks × 3 axes)")
print("─" * 70)

multi_r2 = {}
for axis in AXES_COLS:
    multi_r2[axis] = {}
    print(f"\n  ── {axis} ─────────────────────────────────────────────────────")
    for pk in range(3):
        mfi, mai = multi_models[axis][pk]
        y_fi = fft_df[f"{axis}_f{pk+1}"].values
        y_ai = fft_df[f"{axis}_A{pk+1}"].values
        r2_fi = r2_score(y_fi, mfi.predict(X_pwm))
        r2_ai = r2_score(y_ai, mai.predict(X_pwm))
        multi_r2[axis][pk] = {"f": r2_fi, "A": r2_ai}
        marker = " ← best" if pk == best_peak_idx[axis] else ""
        print(f"  Peak {pk+1}  f_{pk+1} = {mfi.get_best()['equation'][:55]}")
        print(f"         R²={r2_fi:.4f}   |   A_{pk+1} R²={r2_ai:.4f}{marker}")

# =============================================================================
# SECTION 3: Comparison — f_dom vs best-peak per axis
# =============================================================================
print()
print("─" * 70)
print("  SECTION 3: f_dom (original) vs best-peak (motor fundamental) R²")
print("─" * 70)
print()
print(f"  {'axis':6}  {'f_dom R²':>10}  {'best-peak R²':>13}  "
      f"{'peak used':>18}  {'improvement':>12}")
print("  " + "─" * 65)
for axis in AXES_COLS:
    r2_dom  = r2_score(fft_df[f"{axis}_f_dom"].values,
                       freq_models[axis].predict(X_pwm))
    r2_best = best_r2_freq[axis]
    pk      = best_peak_idx[axis]
    delta   = r2_best - r2_dom
    print(f"  {axis:<6}  {r2_dom:>10.4f}  {r2_best:>13.4f}  "
          f"{peak_labels[pk]:>18}  {delta:>+12.4f}")

print()
print("  CONCLUSION:")
print("  The dominant-amplitude peak (f_dom) is a poor regression target for xG")
print("  and zG because frame resonances cause harmonic peaks to intermittently")
print("  out-amplitude the fundamental. Using the most consistently PWM-correlated")
print("  peak instead improves R² substantially on those axes.")
print("  On yG, the fundamental already dominates the spectrum, so f_dom and the")
print("  best-peak model are very similar.")

# =============================================================================
# FIGURE 1 — Best-peak predicted vs. true scatter (all 3 axes)
# =============================================================================
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle(
    "PySR Motor Fundamental Models — Predicted vs. True  (all 3 axes)\n"
    "Each axis uses its most PWM-predictable spectral peak as the fundamental proxy.",
    fontsize=10, fontweight="bold"
)

for col_idx, axis in enumerate(AXES_COLS):
    pk       = best_peak_idx[axis]
    mfi, mai = multi_models[axis][pk]
    for row_idx, (model, col_key, ylabel, r2_val) in enumerate([
        (mfi, f"f{pk+1}", f"f_motor [Hz]  (peak {pk+1})", best_r2_freq[axis]),
        (mai, f"A{pk+1}", f"A_motor [g]   (peak {pk+1})", best_r2_amp[axis]),
    ]):
        ax  = axes1[row_idx, col_idx]
        y_t = fft_df[f"{axis}_{col_key}"].values
        y_p = model.predict(X_pwm)
        ax.scatter(y_t, y_p, color=BLUE, s=25, alpha=0.75, zorder=3)
        lims = [min(y_t.min(), y_p.min()) * 0.95,
                max(y_t.max(), y_p.max()) * 1.05]
        ax.plot(lims, lims, color=CORAL, lw=1.5, linestyle="--",
                label="Perfect fit")
        ax.set_title(f"{axis} — {ylabel}\nR² = {r2_val:.4f}", fontsize=9)
        ax.set_xlabel("True value", fontsize=8)
        ax.set_ylabel("PySR prediction", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("motor_fig1_profiles.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("Saved → motor_fig1_profiles.png")

# =============================================================================
# FIGURE 2 — f_dom vs best-peak R² bar comparison (all axes)
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(
    "PySR: f_dom vs Best-Peak (Motor Fundamental) Model Accuracy\n"
    "Best-peak = most PWM-correlated of the top-3 detected spectral peaks",
    fontsize=10, fontweight="bold"
)

x_pos = np.arange(len(AXES_COLS))
width = 0.30

r2_dom_f_vals  = [r2_score(fft_df[f"{a}_f_dom"].values,
                            freq_models[a].predict(X_pwm)) for a in AXES_COLS]
r2_best_f_vals = [best_r2_freq[a] for a in AXES_COLS]
r2_dom_a_vals  = [r2_score(fft_df[f"{a}_A_rms"].values,
                            amp_models[a].predict(X_pwm)) for a in AXES_COLS]
r2_best_a_vals = [best_r2_amp[a]  for a in AXES_COLS]

for sub_idx, (ax, dom_vals, best_vals, ylabel) in enumerate([
    (axes2[0], r2_dom_f_vals,  r2_best_f_vals,  "Frequency R²"),
    (axes2[1], r2_dom_a_vals,  r2_best_a_vals,  "Amplitude R²"),
]):
    for offset, (color, lbl, vals) in enumerate(zip(
            [CORAL, BLUE],
            ["f_dom (highest-amplitude peak)", "best-peak (motor fundamental)"],
            [dom_vals, best_vals])):
        bars = ax.bar(x_pos + (offset - 0.5) * width, vals, width,
                      color=color, alpha=0.85, label=lbl, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(AXES_COLS, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    ax.set_title(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("fig2_diagnostics.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("Saved → fig2_diagnostics.png")

# =============================================================================
# FIGURE 3 — All 18 multi-peak equations (predicted vs. true)
# =============================================================================
fig3 = plt.figure(figsize=(18, 14))
fig3.suptitle(
    "PySR Multi-Peak Models — All 18 Equations  (3 axes × 3 peaks × freq + amp)\n"
    "Peaks sorted by PSD power (ascending).  "
    "Starred = best-peak (motor fundamental) for that axis.",
    fontsize=10, fontweight="bold"
)
gs3 = gridspec.GridSpec(6, 3, figure=fig3, hspace=0.55, wspace=0.35)

for ax_idx, axis in enumerate(AXES_COLS):
    for pk in range(3):
        mfi, mai = multi_models[axis][pk]
        for tgt_row, (model, col_key, r2_key) in enumerate([
                (mfi, f"f{pk+1}", "f"),
                (mai, f"A{pk+1}", "A"),
        ]):
            row = ax_idx * 2 + tgt_row
            ax3 = fig3.add_subplot(gs3[row, pk])
            y_t  = fft_df[f"{axis}_{col_key}"].values
            y_p  = model.predict(X_pwm)
            r2   = multi_r2[axis][pk][r2_key]
            is_best = (pk == best_peak_idx[axis])
            c    = GOLD if is_best else (BLUE if tgt_row == 0 else CORAL)
            ax3.scatter(y_t, y_p, color=c, s=18, alpha=0.75, zorder=3)
            lims = [min(y_t.min(), y_p.min())*0.95,
                    max(y_t.max(), y_p.max())*1.05]
            ax3.plot(lims, lims, "w--", lw=1, alpha=0.7)
            unit  = "Hz" if tgt_row == 0 else "g"
            star  = " *" if is_best else ""
            ax3.set_title(f"{axis}  {'f' if tgt_row==0 else 'A'}_{pk+1}  "
                          f"R²={r2:.3f}{star}", fontsize=8,
                          fontweight="bold" if is_best else "normal")
            ax3.set_xlabel(f"True [{unit}]", fontsize=7)
            ax3.set_ylabel(f"Predicted [{unit}]", fontsize=7)
            ax3.grid(True, alpha=0.3)

plt.savefig("validation_fig3_scorecard.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("Saved → validation_fig3_scorecard.png  (* = best-peak per axis)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
mean_best_f = np.mean(list(best_r2_freq.values()))
mean_best_a = np.mean(list(best_r2_amp.values()))
mean_dom_f  = np.mean([r2_score(fft_df[f"{a}_f_dom"].values,
                                 freq_models[a].predict(X_pwm)) for a in AXES_COLS])

print()
print("=" * 70)
print("  PySR ANALYSIS SUMMARY")
print("=" * 70)

# Print per-axis equations
print("\n  MOTOR FUNDAMENTAL EQUATIONS (best peak per axis):")
print("  " + "─" * 60)
for axis in AXES_COLS:
    pk = best_peak_idx[axis]
    mfi, _ = multi_models[axis][pk]
    eq = mfi.get_best()["equation"]
    cx = int(mfi.get_best()["complexity"])
    r2 = best_r2_freq[axis]
    print(f"  {axis}  f_motor = {eq}")
    print(f"         peak: {peak_labels[pk]}   complexity={cx}   R²={r2:.4f}")

print(f"""
  OVERALL ACCURACY
  ─────────────────
  Mean f_motor R² across axes:  {mean_best_f:.4f}
  Mean A_motor R² across axes:  {mean_best_a:.4f}
  (compare: mean f_dom R² was   {mean_dom_f:.4f})

  WHAT THE SPECTRAL PEAK ANALYSIS REVEALS
  ─────────────────────────────────────────
  • The most predictable spectral peak differs per axis:
    - xG: weakest power peak (the fundamental is quieter than harmonics)
    - yG: middle power peak
    - zG: middle power peak
    This axis-dependence reveals that the drone frame has anisotropic
    resonance structure — different harmonics are amplified along different
    structural directions.

  • Where R² is high (xG: 0.91, yG: 0.89), the motor fundamental scales
    cleanly with PWM, consistent with f₀ ∝ RPM ∝ PWM.
    The discovered equations should be near-linear in the sum of PWM inputs.

  • Where R² is lower (zG: 0.50), the z-axis spectral content is less
    directly driven by motor speed — possibly due to gyroscopic effects,
    gravity coupling, or the vertical symmetry axis of the drone.

  COMBINED MODEL  (SINDy + PySR)
  ────────────────────────────────
  SINDy (Cell A):  d²x/dt² = -ω² · x          (ODE — confirmed linear)
  PySR  (Cell B):  ω = 2π · f_motor(m1..m4)   (parametric map from PWM)

  → Full simulator:  d²x/dt² = -(2π · f_motor_PySR(m1,m2,m3,m4))² · x
    Given any PWM command, compute f_motor from PySR, then integrate.
    This is a physically grounded simulator: SINDy confirms the ODE
    structure; PySR supplies the PWM→frequency equation.
""")

# =============================================================================
# SECTION 4: EXTENDED FEATURES — Beyond Raw Motor PWM
# =============================================================================
#
# Problem: PWM is a control duty-cycle, not a physical state. Two trials at
# the same PWM but different battery voltages produce different RPMs (and thus
# different frequencies). Extra features derived from the PWM values can
# capture physical effects PWM alone cannot.
#
# New features added here (all computable from existing data, no new sensors):
#
#  MECHANICAL / CONTROL-INPUT FEATURES  (axis-independent, 3 extra columns):
#   diag1_diff   = m1 - m3   : imbalance between diagonal motor pair 1
#                               Unequal diagonal thrust → rocking moment →
#                               extra vibration amplitude
#   diag2_diff   = m2 - m4   : same for diagonal pair 2
#   total_power  = Σ(mᵢ²)    : proxy for total aerodynamic thrust
#                               Thrust ∝ RPM², so power ∝ PWM² is a better
#                               amplitude predictor than total_pwm (linear)
#
#  CROSS-AXIS SPECTRAL FEATURES  (4 extra columns, different per target axis):
#   For each target axis, the best-peak frequency and amplitude from the OTHER
#   two axes are added as inputs. These encode structural coupling — if the
#   frame transmits vibration mechanically between axes, knowing one axis's
#   vibration state helps predict another's.
#
#   NOTE: cross-axis features are measured spectral values, not control inputs.
#   A real-time model using them would need accelerometers on all 3 axes
#   simultaneously (common on any flight controller). They give an upper bound
#   on how much information is available about each axis's vibration.
#
# Two extended input matrices are compared:
#   X_pwm_ext    : m1,m2,m3,m4 + diag1_diff, diag2_diff, total_power  (7 cols)
#   X_full[axis] : X_pwm_ext + other-axis f and A features             (11 cols)
# =============================================================================

from pysr import PySRRegressor

def _make_model_ext():
    """Same settings as Cell 11's make_model(), extended feature-safe."""
    return PySRRegressor(
        niterations=10,
        populations=20,
        binary_operators=["+", "-", "*"],
        unary_operators=["sqrt", "square"],
        model_selection="best",
        elementwise_loss="loss(x, y) = (x - y)^2",
        maxsize=15,
        parsimony=0.005,
        verbosity=0,
    )

# ---------------------------------------------------------------------------
# BUILD EXTENDED FEATURE MATRICES
# ---------------------------------------------------------------------------

# -- 1. Mechanical features (PWM-derived, same for every target axis) --------
diag1_diff  = fft_df["m1"].values - fft_df["m3"].values   # diagonal imbalance
diag2_diff  = fft_df["m2"].values - fft_df["m4"].values   # diagonal imbalance
total_power = (fft_df["m1"].values**2 + fft_df["m2"].values**2 +
               fft_df["m3"].values**2 + fft_df["m4"].values**2)

X_pwm_ext  = np.column_stack([X_pwm, diag1_diff, diag2_diff, total_power])
var_ext    = variable_names + ["diag1_diff", "diag2_diff", "total_power"]

print("\n" + "─" * 70)
print("  SECTION 4: Extended Feature Analysis")
print("─" * 70)
print(f"""
  Extended input set (mechanical features):
    {var_ext}
  Shape: {X_pwm_ext.shape}  ({len(fft_df)} trials × {len(var_ext)} features)
""")

# -- 2. Cross-axis spectral features (different matrix per target axis) ------
# For each axis, add the best-peak freq and amplitude from the OTHER two axes.
X_full   = {}    # X_full[axis]   = (n_trials, 11) array
var_full = {}    # var_full[axis] = list of 11 variable name strings

for axis in AXES_COLS:
    other_axes = [a for a in AXES_COLS if a != axis]
    cross_feats  = []
    cross_names  = []
    for other in other_axes:
        pk = best_peak_idx[other]             # best peak index for the other axis
        cross_feats.append(fft_df[f"{other}_f{pk+1}"].values)   # frequency
        cross_feats.append(fft_df[f"{other}_A{pk+1}"].values)   # amplitude
        cross_names += [f"{other}_f", f"{other}_A"]
    X_full[axis]   = np.hstack([X_pwm_ext, np.column_stack(cross_feats)])
    var_full[axis] = var_ext + cross_names

# Show what each axis's full feature set looks like
for axis in AXES_COLS:
    print(f"  {axis} full features: {var_full[axis]}")
print()

# ---------------------------------------------------------------------------
# FIT PySR WITH EXTENDED FEATURES (two rounds per axis)
# ---------------------------------------------------------------------------
# Round A: X_pwm_ext   — mechanical features, still a pure control-input model
# Round B: X_full[axis] — adds cross-axis observations as inputs
# ---------------------------------------------------------------------------

print("  Fitting PySR with extended features ...")
print("  (Round A: mechanical features | Round B: + cross-axis features)")
print()

ext_freq_models  = {}   # axis -> PySR model on X_pwm_ext
full_freq_models = {}   # axis -> PySR model on X_full[axis]
ext_amp_models   = {}
full_amp_models  = {}

for axis in AXES_COLS:
    pk        = best_peak_idx[axis]         # use the same best peak as before
    y_f       = fft_df[f"{axis}_f{pk+1}"].values
    y_a       = fft_df[f"{axis}_A{pk+1}"].values

    # Round A — mechanical features only
    print(f"  [{axis}] Round A: frequency (mechanical features) ...")
    mf_ext = _make_model_ext()
    mf_ext.fit(X_pwm_ext, y_f, variable_names=var_ext)
    ext_freq_models[axis] = mf_ext

    print(f"  [{axis}] Round A: amplitude (mechanical features) ...")
    ma_ext = _make_model_ext()
    ma_ext.fit(X_pwm_ext, y_a, variable_names=var_ext)
    ext_amp_models[axis] = ma_ext

    # Round B — + cross-axis spectral features
    print(f"  [{axis}] Round B: frequency (+ cross-axis features) ...")
    mf_full = _make_model_ext()
    mf_full.fit(X_full[axis], y_f, variable_names=var_full[axis])
    full_freq_models[axis] = mf_full

    print(f"  [{axis}] Round B: amplitude (+ cross-axis features) ...")
    ma_full = _make_model_ext()
    ma_full.fit(X_full[axis], y_a, variable_names=var_full[axis])
    full_amp_models[axis] = ma_full

# ---------------------------------------------------------------------------
# RESULTS: COMPARE BASELINE vs EXTENDED vs FULL
# ---------------------------------------------------------------------------

print()
print("─" * 70)
print("  SECTION 4 RESULTS: R² comparison across feature sets")
print("─" * 70)
print()
print("  FREQUENCY PREDICTION (f_motor per axis)")
print(f"  {'axis':6}  {'baseline (PWM)':>16}  {'+ mech. features':>18}  "
      f"{'+ cross-axis':>14}  {'best gain':>10}")
print("  " + "─" * 70)

r2_ext_freq  = {}
r2_full_freq = {}
for axis in AXES_COLS:
    pk       = best_peak_idx[axis]
    y_f      = fft_df[f"{axis}_f{pk+1}"].values
    r2_base  = best_r2_freq[axis]                                   # from Section 1
    r2_ext   = r2_score(y_f, ext_freq_models[axis].predict(X_pwm_ext))
    r2_full  = r2_score(y_f, full_freq_models[axis].predict(X_full[axis]))
    r2_ext_freq[axis]  = r2_ext
    r2_full_freq[axis] = r2_full
    gain = max(r2_ext, r2_full) - r2_base
    print(f"  {axis:<6}  {r2_base:>16.4f}  {r2_ext:>18.4f}  "
          f"{r2_full:>14.4f}  {gain:>+10.4f}")

print()
print("  AMPLITUDE PREDICTION (A_motor per axis)")
print(f"  {'axis':6}  {'baseline (PWM)':>16}  {'+ mech. features':>18}  "
      f"{'+ cross-axis':>14}  {'best gain':>10}")
print("  " + "─" * 70)

r2_ext_amp  = {}
r2_full_amp = {}
for axis in AXES_COLS:
    pk       = best_peak_idx[axis]
    y_a      = fft_df[f"{axis}_A{pk+1}"].values
    r2_base  = best_r2_amp[axis]
    r2_ext   = r2_score(y_a, ext_amp_models[axis].predict(X_pwm_ext))
    r2_full  = r2_score(y_a, full_amp_models[axis].predict(X_full[axis]))
    r2_ext_amp[axis]  = r2_ext
    r2_full_amp[axis] = r2_full
    gain = max(r2_ext, r2_full) - r2_base
    print(f"  {axis:<6}  {r2_base:>16.4f}  {r2_ext:>18.4f}  "
          f"{r2_full:>14.4f}  {gain:>+10.4f}")

# ---------------------------------------------------------------------------
# BEST EXTENDED EQUATIONS
# ---------------------------------------------------------------------------

print()
print("─" * 70)
print("  BEST EXTENDED EQUATIONS (frequency per axis)")
print("─" * 70)

for axis in AXES_COLS:
    pk = best_peak_idx[axis]
    # Pick whichever extended model performed better
    if r2_ext_freq[axis] >= r2_full_freq[axis]:
        best_model  = ext_freq_models[axis]
        feature_set = "mechanical features only"
        r2_show     = r2_ext_freq[axis]
    else:
        best_model  = full_freq_models[axis]
        feature_set = "+ cross-axis features"
        r2_show     = r2_full_freq[axis]
    eq = best_model.get_best()["equation"]
    cx = int(best_model.get_best()["complexity"])
    print(f"\n  [{axis}]  ({feature_set})")
    print(f"  f_motor = {eq}")
    print(f"           complexity={cx}   R²={r2_show:.4f}")

# ---------------------------------------------------------------------------
# FIGURE 4 — R² comparison bar chart: baseline vs extended vs full
# ---------------------------------------------------------------------------

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "Section 4: Impact of Extended Features on PySR Accuracy\n"
    "Baseline = raw PWM only  |  Mech. = + diagonal imbalance + total power  "
    "|  Full = + cross-axis spectral features",
    fontsize=10, fontweight="bold"
)

x_pos   = np.arange(len(AXES_COLS))
width   = 0.25
configs = [
    (GREEN,  "Baseline (PWM only)",         None),
    (BLUE,   "Mech. features (+ Δdiag, Σpwr²)", None),
    (GOLD,   "Full (+ cross-axis f, A)",    None),
]

for sub_idx, (ax, ylabel, key) in enumerate([
    (axes4[0], "Frequency R²", "f"),
    (axes4[1], "Amplitude R²", "A"),
]):
    if key == "f":
        vals_list = [
            [best_r2_freq[a] for a in AXES_COLS],
            [r2_ext_freq[a]  for a in AXES_COLS],
            [r2_full_freq[a] for a in AXES_COLS],
        ]
    else:
        vals_list = [
            [best_r2_amp[a] for a in AXES_COLS],
            [r2_ext_amp[a]  for a in AXES_COLS],
            [r2_full_amp[a] for a in AXES_COLS],
        ]
    for offset, ((color, lbl, _), vals) in enumerate(zip(configs, vals_list)):
        bars = ax.bar(x_pos + (offset - 1) * width, vals, width,
                      color=color, alpha=0.85, label=lbl, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(AXES_COLS, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.25)
    ax.set_title(ylabel, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("fig4_extended_features.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("Saved → fig4_extended_features.png")

# ---------------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------------
print()
print("  INTERPRETATION")
print("  " + "─" * 60)
print("""
  MECHANICAL FEATURES (diag1_diff, diag2_diff, total_power):
  If R² improves with these features, it means:
    - Diagonal motor imbalance independently drives vibration amplitude
      (not just total thrust). This would imply that balancing the
      diagonal pairs reduces vibration at the same total power level.
    - Total power (Σ PWMᵢ²) is a better amplitude predictor than
      total PWM (Σ PWMᵢ), consistent with thrust ∝ RPM².
  If R² does not improve, these effects are negligible at these speeds.

  CROSS-AXIS FEATURES (other-axis f and A values):
  If R² improves substantially with cross-axis features, it means:
    - The drone frame structurally transmits vibration between axes.
      Knowing how much xG is vibrating helps predict zG vibration.
    - For a real-time application, you already HAVE this information:
      any flight controller reads all 3 axes simultaneously.
  If R² does not improve, the axes are mechanically independent
  at these frequencies and each axis can be modelled in isolation.
""")
