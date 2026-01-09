# FlyColor.py (Full override)
# v2026-01-09e
# Version change notes:
# - FIXED: Recipe detail view bar charts now match the main pseudo-fly bar (Altair with controllable colors/width).
# - ADDED: Sidebar controls for pseudo-fly bar width + per-channel colors (R/G/B), applied everywhere.
# - HARDENED: Added explicit unique keys to sidebar sliders/uploader toggles to avoid StreamlitDuplicateElementId surprises.
# - OPTIONAL: "Lock brightness across samples" (uses global Brightness factor; hides per-sample brightness slider when locked).
#
# Keeps:
# - Material optics (Transmittance & Base Reflectance) with optional spectral CSV upload.
# - LED SPD CSV upload, UV cut, human swatch modes (brightness/hue preserve),
# - pseudo-fly (A bars + B desat swatch), dye menu + concentration inputs,
# - per-sample K/W controls, global/per-sample normalization option,
# - ΔS + ΔL matrices, recipe finder + apply to Agar_1/Agar_2, debug expanders.

import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -----------------------------
# Utilities
# -----------------------------
def ensure_1d(x):
    return np.array(x, dtype=float).reshape(-1)

def trapz(y, x):
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)

    # NumPy >= 2.0 uses trapezoid()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    # Fallback for older NumPy
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    raise AttributeError("NumPy has neither trapezoid nor trapz; please check numpy version.")


def interp_to(wl_src, y_src, wl_dst):
    wl_src = ensure_1d(wl_src)
    y_src = ensure_1d(y_src)
    wl_dst = ensure_1d(wl_dst)
    if len(wl_src) != len(y_src):
        raise ValueError(f"interp_to: wl_src len {len(wl_src)} != y_src len {len(y_src)}")
    order = np.argsort(wl_src)
    wl_src = wl_src[order]
    y_src = y_src[order]
    _, idx = np.unique(wl_src, return_index=True)
    wl_src = wl_src[idx]
    y_src = y_src[idx]
    return np.interp(wl_dst, wl_src, y_src, left=y_src[0], right=y_src[-1])

def clamp01(x):
    return np.clip(np.array(x, dtype=float), 0.0, 1.0)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


# -----------------------------
# Reference wavelength grid
# -----------------------------
WL_REF = np.arange(300, 701, 5, dtype=float)


# -----------------------------
# CIE 1931 2° CMFs (compact approximation)
# -----------------------------
def cie1931_cmf_approx(wl):
    wl = ensure_1d(wl)

    def g(mu, sig):
        return np.exp(-0.5 * ((wl - mu) / sig) ** 2)

    xbar = 1.065 * g(595, 33) + 0.366 * g(446, 20) - 0.065 * g(501, 20)
    ybar = 1.000 * g(556, 28)
    zbar = 1.747 * g(446, 18)

    xbar = np.clip(xbar, 0, None)
    ybar = np.clip(ybar, 0, None)
    zbar = np.clip(zbar, 0, None)

    if np.max(ybar) > 0:
        xbar = xbar / np.max(ybar)
        ybar = ybar / np.max(ybar)
        zbar = zbar / np.max(ybar)
    return xbar, ybar, zbar


# -----------------------------
# Human color: XYZ -> sRGB
# -----------------------------
def xyz_to_srgb(XYZ):
    XYZ = np.array(XYZ, dtype=float)
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570],
    ], dtype=float)
    rgb_lin = M @ XYZ
    rgb_lin = np.clip(rgb_lin, 0.0, None)

    a = 0.055
    rgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, (1 + a) * np.power(rgb_lin, 1/2.4) - a)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def rgb_to_hex(rgb01):
    rgb01 = np.clip(np.array(rgb01, dtype=float), 0.0, 1.0)
    r, g, b = (int(round(float(x) * 255)) for x in rgb01)
    return f"#{r:02x}{g:02x}{b:02x}"

def normalize_xyz_brightness(XYZ, target_Y=1.0):
    XYZ = np.array(XYZ, dtype=float)
    Y = float(XYZ[1])
    if Y <= 0:
        return XYZ
    return XYZ * (float(target_Y) / Y)


# -----------------------------
# Presentation-safe desaturation
# -----------------------------
def desaturate_rgb(rgb01, amount=0.55):
    rgb01 = np.clip(np.array(rgb01, dtype=float), 0.0, 1.0)
    gray = float(np.mean(rgb01))
    out = (1.0 - float(amount)) * rgb01 + float(amount) * gray
    return np.clip(out, 0.0, 1.0)


# -----------------------------
# Illuminant SPD: built-in + UV cut
# -----------------------------
def make_coolwhite_spd(wl, coolness=0.7):
    wl = ensure_1d(wl)
    blue = np.exp(-0.5 * ((wl - 450) / 18) ** 2)
    phosphor = np.exp(-0.5 * ((wl - 560) / 55) ** 2)
    spd = (0.7 + 0.8*coolness) * blue + (1.4 - 0.9*coolness) * phosphor
    spd = np.clip(spd, 0, None)
    spd = spd / np.max(spd) if np.max(spd) > 0 else spd
    return spd

def apply_uv_cut(wl, spd, strength=0.95, knee=400, softness=8):
    wl = ensure_1d(wl)
    spd = ensure_1d(spd)
    softness = max(float(softness), 1e-6)
    mask = 1.0 / (1.0 + np.exp((wl - float(knee)) / softness))  # ~1 below knee, ~0 above
    spd2 = spd * (1.0 - float(strength) * mask)
    spd2 = spd2 / np.max(spd2) if np.max(spd2) > 0 else spd2
    return spd2

def load_spd_csv(uploaded_file):
    """
    Accepts CSV with either:
      - header: wavelength, spd (names can vary)
      - no header: first col wavelength, second col spd
    Returns wl, spd as np arrays.
    """
    raw = uploaded_file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), header=None)

    if df.shape[1] < 2:
        raise ValueError("SPD CSV must have at least two columns (wavelength_nm, spd).")

    numeric_counts = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        numeric_counts.append((c, int(s.notna().sum())))
    numeric_counts.sort(key=lambda x: x[1], reverse=True)
    c1 = numeric_counts[0][0]
    c2 = numeric_counts[1][0]

    wl = pd.to_numeric(df[c1], errors="coerce").to_numpy(dtype=float)
    spd = pd.to_numeric(df[c2], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(wl) & np.isfinite(spd)
    wl = wl[mask]
    spd = spd[mask]
    if wl.size < 10:
        raise ValueError("SPD CSV parse failed (too few valid numeric rows).")

    order = np.argsort(wl)
    wl = wl[order]
    spd = spd[order]
    spd = np.clip(spd, 0.0, None)
    return wl, spd


# -----------------------------
# Material curves
# -----------------------------
def load_curve_csv(uploaded_file, value_name="value"):
    """
    Generic 2-col CSV reader:
      - header or no header; picks 2 most-numeric columns as (wavelength_nm, value)
    Returns wl, value arrays.
    """
    raw = uploaded_file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), header=None)

    if df.shape[1] < 2:
        raise ValueError("Curve CSV must have at least two columns (wavelength_nm, value).")

    numeric_counts = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        numeric_counts.append((c, int(s.notna().sum())))
    numeric_counts.sort(key=lambda x: x[1], reverse=True)
    c1 = numeric_counts[0][0]
    c2 = numeric_counts[1][0]

    wl = pd.to_numeric(df[c1], errors="coerce").to_numpy(dtype=float)
    val = pd.to_numeric(df[c2], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(wl) & np.isfinite(val)
    wl = wl[mask]
    val = val[mask]
    if wl.size < 10:
        raise ValueError(f"{value_name} CSV parse failed (too few valid numeric rows).")

    order = np.argsort(wl)
    wl = wl[order]
    val = val[order]
    return wl, clamp01(val)

def make_constant_curve(wl, v):
    wl = ensure_1d(wl)
    return np.full_like(wl, float(np.clip(v, 0.0, 1.0)), dtype=float)

def apply_material_effect(wl, R_dye_model, R_base_curve, T_curve):
    """
    Simple material optics:
      R_eff(λ) = (1 - T(λ)) * [ R_base(λ) + (1 - R_base(λ)) * R_dye_model(λ) ]
    """
    wl = ensure_1d(wl)
    Rm = clamp01(R_dye_model)
    Rb = clamp01(R_base_curve)
    T  = clamp01(T_curve)
    R_eff = (1.0 - T) * (Rb + (1.0 - Rb) * Rm)
    return clamp01(R_eff)


# -----------------------------
# Dye library (template absorbance spectra)
# -----------------------------
def log_gauss_abs(wl, peak_nm, width_nm, amp=1.0):
    wl = ensure_1d(wl)
    return amp * np.exp(-0.5 * ((wl - float(peak_nm)) / float(width_nm)) ** 2)

DYE_LIBRARY = {
    "Blue No.1 (藍色一號)": dict(peaks=[(630, 28, 1.0)], note="template"),
    "Yellow No.4 (黃色四號)": dict(peaks=[(427, 22, 1.0)], note="template"),
    "Red No.6 (紅色六號)": dict(peaks=[(510, 28, 1.0)], note="template"),
    "Red No.7 (紅色七號)": dict(peaks=[(520, 35, 1.0)], note="template"),
}

def dye_absorbance_spectrum(wl, dye_name):
    spec = DYE_LIBRARY[dye_name]
    A = np.zeros_like(wl, dtype=float)
    for (mu, sig, amp) in spec["peaks"]:
        A += log_gauss_abs(wl, mu, sig, amp=amp)
    if np.max(A) > 0:
        A = A / np.max(A)
    return A


# -----------------------------
# Fly receptor sensitivities (template)
# -----------------------------
def receptor_sensitivity_templates(wl):
    wl = ensure_1d(wl)
    def g(mu, sig):
        return np.exp(-0.5 * ((wl - mu) / sig) ** 2)

    sens = {}
    sens["Rh1"] = 0.85*g(480, 55) + 0.15*g(540, 80)  # broad blue-green
    sens["Rh3"] = g(345, 18)                         # UV
    sens["Rh4"] = g(375, 22)                         # UV
    sens["Rh5"] = g(437, 22)                         # Blue
    sens["Rh6"] = g(508, 28)                         # Green

    for k in sens:
        m = np.max(sens[k])
        sens[k] = sens[k] / m if m > 0 else sens[k]
    return sens

DROS_RECEPTORS = receptor_sensitivity_templates(WL_REF)


# -----------------------------
# Agar optics model
# -----------------------------
def agar_reflectance(wl, dye_concs_mgml, K=80.0, whitening=0.25):
    wl = ensure_1d(wl)
    whitening = float(np.clip(whitening, 0.0, 1.0))
    totalA = np.zeros_like(wl, dtype=float)

    for dye_name, conc in dye_concs_mgml.items():
        conc = float(conc)
        if conc <= 0:
            continue
        A = dye_absorbance_spectrum(wl, dye_name)
        totalA += conc * A

    core = np.exp(-float(K) * totalA)
    R = whitening + (1.0 - whitening) * core
    return np.clip(R, 0.0, 1.0)

def stimulus_spd(wl, illuminant, reflectance, brightness_factor=1.0):
    wl = ensure_1d(wl)
    illum = ensure_1d(illuminant)
    refl = ensure_1d(reflectance)
    stim = (illum * refl) * float(brightness_factor)
    return np.clip(stim, 0.0, None)


# -----------------------------
# Human XYZ integration
# -----------------------------
def integrate_xyz(wl, stim):
    wl = ensure_1d(wl)
    stim = ensure_1d(stim)
    xbar, ybar, zbar = cie1931_cmf_approx(wl)
    X = trapz(stim * xbar, wl)
    Y = trapz(stim * ybar, wl)
    Z = trapz(stim * zbar, wl)
    return np.array([X, Y, Z], dtype=float)


# -----------------------------
# Fly catches + Von Kries
# -----------------------------
def receptor_catches(wl, stim, receptors):
    wl = ensure_1d(wl)
    stim = ensure_1d(stim)
    out = {}
    for name, sens in receptors.items():
        sens_i = sens if len(sens) == len(wl) else interp_to(WL_REF, sens, wl)
        out[name] = float(trapz(stim * sens_i, wl))
    return out

def von_kries_normalize(catches, white_catches, eps=1e-12):
    out = {}
    for k in catches:
        out[k] = float(catches[k]) / max(float(white_catches.get(k, 1.0)), eps)
    return out


# -----------------------------
# RNL ΔS (simple)
# -----------------------------
def rnl_delta_s(c1, c2, noise=None, channels=("Rh3", "Rh4", "Rh5", "Rh6"), eps=1e-12):
    if noise is None:
        noise = {ch: 0.10 for ch in channels}  # placeholder
    v = []
    for ch in channels:
        q1 = max(float(c1.get(ch, 0.0)), eps)
        q2 = max(float(c2.get(ch, 0.0)), eps)
        d = np.log(q1) - np.log(q2)
        v.append(d / max(float(noise.get(ch, 0.10)), eps))
    v = np.array(v, dtype=float)
    return float(np.sqrt(np.sum(v**2)))

def delta_rel(a, b, eps=1e-12):
    a = max(float(a), eps)
    b = max(float(b), eps)
    return float(abs(a - b) / max(a, b))

def luminance_delta_rel(c1, c2, lum_ch="Rh1", eps=1e-12):
    l1 = max(float(c1.get(lum_ch, 0.0)), eps)
    l2 = max(float(c2.get(lum_ch, 0.0)), eps)
    return float(abs(l1 - l2) / max(l1, l2))

def fly_Lsum(c, channels=("Rh3", "Rh4", "Rh5", "Rh6")):
    return float(sum(float(c.get(ch, 0.0)) for ch in channels))


# -----------------------------
# Pseudo-fly mapping helpers
# -----------------------------
FLY_CH = ["Rh1", "Rh3", "Rh4", "Rh5", "Rh6"]

def get_mapping_channels(preset, custom_r=None, custom_g=None, custom_b=None):
    if preset == "Talk-safe (no UV) (R=Rh6,G=Rh1,B=Rh5)":
        return ("Rh6", "Rh1", "Rh5")
    if preset == "Intuitive (low-UV) (R=Rh5,G=Rh6,B=Rh4)":
        return ("Rh5", "Rh6", "Rh4")
    if preset == "Intuitive (with UV) (R=Rh5,G=Rh6,B=Rh3)":
        return ("Rh5", "Rh6", "Rh3")
    if preset == "Current (with UV) (R=Rh6,G=Rh5,B=Rh3)":
        return ("Rh6", "Rh5", "Rh3")
    r = custom_r or "Rh5"
    g = custom_g or "Rh6"
    b = custom_b or "Rh4"
    return (r, g, b)

def compute_global_scale_max(catches_list, chR, chG, chB):
    if not catches_list:
        return 1.0
    mx = 0.0
    for c in catches_list:
        v = np.array([float(c[chR]), float(c[chG]), float(c[chB])], dtype=float)
        mx = max(mx, float(np.max(v)))
    return mx if mx > 0 else 1.0

def pseudo_hex_from_vec(vec, scale_max, desat_amount=0.0):
    m = max(float(scale_max), 1e-12)
    v = np.clip(np.array(vec, dtype=float) / m, 0.0, 1.0)
    if desat_amount and float(desat_amount) > 0:
        v = desaturate_rgb(v, amount=float(desat_amount))
    return rgb_to_hex(v)


# -----------------------------
# Altair bar helper (single source of truth)
# -----------------------------
def alt_channel_bar(df_bar, bar_channels, ch_colors, bar_width_px=24, height_px=180):
    """
    df_bar columns: Channel (str), Relative (float in 0..1)
    """
    # keep deterministic order + stable color mapping
    ranges = [ch_colors.get(ch, "#888888") for ch in bar_channels]
    return (
        alt.Chart(df_bar)
        .mark_bar(size=int(bar_width_px))
        .encode(
            x=alt.X("Channel:N", sort=bar_channels, title=None),
            y=alt.Y("Relative:Q", scale=alt.Scale(domain=[0, 1]), title="Rel"),
            color=alt.Color(
                "Channel:N",
                scale=alt.Scale(domain=bar_channels, range=ranges),
                legend=None,
            ),
            tooltip=["Channel", alt.Tooltip("Relative:Q", format=".3f")],
        )
        .properties(height=int(height_px))
    )


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="FlyColor (Human vs Fly vision)", layout="wide")
st.title("FlyColor — Human vs Fly (template model + LED upload + material optics + recipe finder)")


# Sidebar: Illuminant & settings
with st.sidebar:
    st.header("Illuminant (LED)")
    illum_source = st.radio(
        "Illuminant source",
        ["Built-in Cool White", "Upload SPD CSV"],
        index=0,
        key="illum_source"
    )

    spd_uploaded_raw = None
    wl_led_raw = None

    if illum_source == "Built-in Cool White":
        coolness = st.slider(
            "Coolness (0 warm → 1 cool)",
            0.0, 1.0, 0.70, 0.01,
            key="coolness"
        )
        illum_raw = make_coolwhite_spd(WL_REF, coolness=coolness)
    else:
        up = st.file_uploader(
            "Upload LED SPD CSV (2 columns: wavelength_nm, spd)",
            type=["csv"],
            key="up_spd"
        )
        if up is None:
            st.warning("Upload a CSV to use custom LED SPD, or switch back to Built-in.")
            illum_raw = make_coolwhite_spd(WL_REF, coolness=0.70)
        else:
            wl_led_raw, spd_uploaded_raw = load_spd_csv(up)
            spd_interp = interp_to(wl_led_raw, spd_uploaded_raw, WL_REF)
            spd_interp = np.clip(spd_interp, 0.0, None)
            spd_interp = spd_interp / np.max(spd_interp) if np.max(spd_interp) > 0 else spd_interp
            illum_raw = spd_interp

    st.subheader("UV cut (default ON)")
    apply_uv = st.checkbox("Apply UV cut", value=True, key="apply_uv")
    uv_cut = st.slider(
        "UV cut strength (300–~knee)",
        0.0, 1.0, 0.95, 0.01,
        key="uv_cut"
    )
    uv_knee = st.slider(
        "UV cut knee (nm)",
        360, 420, 400, 1,
        key="uv_knee"
    )
    uv_soft = st.slider(
        "UV cut softness (nm)",
        1, 30, 8, 1,
        key="uv_soft"
    )

    illum = apply_uv_cut(WL_REF, illum_raw, strength=uv_cut, knee=uv_knee, softness=uv_soft) if apply_uv else illum_raw

    # -------------------------
    # Material optics
    # -------------------------
    st.header("Material optics")
    st.caption("Controls for overall translucency / base reflectance. Default preset: 1% agar (default).")

    mat_preset = st.selectbox(
        "Material preset",
        ["1% agar (default)", "Opaque matte plastic (demo)", "Glass-like (demo)", "Custom"],
        index=0,
        key="mat_preset"
    )

    # Defaults are tunable knobs (not claims of true physical values)
    if mat_preset == "1% agar (default)":
        base_R0 = 0.03
        base_T0 = 0.70
    elif mat_preset == "Opaque matte plastic (demo)":
        base_R0 = 0.20
        base_T0 = 0.05
    elif mat_preset == "Glass-like (demo)":
        base_R0 = 0.02
        base_T0 = 0.90
    else:
        base_R0 = 0.03
        base_T0 = 0.70

    use_Rbase_csv = st.checkbox("Use spectral Base Reflectance R_base(λ) CSV", value=False, key="use_Rbase_csv")
    use_T_csv = st.checkbox("Use spectral Transmittance T(λ) CSV", value=False, key="use_T_csv")

    if not use_Rbase_csv:
        R_base_scalar = st.slider(
            "Base reflectance R_base (scalar)",
            0.0, 1.0, float(base_R0), 0.01,
            key="R_base_scalar"
        )
        R_base_curve = make_constant_curve(WL_REF, R_base_scalar)
        wl_Rb_raw = None
        Rb_raw = None
    else:
        upRb = st.file_uploader(
            "Upload R_base(λ) CSV (wavelength_nm, R in 0–1)",
            type=["csv"],
            key="up_Rb"
        )
        if upRb is None:
            st.warning("R_base CSV not uploaded yet — falling back to scalar.")
            R_base_scalar = st.slider(
                "Base reflectance R_base (scalar)",
                0.0, 1.0, float(base_R0), 0.01,
                key="R_base_scalar_fallback"
            )
            R_base_curve = make_constant_curve(WL_REF, R_base_scalar)
            wl_Rb_raw = None
            Rb_raw = None
        else:
            wl_Rb_raw, Rb_raw = load_curve_csv(upRb, value_name="R_base")
            R_base_curve = interp_to(wl_Rb_raw, Rb_raw, WL_REF)
            R_base_curve = clamp01(R_base_curve)

    if not use_T_csv:
        T_scalar = st.slider(
            "Transmittance T (scalar)",
            0.0, 1.0, float(base_T0), 0.01,
            key="T_scalar"
        )
        T_curve = make_constant_curve(WL_REF, T_scalar)
        wl_T_raw = None
        T_raw = None
    else:
        upT = st.file_uploader(
            "Upload T(λ) CSV (wavelength_nm, T in 0–1)",
            type=["csv"],
            key="up_T"
        )
        if upT is None:
            st.warning("T CSV not uploaded yet — falling back to scalar.")
            T_scalar = st.slider(
                "Transmittance T (scalar)",
                0.0, 1.0, float(base_T0), 0.01,
                key="T_scalar_fallback"
            )
            T_curve = make_constant_curve(WL_REF, T_scalar)
            wl_T_raw = None
            T_raw = None
        else:
            wl_T_raw, T_raw = load_curve_csv(upT, value_name="T")
            T_curve = interp_to(wl_T_raw, T_raw, WL_REF)
            T_curve = clamp01(T_curve)

    st.divider()

    st.header("Samples")
    n_samples = st.number_input(
        "Number of agar samples",
        min_value=2, max_value=12, value=2, step=1,
        key="n_samples"
    )

    st.header("Agar optics (defaults)")
    K_default = st.number_input(
        "K (color strength)",
        min_value=0.0, max_value=500.0, value=80.0, step=1.0,
        key="K_default"
    )
    whitening_default = st.slider(
        "Whitening / scattering (flat component)",
        0.0, 1.0, 0.25, 0.01,
        key="whitening_default"
    )
    bright_default = st.slider(
        "Brightness factor (global)",
        0.1, 3.0, 1.0, 0.01,
        key="bright_default"
    )
    lock_brightness = st.checkbox(
        "Lock brightness across samples (use global)",
        value=True,
        key="lock_brightness"
    )

    st.header("Human visualization")
    human_mode = st.radio(
        "Human swatch mode",
        ["Brightness preserve", "Hue preserve"],
        index=0,
        key="human_mode"
    )

    st.header("Pseudo-fly mapping / normalization")
    mapping_preset = st.selectbox(
        "Mapping preset",
        [
            "Talk-safe (no UV) (R=Rh6,G=Rh1,B=Rh5)",
            "Intuitive (low-UV) (R=Rh5,G=Rh6,B=Rh4)",
            "Intuitive (with UV) (R=Rh5,G=Rh6,B=Rh3)",
            "Current (with UV) (R=Rh6,G=Rh5,B=Rh3)",
            "Custom",
        ],
        index=0,
        key="mapping_preset"
    )
    if mapping_preset == "Custom":
        cR = st.selectbox("Channel for R", FLY_CH, index=FLY_CH.index("Rh5"), key="cR")
        cG = st.selectbox("Channel for G", FLY_CH, index=FLY_CH.index("Rh6"), key="cG")
        cB = st.selectbox("Channel for B", FLY_CH, index=FLY_CH.index("Rh4"), key="cB")
    else:
        cR = cG = cB = None

    norm_mode = st.radio(
        "Normalization",
        ["Global (recommended)", "Per-sample (old)"],
        index=0,
        key="norm_mode"
    )
    swatch_desat = st.slider(
        "Swatch desaturation (presentation-safe)",
        0.0, 1.0, 0.55, 0.01,
        key="swatch_desat"
    )

    st.header("Pseudo-fly bar style")
    bar_width_px = st.slider(
        "Bar width (px)",
        4, 60, 24, 1,
        key="bar_width_px"
    )
    colR_hex = st.color_picker("Bar color for R channel", value="#d62728", key="colR_hex")
    colG_hex = st.color_picker("Bar color for G channel", value="#2ca02c", key="colG_hex")
    colB_hex = st.color_picker("Bar color for B channel", value="#1f77b4", key="colB_hex")


chR, chG, chB = get_mapping_channels(mapping_preset, cR, cG, cB)


# -----------------------------
# Session_state keys (apply recipe)
# -----------------------------
DYE_NAMES = list(DYE_LIBRARY.keys())

def ss_key_conc(sample_i, dye_j):
    return f"conc_{sample_i}_{dye_j}"

def ss_key_K(sample_i):
    return f"K_{sample_i}"

def ss_key_W(sample_i):
    return f"W_{sample_i}"

def ss_key_B(sample_i):
    return f"B_{sample_i}"

def apply_recipe_to_samples(recipe, sampleA=0, sampleB=1):
    for j, dye in enumerate(DYE_NAMES):
        st.session_state[ss_key_conc(sampleA, j)] = float(recipe["A_concs"].get(dye, 0.0))
        st.session_state[ss_key_conc(sampleB, j)] = float(recipe["B_concs"].get(dye, 0.0))
    if "A_optics" in recipe:
        st.session_state[ss_key_K(sampleA)] = float(recipe["A_optics"]["K"])
        st.session_state[ss_key_W(sampleA)] = float(recipe["A_optics"]["W"])
        st.session_state[ss_key_B(sampleA)] = float(recipe["A_optics"]["B"])
    if "B_optics" in recipe:
        st.session_state[ss_key_K(sampleB)] = float(recipe["B_optics"]["K"])
        st.session_state[ss_key_W(sampleB)] = float(recipe["B_optics"]["W"])
        st.session_state[ss_key_B(sampleB)] = float(recipe["B_optics"]["B"])


# -----------------------------
# Build samples UI
# -----------------------------
samples = []
for i in range(int(n_samples)):
    name = f"Agar_{i+1}"
    with st.expander(f"{name} — dye mix", expanded=(i < 2)):
        st.caption("Concentration unit: mg/mL. If you only know relative, keep it consistent across samples.")
        dye_concs = {}
        cols = st.columns(2)

        for j, dye in enumerate(DYE_NAMES):
            with cols[j % 2]:
                if ss_key_conc(i, j) not in st.session_state:
                    default_val = 0.0
                    if i == 0 and "Yellow No.4" in dye:
                        default_val = 0.02
                    if i == 1 and "Blue No.1" in dye:
                        default_val = 0.02
                    st.session_state[ss_key_conc(i, j)] = default_val

                dye_concs[dye] = st.number_input(
                    f"{dye} conc (mg/mL)",
                    min_value=0.0, max_value=20.0,
                    value=float(st.session_state[ss_key_conc(i, j)]),
                    step=0.01,
                    key=ss_key_conc(i, j)
                )

        if ss_key_K(i) not in st.session_state:
            st.session_state[ss_key_K(i)] = float(K_default)
        if ss_key_W(i) not in st.session_state:
            st.session_state[ss_key_W(i)] = float(whitening_default)
        if ss_key_B(i) not in st.session_state:
            st.session_state[ss_key_B(i)] = float(bright_default)

        K = st.number_input(
            "K (override; leave default if unsure)",
            0.0, 500.0,
            value=float(st.session_state[ss_key_K(i)]),
            step=1.0,
            key=ss_key_K(i)
        )
        whitening = st.slider(
            "Whitening / scattering (override)",
            0.0, 1.0,
            value=float(st.session_state[ss_key_W(i)]),
            step=0.01,
            key=ss_key_W(i)
        )

        if lock_brightness:
            brightness = float(bright_default)
            st.info(f"Brightness locked to global value: {brightness:.2f}")
            st.session_state[ss_key_B(i)] = float(brightness)
        else:
            brightness = st.slider(
                "Brightness factor (override)",
                0.1, 3.0,
                value=float(st.session_state[ss_key_B(i)]),
                step=0.01,
                key=ss_key_B(i)
            )

    samples.append(dict(name=name, dye_concs=dye_concs, K=K, whitening=whitening, brightness=brightness))


# -----------------------------
# Compute spectra for current samples
# -----------------------------
white_stim = illum.copy()

stim_list, refl_list = [], []
for s in samples:
    R_model = agar_reflectance(WL_REF, s["dye_concs"], K=s["K"], whitening=s["whitening"])
    R = apply_material_effect(WL_REF, R_model, R_base_curve, T_curve)
    stim = stimulus_spd(WL_REF, illum, R, brightness_factor=s["brightness"])
    refl_list.append(R)
    stim_list.append(stim)

# Stimulus spectra plot
st.subheader("Stimulus spectra (relative radiance)")
df_spec = pd.DataFrame({"Wavelength (nm)": WL_REF})
for s, stim in zip(samples, stim_list):
    df_spec[s["name"]] = stim / (np.max(stim) if np.max(stim) > 0 else 1.0)
st.line_chart(df_spec.set_index("Wavelength (nm)"))

# Human swatches
st.subheader("Predicted human swatches (approx, for communication)")
XYZs = [integrate_xyz(WL_REF, stim) for stim in stim_list]
human_hexes = []
if human_mode == "Brightness preserve":
    Ys = [float(x[1]) for x in XYZs]
    maxY = max(Ys) if max(Ys) > 0 else 1.0
    for XYZ in XYZs:
        XYZn = XYZ / maxY
        human_hexes.append(rgb_to_hex(xyz_to_srgb(XYZn)))
else:
    for XYZ in XYZs:
        XYZn = normalize_xyz_brightness(XYZ, target_Y=1.0)
        human_hexes.append(rgb_to_hex(xyz_to_srgb(XYZn)))

cols = st.columns(len(samples))
for i, (col, s) in enumerate(zip(cols, samples)):
    with col:
        st.markdown(f"**{s['name']}**")
        st.color_picker(" ", value=human_hexes[i], key=f"human_{i}", disabled=True)
        st.code(human_hexes[i], language="text")

# Fly catches
st.subheader("Fly receptor catches (Von Kries normalized)")
white_c = receptor_catches(WL_REF, white_stim, DROS_RECEPTORS)
fly_raw = [receptor_catches(WL_REF, stim, DROS_RECEPTORS) for stim in stim_list]
fly_vk = [von_kries_normalize(c, white_c) for c in fly_raw]
df_catch = pd.DataFrame([{"Sample": s["name"], **fly_vk[i]} for i, s in enumerate(samples)])
st.dataframe(df_catch, use_container_width=True)

# ΔS / ΔL matrices
st.subheader("Fly discriminability (ΔS) and luminance proxies (L_fly & Rh1)")
names = [s["name"] for s in samples]
n = len(samples)
deltaS = np.zeros((n, n), dtype=float)
deltaL_rh1 = np.zeros((n, n), dtype=float)
deltaL_fly = np.zeros((n, n), dtype=float)

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        deltaS[i, j] = rnl_delta_s(fly_vk[i], fly_vk[j])
        deltaL_rh1[i, j] = luminance_delta_rel(fly_vk[i], fly_vk[j], lum_ch="Rh1")
        Li = fly_Lsum(fly_vk[i], channels=("Rh3", "Rh4", "Rh5", "Rh6"))
        Lj = fly_Lsum(fly_vk[j], channels=("Rh3", "Rh4", "Rh5", "Rh6"))
        deltaL_fly[i, j] = delta_rel(Li, Lj)

df_deltaS = pd.DataFrame(deltaS, index=names, columns=names)
df_deltaL1 = pd.DataFrame(deltaL_rh1, index=names, columns=names)
df_deltaLf = pd.DataFrame(deltaL_fly, index=names, columns=names)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**ΔS matrix (RNL; Rh3–Rh6)**")
    st.dataframe(df_deltaS.style.format("{:.2f}"), use_container_width=True)
with c2:
    st.markdown("**ΔL_fly_rel matrix (L_fly=Rh3+Rh4+Rh5+Rh6)**")
    st.dataframe(df_deltaLf.style.format("{:.3f}"), use_container_width=True)
with c3:
    st.markdown("**Rh1 ΔL_rel matrix (safety proxy)**")
    st.dataframe(df_deltaL1.style.format("{:.3f}"), use_container_width=True)

st.caption(
    "Design hint (for learning tasks): aim ΔS ~ 1.5–3, keep ΔL_fly_rel small (e.g., <0.05–0.10). "
    "Optionally also keep Rh1 ΔL_rel small (e.g., <0.10–0.15) to reduce brightness/contrast shortcuts."
)

# Pseudo-fly display: A (bars) + B (desaturated swatch)
st.subheader("Pseudo-fly false-color (for intuition / comparison)")
st.caption(
    f"Mapping: R={chR}, G={chG}, B={chB}. "
    f"{'Global normalization.' if norm_mode.startswith('Global') else 'Per-sample normalization.'}"
)

scale_max = compute_global_scale_max(fly_vk, chR, chG, chB) if norm_mode.startswith("Global") else None

# bar style mapping (channel -> color)
CH_COLORS = {chR: colR_hex, chG: colG_hex, chB: colB_hex}

for idx, (s, c) in enumerate(zip(samples, fly_vk)):
    st.markdown(f"### {s['name']}")
    left, right = st.columns([2.2, 1.0])

    with left:
        bar_channels = [chR, chG, chB]
        vals = np.array([float(c[ch]) for ch in bar_channels], dtype=float)
        vmax = float(np.max(vals)) if np.max(vals) > 0 else 1.0

        df_bar = pd.DataFrame({
            "Channel": bar_channels,
            "Relative": (vals / vmax),
        })

        chart = alt_channel_bar(
            df_bar,
            bar_channels=bar_channels,
            ch_colors=CH_COLORS,
            bar_width_px=int(bar_width_px),
            height_px=180
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        vec = np.array([float(c[chR]), float(c[chG]), float(c[chB])], dtype=float)
        if norm_mode.startswith("Global"):
            hx = pseudo_hex_from_vec(vec, scale_max, desat_amount=swatch_desat)
        else:
            per_max = max(float(np.max(vec)), 1e-12)
            hx = pseudo_hex_from_vec(vec, per_max, desat_amount=swatch_desat)
        st.markdown("B) Presentation-safe swatch (desaturated false color)")
        st.color_picker(" ", value=hx, key=f"pf_{idx}", disabled=True)
        st.code(hx, language="text")

st.caption(
    "Note: pseudo-fly swatches are false-color aids. For conclusions, rely on receptor catches + ΔS and luminance constraints."
)


# -----------------------------
# Recipe Finder (Agar_1 vs Agar_2)
# -----------------------------
st.subheader("Recipe Finder (Agar_1 vs Agar_2)")
st.caption(
    "Random search proposes recipes meeting ΔS and isoluminance constraints. "
    "Primary luminance constraint: ΔL_fly_rel where L_fly = Rh3+Rh4+Rh5+Rh6."
)

with st.expander("Find recipes", expanded=True):
    if int(n_samples) < 2:
        st.warning("Need at least 2 samples (Agar_1 and Agar_2).")
    else:
        colA, colB, colC = st.columns(3)
        with colA:
            allowed_dyes = st.multiselect("Allowed dyes", options=DYE_NAMES, default=DYE_NAMES, key="allowed_dyes")
            conc_min = st.number_input("Concentration min (mg/mL)", 0.0, 20.0, 0.00, 0.01, key="conc_min")
            conc_max = st.number_input("Concentration max (mg/mL)", 0.0, 20.0, 0.20, 0.01, key="conc_max")
            p_zero = st.slider("Sparsity (prob dye=0)", 0.0, 1.0, 0.60, 0.01, key="p_zero")

        with colB:
            target_dS_min = st.number_input("ΔS min", 0.0, 20.0, 1.50, 0.10, key="target_dS_min")
            target_dS_max = st.number_input("ΔS max (soft)", 0.0, 50.0, 6.00, 0.10, key="target_dS_max")
            dS_mid = st.number_input("ΔS preferred center", 0.0, 20.0, 2.50, 0.10, key="dS_mid")

            dLf_max = st.number_input("ΔL_fly_rel max (primary)", 0.0, 1.0, 0.08, 0.01, key="dLf_max")
            use_rh1_guard = st.checkbox("Also enforce Rh1 ΔL_rel max (safety)", value=True, key="use_rh1_guard")
            dL1_max = st.number_input(
                "Rh1 ΔL_rel max (safety)",
                0.0, 1.0, 0.12, 0.01,
                disabled=not use_rh1_guard,
                key="dL1_max"
            )

        with colC:
            n_iter = st.number_input("Search iterations", 100, 200000, 8000, 100, key="n_iter")
            n_keep = st.number_input("Return top N recipes", 1, 200, 15, 1, key="n_keep")
            use_current_optics = st.checkbox("Use current Agar_1/Agar_2 optics (K/W/B)", value=True, key="use_current_optics")

        if use_current_optics:
            A_opt = dict(K=float(samples[0]["K"]), W=float(samples[0]["whitening"]), B=float(samples[0]["brightness"]))
            B_opt = dict(K=float(samples[1]["K"]), W=float(samples[1]["whitening"]), B=float(samples[1]["brightness"]))
        else:
            A_opt = dict(K=float(K_default), W=float(whitening_default), B=float(bright_default))
            B_opt = dict(K=float(K_default), W=float(whitening_default), B=float(bright_default))

        allowed_set = set(allowed_dyes)

        def sample_concs():
            d = {dn: 0.0 for dn in DYE_NAMES}
            for dn in allowed_set:
                if np.random.rand() < float(p_zero):
                    d[dn] = 0.0
                else:
                    d[dn] = float(np.random.uniform(float(conc_min), float(conc_max)))
            return d

        def eval_recipe(A_concs, B_concs):
            RmA = agar_reflectance(WL_REF, A_concs, K=A_opt["K"], whitening=A_opt["W"])
            RmB = agar_reflectance(WL_REF, B_concs, K=B_opt["K"], whitening=B_opt["W"])
            RA = apply_material_effect(WL_REF, RmA, R_base_curve, T_curve)
            RB = apply_material_effect(WL_REF, RmB, R_base_curve, T_curve)

            stimA = stimulus_spd(WL_REF, illum, RA, brightness_factor=A_opt["B"])
            stimB = stimulus_spd(WL_REF, illum, RB, brightness_factor=B_opt["B"])

            cA = von_kries_normalize(receptor_catches(WL_REF, stimA, DROS_RECEPTORS), white_c)
            cB = von_kries_normalize(receptor_catches(WL_REF, stimB, DROS_RECEPTORS), white_c)

            dS = rnl_delta_s(cA, cB)

            LA = fly_Lsum(cA, channels=("Rh3", "Rh4", "Rh5", "Rh6"))
            LB = fly_Lsum(cB, channels=("Rh3", "Rh4", "Rh5", "Rh6"))
            dLf = delta_rel(LA, LB)

            dL1 = luminance_delta_rel(cA, cB, lum_ch="Rh1")

            XYZ_A = integrate_xyz(WL_REF, stimA)
            XYZ_B = integrate_xyz(WL_REF, stimB)
            if human_mode == "Brightness preserve":
                YA, YB = float(XYZ_A[1]), float(XYZ_B[1])
                maxY = max(YA, YB, 1e-12)
                hA = rgb_to_hex(xyz_to_srgb(XYZ_A / maxY))
                hB = rgb_to_hex(xyz_to_srgb(XYZ_B / maxY))
            else:
                hA = rgb_to_hex(xyz_to_srgb(normalize_xyz_brightness(XYZ_A, 1.0)))
                hB = rgb_to_hex(xyz_to_srgb(normalize_xyz_brightness(XYZ_B, 1.0)))

            vecA = np.array([float(cA[chR]), float(cA[chG]), float(cA[chB])], dtype=float)
            vecB = np.array([float(cB[chR]), float(cB[chG]), float(cB[chB])], dtype=float)
            pair_scale = max(float(np.max(vecA)), float(np.max(vecB)), 1e-12)
            pfA = pseudo_hex_from_vec(vecA, pair_scale, desat_amount=swatch_desat)
            pfB = pseudo_hex_from_vec(vecB, pair_scale, desat_amount=swatch_desat)

            score = (
                abs(float(dS) - float(dS_mid))
                + 3.0 * float(dLf)
                + (1.5 * float(dL1) if use_rh1_guard else 0.0)
                + 0.10 * max(0.0, float(dS) - float(target_dS_max))
            )

            return dict(
                dS=float(dS), dLf=float(dLf), dL1=float(dL1),
                hA=hA, hB=hB, pfA=pfA, pfB=pfB,
                cA=cA, cB=cB,
                score=float(score)
            )

        run = st.button("Run recipe search", key="run_recipe_search")
        if run:
            np.random.seed()
            hits = []
            for _ in range(int(n_iter)):
                A_concs = sample_concs()
                B_concs = sample_concs()

                res = eval_recipe(A_concs, B_concs)
                if res["dS"] < float(target_dS_min):
                    continue
                if res["dLf"] > float(dLf_max):
                    continue
                if use_rh1_guard and res["dL1"] > float(dL1_max):
                    continue

                hits.append(dict(
                    score=res["score"],
                    dS=res["dS"],
                    dLf=res["dLf"],
                    dL1=res["dL1"],
                    human_A=res["hA"],
                    human_B=res["hB"],
                    pseudo_A=res["pfA"],
                    pseudo_B=res["pfB"],
                    A_concs=A_concs,
                    B_concs=B_concs,
                    A_optics=A_opt,
                    B_optics=B_opt,
                    cA=res["cA"],
                    cB=res["cB"],
                ))

            if not hits:
                st.warning(
                    "No recipes met your constraints. Try: increase iterations, relax ΔL_fly_rel, "
                    "widen conc range, reduce sparsity, or (if UV unreliable) increase UV cut."
                )
            else:
                hits.sort(key=lambda x: x["score"])
                hits = hits[:int(n_keep)]
                st.success(f"Found {len(hits)} recipes meeting constraints (showing top {len(hits)}).")

                rows = []
                for i, h in enumerate(hits):
                    row = {
                        "rank": i + 1,
                        "score": h["score"],
                        "ΔS": h["dS"],
                        "ΔL_fly_rel": h["dLf"],
                        "Rh1 ΔL_rel": h["dL1"],
                        "human_A": h["human_A"],
                        "human_B": h["human_B"],
                        "pseudo_A": h["pseudo_A"],
                        "pseudo_B": h["pseudo_B"],
                    }
                    for dn in allowed_dyes:
                        row[f"A:{dn}"] = h["A_concs"].get(dn, 0.0)
                        row[f"B:{dn}"] = h["B_concs"].get(dn, 0.0)
                    rows.append(row)

                df_hits = pd.DataFrame(rows)
                st.dataframe(df_hits, use_container_width=True)

                csv_bytes = df_hits.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download recipes CSV",
                    data=csv_bytes,
                    file_name="flycolor_recipes.csv",
                    mime="text/csv",
                    key="dl_recipes_csv"
                )

                st.markdown("---")
                st.markdown("## Top recipes (detail view + Apply)")
                for i, h in enumerate(hits):
                    with st.expander(
                        f"Recipe {i+1}  |  ΔS={h['dS']:.2f}, ΔL_fly_rel={h['dLf']:.3f}, Rh1 ΔL_rel={h['dL1']:.3f}, score={h['score']:.2f}",
                        expanded=(i < 3)
                    ):
                        cL, cRr = st.columns([2.0, 1.2])

                        with cL:
                            st.markdown("### Concentrations (mg/mL)")
                            df_ab = pd.DataFrame({
                                "Dye": allowed_dyes,
                                "Agar_1": [h["A_concs"].get(dn, 0.0) for dn in allowed_dyes],
                                "Agar_2": [h["B_concs"].get(dn, 0.0) for dn in allowed_dyes],
                            })
                            st.dataframe(df_ab, use_container_width=True)

                            st.markdown("### Receptor pattern (mapped channels)")
                            bar_ch = [chR, chG, chB]
                            valsA = np.array([float(h["cA"][ch]) for ch in bar_ch], dtype=float)
                            valsB = np.array([float(h["cB"][ch]) for ch in bar_ch], dtype=float)
                            vmaxA = float(np.max(valsA)) if np.max(valsA) > 0 else 1.0
                            vmaxB = float(np.max(valsB)) if np.max(valsB) > 0 else 1.0

                            df_barA = pd.DataFrame({"Channel": bar_ch, "Relative": valsA / vmaxA})
                            df_barB = pd.DataFrame({"Channel": bar_ch, "Relative": valsB / vmaxB})

                            sub1, sub2 = st.columns(2)
                            with sub1:
                                st.markdown("**Agar_1**")
                                st.altair_chart(
                                    alt_channel_bar(
                                        df_barA,
                                        bar_channels=bar_ch,
                                        ch_colors=CH_COLORS,
                                        bar_width_px=int(bar_width_px),
                                        height_px=180
                                    ),
                                    use_container_width=True
                                )
                            with sub2:
                                st.markdown("**Agar_2**")
                                st.altair_chart(
                                    alt_channel_bar(
                                        df_barB,
                                        bar_channels=bar_ch,
                                        ch_colors=CH_COLORS,
                                        bar_width_px=int(bar_width_px),
                                        height_px=180
                                    ),
                                    use_container_width=True
                                )

                        with cRr:
                            st.markdown("### Swatches")
                            st.markdown("**Human (approx)**")
                            st.color_picker("Agar_1 human", value=h["human_A"], key=f"hA_{i}", disabled=True)
                            st.color_picker("Agar_2 human", value=h["human_B"], key=f"hB_{i}", disabled=True)
                            st.markdown("**Pseudo-fly (presentation-safe)**")
                            st.color_picker("Agar_1 pseudo", value=h["pseudo_A"], key=f"pA_{i}", disabled=True)
                            st.color_picker("Agar_2 pseudo", value=h["pseudo_B"], key=f"pB_{i}", disabled=True)

                        if st.button("Apply this recipe to current Agar_1/Agar_2", key=f"apply_{i}"):
                            apply_recipe_to_samples(
                                recipe=dict(
                                    A_concs=h["A_concs"],
                                    B_concs=h["B_concs"],
                                    A_optics=h["A_optics"],
                                    B_optics=h["B_optics"]
                                ),
                                sampleA=0, sampleB=1
                            )
                            st.success("Applied. Re-running with these concentrations…")
                            st.rerun()


# -----------------------------
# Debug / transparency
# -----------------------------
with st.expander("Debug: illuminant, material curves, reflectance", expanded=False):
    st.markdown("### Illuminant (after optional UV cut)")
    df_illum = pd.DataFrame({"Wavelength (nm)": WL_REF, "Illuminant (rel)": illum})
    st.line_chart(df_illum.set_index("Wavelength (nm)"))

    st.markdown("### Material curves (applied)")
    df_mat = pd.DataFrame({
        "Wavelength (nm)": WL_REF,
        "R_base(λ)": R_base_curve,
        "T(λ)": T_curve,
    })
    st.line_chart(df_mat.set_index("Wavelength (nm)"))

    if wl_led_raw is not None and spd_uploaded_raw is not None:
        st.markdown("### Uploaded SPD (raw) — first rows")
        df_up = pd.DataFrame({"wl_raw": wl_led_raw, "spd_raw": spd_uploaded_raw})
        st.dataframe(df_up.head(30), use_container_width=True)

    if use_Rbase_csv and wl_Rb_raw is not None and Rb_raw is not None:
        st.markdown("### Uploaded R_base(λ) (raw) — first rows")
        df_rb = pd.DataFrame({"wl_raw": wl_Rb_raw, "R_base_raw": Rb_raw})
        st.dataframe(df_rb.head(30), use_container_width=True)

    if use_T_csv and wl_T_raw is not None and T_raw is not None:
        st.markdown("### Uploaded T(λ) (raw) — first rows")
        df_t = pd.DataFrame({"wl_raw": wl_T_raw, "T_raw": T_raw})
        st.dataframe(df_t.head(30), use_container_width=True)

    st.markdown("### Effective reflectance curves (after material optics)")
    for s, R in zip(samples, refl_list):
        st.markdown(f"**{s['name']} effective reflectance**")
        dfR = pd.DataFrame({"Wavelength (nm)": WL_REF, "Reflectance": R})
        st.line_chart(dfR.set_index("Wavelength (nm)"))

st.success("Loaded. For higher accuracy: replace dye spectra + receptor sensitivities with measured/literature curves.")
