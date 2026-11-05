"""
micro_ring_modulator_meep.py
============================
用 Meep (FDTD) 模擬「微環調變器」(2D 等效折射率版本)。

做的事：
  1. 建立 直波導 + 環 的 2D 結構
  2. 跑三次模擬：
       (a) 參考：只有直波導（用來歸一化）
       (b) 電壓 OFF：環的折射率 = 原本的矽
       (c) 電壓 ON ：環的折射率 = 矽 + Δn（Soref 效應），並加一點吸收 Δα
  3. 輸出：
       mrm_structure.png     -> 2D 結構平面圖（介電常數分布）
       mrm_transmission.png  -> 穿透率 vs 波長（OFF / ON）
       mrm_spectra.npz       -> 光譜資料（之後做逆向設計 / 訓練資料用）
  4. 印出指標：共振波長、FWHM、Q、ER、共振位移、OMA、IL

單位：Meep 的長度單位這裡設為 1 µm，所以波長 1.55 µm 對應頻率 1/1.55。

執行方式：
    python micro_ring_modulator_meep.py
    mpirun -np 4 python micro_ring_modulator_meep.py     # 平行加速

注意：這是 2D 近似（矽用等效折射率 n_eff），重點是流程與趨勢；
      要貼近真實元件請之後升級成 3D 或用 MPB 求真正的 n_eff。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")              # 沒有視窗的環境（伺服器 / WSL）也能存圖
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import meep as mp


# ============================================================
# 1. 參數（全部集中在這裡，之後逆向設計只要改這個 dict）
# ============================================================
PARAMS = dict(
    R=3.0,            # 環「中心線」半徑 (µm)
    W=0.5,            # 波導寬度 (µm)，環與直波導相同
    GAP=0.20,         # 直波導與環邊緣之間的間隙 (µm)
    N_SI=2.8,         # 矽的等效折射率（2D 近似）
    N_CLAD=1.44,      # 包覆層 SiO2 的折射率
    WVL_MIN=1.535,     # 掃描波長下限 (µm)
    WVL_MAX=1.555,     # 掃描波長上限 (µm)
    NFREQ=1200,        # 頻率取樣點數（越多光譜越細）
    RESOLUTION=60,    # 每 µm 的像素數（40 -> 25 nm/像素）
    DPML=1.5,         # PML 吸收邊界厚度 (µm)
    DN_ON=-0.004,     # 電壓 ON 時的折射率變化 Δn（示意值，之後用 Soref 公式算）
    DALPHA_ON=0,   # 電壓 ON 時的額外吸收 Δα (cm^-1)（示意值）
    DECAY=1e-5,       # 場衰減到多小才停止模擬（越小越準，但跑越久）
)


# ============================================================
# 2. 幾何與材料
# ============================================================
def cell_geometry(p):
    """回傳 (直波導 y 位置, 計算區域寬 sx, 高 sy)。環的圓心放在 (0, 0)。"""
    # 直波導中心到環中心的距離 = R + W/2 (環外緣) + GAP + W/2 (直波導半寬)
    bus_y = -(p["R"] + p["W"] + p["GAP"])
    sx = 2 * p["R"] + 6.0                                   # 左右留空間放光源與監測器
    sy = 2 * (abs(bus_y) + p["W"] / 2 + p["DPML"] + 0.5)    # 上下對稱，留出 PML
    return bus_y, sx, sy


def si_medium(n0, dn=0.0, dalpha_cm=0.0, wvl=1.55):
    """
    建立矽材料：折射率 n0 + dn，並加入吸收 dalpha_cm (cm^-1)。
    吸收用 Meep 的 D_conductivity 表示：
        alpha = 4*pi*kappa / wvl   ->  kappa = alpha*wvl/(4*pi)
        Im(eps) = 2*n*kappa
        sigma   = Im(eps) * omega / n^2      （omega = 2*pi/wvl）
    """
    n = n0 + dn
    alpha = dalpha_cm * 1e-4                    # cm^-1 -> µm^-1
    kappa = alpha * wvl / (4 * np.pi)           # 消光係數
    sigma = 2 * n * kappa * (2 * np.pi / wvl) / n ** 2
    return mp.Medium(index=n, D_conductivity=sigma)


def build_geometry(p, with_ring, ring_medium):
    """建立幾何物件清單：直波導，以及（選擇性）環。"""
    bus_y, _, _ = cell_geometry(p)
    clad = mp.Medium(index=p["N_CLAD"])
    bus_medium = mp.Medium(index=p["N_SI"])     # 直波導不受電壓影響，用原本的矽

    # 直波導：沿 x 方向無限延伸（會一路延伸進 PML，避免邊界反射）
    geometry = [mp.Block(size=mp.Vector3(mp.inf, p["W"], mp.inf),
                         center=mp.Vector3(0, bus_y),
                         material=bus_medium)]

    if with_ring:
        # 環 = 外圓（矽）減去內圓（包覆層）。後放的物件會蓋過先放的。
        geometry += [
            mp.Cylinder(radius=p["R"] + p["W"] / 2, material=ring_medium),
            mp.Cylinder(radius=p["R"] - p["W"] / 2, material=clad),
        ]
    return geometry


# ============================================================
# 3. 跑一次模擬，回傳各頻率的穿透功率
# ============================================================
def run_spectrum(p, with_ring, ring_medium=None, want_eps=False):
    """
    with_ring=False -> 參考模擬（只有直波導）
    with_ring=True  -> 有環，環的材料由 ring_medium 決定（OFF / ON 差在這）
    回傳 (頻率陣列, 穿透功率 flux 陣列, 介電常數圖或 None)
    """
    bus_y, sx, sy = cell_geometry(p)

    # --- 頻率設定 ---
    f_min, f_max = 1 / p["WVL_MAX"], 1 / p["WVL_MIN"]       # 波長 -> 頻率（大小會反過來）
    fcen = (f_min + f_max) / 2                               # 中心頻率
    df = f_max - f_min                                       # 監測的頻寬
    src_fwidth = 2.5 * df                                    # 光源頻寬要比監測範圍寬

    # --- 光源：直波導左端的寬頻高斯脈衝 ---
    # 這裡用最單純的 Ez 點源（沿波導截面鋪開），主要激發基模。
    # 之後可改用 mp.EigenModeSource 只激發乾淨的基模。
    src_x = -sx / 2 + p["DPML"] + 0.5
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=src_fwidth),
                         component=mp.Ez,
                         center=mp.Vector3(src_x, bus_y),
                         size=mp.Vector3(0, p["W"]))]

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        geometry=build_geometry(p, with_ring, ring_medium),
        sources=sources,
        boundary_layers=[mp.PML(p["DPML"])],
        resolution=p["RESOLUTION"],
        default_material=mp.Medium(index=p["N_CLAD"]),       # 背景 = 包覆層
    )

    # --- 監測器：直波導右端（through port），在環的耦合區之外 ---
    mon_x = sx / 2 - p["DPML"] - 0.5
    through = sim.add_flux(
        fcen, df, p["NFREQ"],
        mp.FluxRegion(center=mp.Vector3(mon_x, bus_y),
                      size=mp.Vector3(0, 3 * p["W"])))

    # --- 決定「什麼時候可以停」---
    # 有環：看環裡的場衰減完（共振光要等它漏光漏完，Q 越高越久）
    # 沒環：看直波導出口的場衰減完
    if with_ring:
        decay_pt = mp.Vector3(p["R"], 0)                     # 環右側的一個點（在矽內）
    else:
        decay_pt = mp.Vector3(mon_x, bus_y)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, decay_pt, p["DECAY"]))

    flux = np.array(mp.get_fluxes(through))
    freqs = np.array(mp.get_flux_freqs(through))

    # 取出介電常數分布，用來畫 2D 結構圖
    eps = None
    if want_eps:
        eps = sim.get_array(center=mp.Vector3(),
                            size=mp.Vector3(sx, sy),
                            component=mp.Dielectric)
    return freqs, flux, eps


# ============================================================
# 4. 從穿透率曲線抓指標（第六步）
# ============================================================
def analyze_dip(wvl, T):
    """
    在 T(wvl) 裡找最深的凹谷，回傳 λ0、FWHM、Q、ER(dB)。
    wvl 必須由小到大排序。若凹谷不完整（半高寬超出掃描範圍），FWHM/Q 為 nan。
    """
    i = int(np.argmin(T))                       # 凹谷最低點
    lam0, t_min, t_max = wvl[i], T[i], np.max(T)
    half = (t_max + t_min) / 2                  # 半深度（最亮與最暗的正中間）

    # 往左、往右走，直到穿透率回到半深度以上
    l = i
    while l > 0 and T[l] < half:
        l -= 1
    r = i
    while r < len(T) - 1 and T[r] < half:
        r += 1

    fwhm = np.nan
    if T[l] >= half and T[r] >= half:           # 兩邊都真的越過半深度才算有效
        # 線性內插求出越過半深度的精確波長
        wl = wvl[l] + (half - T[l]) * (wvl[l + 1] - wvl[l]) / (T[l + 1] - T[l])
        wr = wvl[r - 1] + (half - T[r - 1]) * (wvl[r] - wvl[r - 1]) / (T[r] - T[r - 1])
        fwhm = wr - wl

    return dict(lam0=lam0, fwhm=fwhm, Q=lam0 / fwhm,
                ER=10 * np.log10(t_max / max(t_min, 1e-12)))


def modulator_metrics(wvl, T_off, T_on):
    """
    把雷射放在「OFF 與 ON 差最大」的波長 λ_L，算調變器指標：
      OMA = |T_on - T_off|        調變振幅（越大越好）
      ER  = 10log10(亮/暗)         調變消光比 (dB)
      IL  = -10log10(亮)           插入損耗 (dB)（越小越好）
    """
    k = int(np.argmax(np.abs(T_on - T_off)))
    bright, dark = max(T_on[k], T_off[k]), min(T_on[k], T_off[k])
    return dict(lam_L=wvl[k], OMA=bright - dark,
                ER_mod=10 * np.log10(bright / max(dark, 1e-12)),
                IL=-10 * np.log10(max(bright, 1e-12)))


# ============================================================
# 5. 畫圖
# ============================================================
def plot_structure(eps, p, filename="mrm_structure.png"):
    """畫 2D 結構平面圖：顏色越深 = 折射率越高（矽）。"""
    bus_y, sx, sy = cell_geometry(p)
    fig, ax = plt.subplots(figsize=(7, 6))

    # eps 的軸順序是 (x, y)，imshow 要 (y, x)，所以轉置；origin="lower" 讓 y 往上
    im = ax.imshow(eps.T, origin="lower", cmap="Greys",
                   extent=[-sx / 2, sx / 2, -sy / 2, sy / 2])
    fig.colorbar(im, ax=ax, label=r"relative permittivity $\varepsilon$")

    # PML 內緣（虛線框內才是有效計算區）
    d = p["DPML"]
    ax.add_patch(Rectangle((-sx / 2 + d, -sy / 2 + d), sx - 2 * d, sy - 2 * d,
                           fill=False, ls=":", ec="gray", label="PML inner edge"))
    # 光源與監測器位置
    ax.axvline(-sx / 2 + d + 0.5, color="tab:red", ls="--", label="source")
    ax.axvline(sx / 2 - d - 0.5, color="tab:blue", ls="--", label="through monitor")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Micro-ring structure (R={p['R']} µm, W={p['W']} µm, "
                 f"gap={p['GAP']} µm)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_transmission(wvl, T_off, T_on, m_off, m_on, mod, filename="mrm_transmission.png"):
    """畫穿透率 vs 波長（OFF / ON），並標出共振與雷射位置。"""
    w_nm = wvl * 1000                                        # µm -> nm
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w_nm, T_off, color="tab:red", lw=1.8, label="Voltage OFF")
    ax.plot(w_nm, T_on, color="tab:green", lw=1.8, ls="--", label="Voltage ON")

    ax.axvline(m_off["lam0"] * 1000, color="tab:red", ls=":", lw=0.8)
    ax.axvline(m_on["lam0"] * 1000, color="tab:green", ls=":", lw=0.8)
    ax.axvline(mod["lam_L"] * 1000, color="k", ls="-.", lw=1, label="laser wavelength")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission T")
    ax.set_title("Through-port transmission of the micro-ring modulator")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


# ============================================================
# 6. 完整流程（包成函式，之後逆向設計可以直接呼叫 evaluate）
# ============================================================
def evaluate(p, make_plots=False, ref_flux=None):
    """
    給一組參數 p，回傳指標 dict 與參考光譜。
    ref_flux：若直波導沒變，可把上次的參考光譜傳進來，省掉一次模擬。
    """
    # (a) 參考模擬：只有直波導，用來歸一化
    if ref_flux is None:
        freqs, ref_flux, _ = run_spectrum(p, with_ring=False)
    # (b) 電壓 OFF：環為原本的矽
    freqs, off_flux, eps = run_spectrum(
        p, True, si_medium(p["N_SI"]), want_eps=make_plots)
    # (c) 電壓 ON：折射率改變 + 額外吸收
    freqs, on_flux, _ = run_spectrum(
        p, True, si_medium(p["N_SI"], p["DN_ON"], p["DALPHA_ON"]))

    # 穿透率 = 有環 / 參考；頻率轉波長後，排成由短到長
    wvl = 1 / freqs
    order = np.argsort(wvl)
    wvl = wvl[order]
    T_off = (off_flux / ref_flux)[order]
    T_on = (on_flux / ref_flux)[order]

    # 指標
    m_off, m_on = analyze_dip(wvl, T_off), analyze_dip(wvl, T_on)
    mod = modulator_metrics(wvl, T_off, T_on)
    metrics = dict(
        lam0_off_nm=m_off["lam0"] * 1e3, lam0_on_nm=m_on["lam0"] * 1e3,
        shift_nm=(m_on["lam0"] - m_off["lam0"]) * 1e3,
        fwhm_nm=m_off["fwhm"] * 1e3, Q=m_off["Q"], ER_dB=m_off["ER"],
        OMA=mod["OMA"], ER_mod_dB=mod["ER_mod"], IL_dB=mod["IL"],
        lam_L_nm=mod["lam_L"] * 1e3)

    # 只讓主程序存檔（用 MPI 平行時避免每個 rank 都寫檔）
    if make_plots and mp.am_master():
        plot_structure(eps, p)
        plot_transmission(wvl, T_off, T_on, m_off, m_on, mod)
        np.savez("mrm_spectra.npz", wvl_um=wvl, T_off=T_off, T_on=T_on)
    return metrics, ref_flux


def main():
    metrics, _ = evaluate(PARAMS, make_plots=True)
    if mp.am_master():
        print("=== Micro-ring modulator metrics ===")
        for k, v in metrics.items():
            print(f"{k:>14s} : {v:.4g}")
        print("Saved: mrm_structure.png, mrm_transmission.png, mrm_spectra.npz")


if __name__ == "__main__":
    main()
