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
       mrm_transmission.png  -> 穿透率 vs 波長（自動放大到凹谷附近，含指標標註）
       mrm_transmission_full.png -> 穿透率 vs 波長（完整掃描範圍總覽）
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
    GAP=0.20,         # 直波導與環邊緣之間的間隙 (µm)；加大 -> 耦合變弱、凹谷更深更窄
    N_SI=2.8,         # 矽的等效折射率（2D 近似）
    N_CLAD=1.44,      # 包覆層 SiO2 的折射率
    WVL_MIN=1.535,    # 掃描波長下限 (µm)：只涵蓋 OFF 與 ON 的那一組共振
    WVL_MAX=1.555,    # 掃描波長上限 (µm)：範圍要小於 FSR（約 41~44 nm），避免抓到別的共振
    NFREQ=600,       # 頻率取樣點數（20 nm / 1200 ≈ 0.017 nm 間隔）
    RESOLUTION=25,    # 每 µm 的像素數（40 -> 25 nm/像素）
    DPML=1.0,         # PML 吸收邊界厚度 (µm)
    DN_ON=-0.004,     # 電壓 ON 時的折射率變化 Δn（示意值，之後用 Soref 公式算）
    DALPHA_ON=0.0,    # 電壓 ON 時的額外吸收 Δα (cm^-1)；先設 0，只看折射率造成的純位移
    DECAY=1e-5,       # 場衰減到多小才停止模擬（越小越準，但跑越久）
)

# 穿透率放大圖的 x 軸範圍：兩個凹谷外側各多留幾倍 FWHM（越小畫面越放大）
PLOT_ZOOM = 1.5


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
    if t_max - t_min < 1e-9:                    # 光譜完全是平的：沒有凹谷可量
        return dict(lam0=lam0, fwhm=np.nan, Q=np.nan, ER=0.0)
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
# 4b. 有效性檢查與窗口追蹤（逆向設計時，評估函數必須穩健）
# ============================================================
# 為什麼需要：逆向設計會用成千上萬組不同參數呼叫 evaluate。
# 參數一改，共振位置就會移動（例如 R 變 1.7%，共振約移 26 nm），
# 可能跑出掃描窗口、或窗口裡出現兩個共振、或模擬沒跑到收斂。
# 這些情況下 analyze_dip 仍會「算出數字」，但數字是錯的，會誤導最佳化器。
# 做法：每組結果先檢查，不合格就標記為無效，能自動修的就自動修（移窗口、加嚴 DECAY）。

# 錯誤代碼 -> 中文說明（方便記錄與除錯）
REASONS = {
    "ok":            "ok",
    "not_converged": "T>1，模擬尚未收斂（DECAY 太鬆）",
    "no_dip":        "沒有明顯凹谷（共振可能完全在窗口外，或耦合太弱；可用 center_um 預測窗口）",
    "edge_low":      "凹谷靠近窗口短波長端（共振可能在窗口外）",
    "edge_high":     "凹谷靠近窗口長波長端（共振可能在窗口外）",
    "incomplete":    "凹谷不完整，量不到半高寬",
    "multi":         "窗口內有多個共振（請縮小 WVL 範圍或改 R）",
}


def check_dip(wvl, T, min_er_db=0.2, edge_frac=0.05, max_T=1.02, multi_frac=0.3):
    """
    檢查一條穿透率曲線的凹谷是否「可信」。回傳錯誤代碼（見 REASONS），"ok" 代表合格。
    wvl 必須由小到大排序。檢查順序很重要：先排除模擬本身的問題，再看凹谷。

    min_er_db : 凹谷深度（dB）低於此值視為「沒有凹谷」。0.2 dB 約等於深度 4.5%，
                高於一般收斂後的雜訊（<1%）。想嚴格一點可調高。
    edge_frac : 凹谷最低點落在窗口兩端各 5% 以內，視為在邊緣（可能被切掉）。
    max_T     : 穿透率最大值超過此值（理論上不超過 1），視為尚未收斂。
    multi_frac: 比「最亮 - 30% 深度」更低的區段超過一段，視為有多個共振。
    """
    n = len(wvl)
    i = int(np.argmin(T))                       # 凹谷最低點位置
    t_min, t_max = T[i], np.max(T)

    # 1. 模擬品質：T 明顯大於 1 -> 環內的光還沒漏完就停了
    if t_max > max_T:
        return "not_converged"

    # 2. 有沒有凹谷：深度太淺就是雜訊（會被誤判成超高 Q）
    er_db = 10 * np.log10(t_max / max(t_min, 1e-12))
    if er_db < min_er_db:
        return "no_dip"

    # 3. 凹谷在不在邊緣：在邊緣代表共振可能在窗口外
    if i < edge_frac * n:
        return "edge_low"
    if i > (1 - edge_frac) * n:
        return "edge_high"

    # 4. 半高寬量不量得到（兩側都要越過半深度）
    if np.isnan(analyze_dip(wvl, T)["fwhm"]):
        return "incomplete"

    # 5. 有幾個凹谷：數「低於門檻」的連續區段
    level = t_max - multi_frac * (t_max - t_min)
    below = (T < level).astype(int)
    n_dips = below[0] + int(np.sum(np.diff(below) == 1))
    if n_dips > 1:
        return "multi"

    return "ok"


def set_window(p, center_um=None, shift_um=0.0):
    """
    回傳新的參數 dict，掃描窗口寬度不變，但平移或重設中心。
      center_um : 指定新的窗口中心 (µm)
      shift_um  : 在現有位置上平移多少 (µm)
    """
    q = dict(p)
    width = p["WVL_MAX"] - p["WVL_MIN"]
    center = (p["WVL_MIN"] + p["WVL_MAX"]) / 2 if center_um is None else center_um
    center += shift_um
    q["WVL_MIN"], q["WVL_MAX"] = center - width / 2, center + width / 2
    return q


def predict_center(R_new, R_ref, lam_ref_um):
    """
    用已知的一組結果預測新 R 的共振位置：共振條件 m·λ = n_eff·2πR，
    同一個 m 下 λ 與 R 成正比。這只是一階近似（沒考慮 W、GAP 的影響），
    殘餘誤差交給 adjust_for_retry 的窗口追蹤處理。
    """
    return lam_ref_um * R_new / R_ref


def adjust_for_retry(p, wvl, T_off, T_on, c_off, c_on):
    """
    根據兩條光譜的錯誤代碼，嘗試自動修正參數。
    回傳 (新參數, 說明文字)；說明為 None 代表無法自動修正，呼叫端應放棄重試。
    """
    q = dict(p)
    window_codes = {"edge_low", "edge_high", "incomplete"}

    # --- 優先處理窗口位置：把窗口朝凹谷所在的那一側平移 ---
    if {c_off, c_on} & window_codes:
        center = (p["WVL_MIN"] + p["WVL_MAX"]) / 2
        width = p["WVL_MAX"] - p["WVL_MIN"]
        sides = set()                           # 凹谷偏向哪一側（+1 長波長 / -1 短波長）
        for code, T in ((c_off, T_off), (c_on, T_on)):
            if code in window_codes:
                sides.add(int(np.sign(wvl[int(np.argmin(T))] - center)))
        if len(sides) > 1:                      # OFF 偏一邊、ON 偏另一邊：窗口根本太窄
            return p, None
        side = sides.pop()
        # 平移 0.6 倍窗口寬度：原本在邊緣的凹谷會移到靠近窗口中央
        return set_window(p, shift_um=side * 0.6 * width), \
            f"窗口平移 {side * 0.6 * width * 1000:+.1f} nm"

    # --- 其次處理收斂：把 DECAY 縮小 100 倍（跑得更久但更準）---
    if "not_converged" in {c_off, c_on} and p["DECAY"] > 1e-10:
        q["DECAY"] = max(p["DECAY"] * 0.01, 1e-10)
        return q, f"DECAY 加嚴為 {q['DECAY']:.0e}"

    # --- no_dip、multi 等無法靠平移修正 ---
    return p, None


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
    """
    畫穿透率 vs 波長（OFF / ON）。
    上圖：自動放大到兩個凹谷附近，才看得出位移與深度差異。
    下圖：ΔT = T_on - T_off，亮暗差最大的地方就是雷射最佳位置。
    """
    w_nm = wvl * 1000                                        # µm -> nm
    l_off, l_on = m_off["lam0"] * 1000, m_on["lam0"] * 1000  # 兩個共振波長 (nm)

    # --- 決定放大範圍：兩個共振各往外延伸 3 倍 FWHM（量不到 FWHM 就用 2 nm）---
    fw = np.nanmax([m_off["fwhm"], m_on["fwhm"]]) * 1000
    pad = 3 * fw if np.isfinite(fw) else 2.0
    lo, hi = min(l_off, l_on) - pad, max(l_off, l_on) + pad
    keep = (w_nm >= lo) & (w_nm <= hi)                       # 範圍內的資料點
    y_low = min(T_off[keep].min(), T_on[keep].min())

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.3]))

    # --- 上圖：穿透率 ---
    ax.plot(w_nm, T_off, color="tab:red", lw=1.8, label="Voltage OFF")
    ax.plot(w_nm, T_on, color="tab:green", lw=1.8, ls="--", label="Voltage ON")
    ax.axvline(l_off, color="tab:red", ls=":", lw=0.8)       # OFF 共振位置
    ax.axvline(l_on, color="tab:green", ls=":", lw=0.8)      # ON 共振位置
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(0.0, y_low - 0.05), 1.03)
    ax.set_ylabel("Transmission T")
    ax.set_title(f"Resonance shift = {l_on - l_off:+.2f} nm   "
                 f"(Q_off = {m_off['Q']:.0f}, Q_on = {m_on['Q']:.0f})", pad=30)
    ax.grid(alpha=0.3)

    # --- 下圖：ΔT = T_on - T_off ---
    dT = T_on - T_off
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.plot(w_nm, dT, color="tab:blue", lw=1.5)
    ax2.fill_between(w_nm, 0, dT, color="tab:blue", alpha=0.2)
    ax2.set_ylabel("ΔT = T_on − T_off")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.grid(alpha=0.3)

    # --- 雷射波長（若在放大範圍內才畫）---
    lam_L = mod["lam_L"] * 1000
    if lo <= lam_L <= hi:
        ax.axvline(lam_L, color="k", ls="-.", lw=1,
                   label=f"laser {lam_L:.2f} nm (OMA = {mod['OMA']:.2f})")
        ax2.axvline(lam_L, color="k", ls="-.", lw=1)

    # 圖例放在標題下方、圖框外，避免擋住凹谷
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


# ============================================================
# 6. 完整流程（包成函式，逆向設計直接呼叫 evaluate）
# ============================================================
# evaluate 回傳的欄位（無效時數值欄位為 nan，欄位固定，方便存成表格）
METRIC_KEYS = ("lam0_off_nm", "lam0_on_nm", "shift_nm", "fwhm_nm", "Q", "ER_dB",
               "OMA", "ER_mod_dB", "IL_dB", "lam_L_nm")


def _simulate_once(p, make_plots, ref_cache):
    """
    跑完「參考 + OFF + ON」三次模擬，回傳 (波長, T_off, T_on, 介電常數圖)。
    參考光譜只和直波導、解析度、掃描窗口有關，所以用 ref_cache 重用：
    只要這些沒變，就不用重跑。（R、GAP 只影響計算區域大小，
    對直波導的功率頻譜影響很小，視為可重用。）
    """
    key = (p["W"], p["N_SI"], p["N_CLAD"], p["RESOLUTION"], p["DPML"],
           p["NFREQ"], round(p["WVL_MIN"], 6), round(p["WVL_MAX"], 6))
    if key not in ref_cache:
        freqs, flux, _ = run_spectrum(p, with_ring=False)
        ref_cache[key] = (freqs, flux)
    freqs, ref_flux = ref_cache[key]

    # 電壓 OFF：環為原本的矽
    _, off_flux, eps = run_spectrum(p, True, si_medium(p["N_SI"]), want_eps=make_plots)
    # 電壓 ON：折射率改變 + 額外吸收
    _, on_flux, _ = run_spectrum(
        p, True, si_medium(p["N_SI"], p["DN_ON"], p["DALPHA_ON"]))

    # 穿透率 = 有環 / 參考；頻率轉波長後，排成由短到長
    wvl = 1 / freqs
    order = np.argsort(wvl)
    return (wvl[order], (off_flux / ref_flux)[order],
            (on_flux / ref_flux)[order], eps)


def evaluate(p, make_plots=False, ref_cache=None, center_um=None, max_retry=2):
    """
    給一組參數 p，回傳指標 dict。逆向設計時每組設計都呼叫這個函式。

    p          : 參數 dict（不會被修改）
    make_plots : True 時存圖與 npz（平常批次掃描請關掉）
    ref_cache  : 空 dict；多次呼叫時傳同一個，可重用參考光譜省時間
    center_um  : 預測的共振位置 (µm)，會把掃描窗口移到以它為中心（見 predict_center）
    max_retry  : 檢查不合格時，最多自動修正並重跑幾次

    回傳 dict 內固定有：
      valid  : True 才代表數字可信；False 時其他數值僅供除錯，不要拿去訓練
      reason : 無效原因（中文），有效時為 "ok"
      retries: 實際重試次數
      window_nm : 最後使用的掃描窗口 (nm)
    以及 METRIC_KEYS 列出的各項指標。
    """
    p = dict(p)                                 # 複製一份，避免改到呼叫者的參數
    if ref_cache is None:
        ref_cache = {}
    if center_um is not None:
        p = set_window(p, center_um=center_um)  # 窗口移到預測的共振位置

    for attempt in range(max_retry + 1):
        # --- 第一層：模擬失敗（當機、記憶體不足等）也要當成「無效」，不要讓批次中斷 ---
        try:
            wvl, T_off, T_on, eps = _simulate_once(p, make_plots, ref_cache)
        except Exception as e:                  # noqa: BLE001（刻意攔下所有錯誤）
            return _invalid_result(p, attempt, f"模擬失敗: {e}")

        # --- 第二層：檢查 OFF 與 ON 兩條光譜 ---
        c_off, c_on = check_dip(wvl, T_off), check_dip(wvl, T_on)
        if c_off == "ok" and c_on == "ok":
            break

        # --- 第三層：能自動修就修（平移窗口 / 加嚴 DECAY），修完重跑 ---
        if attempt == max_retry:
            break
        new_p, note = adjust_for_retry(p, wvl, T_off, T_on, c_off, c_on)
        if note is None:                        # 無法自動修正，放棄
            break
        if mp.am_master():
            print(f"[retry {attempt + 1}] OFF:{c_off} ON:{c_on} -> {note}")
        p = new_p

    # 組合最終的有效性與原因
    valid = (c_off == "ok" and c_on == "ok")
    reason = "ok" if valid else f"OFF: {REASONS[c_off]}；ON: {REASONS[c_on]}"

    # 算指標（無效時仍會算，但數字只供除錯）
    m_off, m_on = analyze_dip(wvl, T_off), analyze_dip(wvl, T_on)
    mod = modulator_metrics(wvl, T_off, T_on)
    metrics = dict(
        valid=valid, reason=reason, retries=attempt,
        window_nm=(p["WVL_MIN"] * 1e3, p["WVL_MAX"] * 1e3),
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
    return metrics


def _invalid_result(p, retries, reason):
    """模擬根本沒跑出結果時，回傳欄位一致的無效結果（數值全為 nan）。"""
    m = {k: np.nan for k in METRIC_KEYS}
    m.update(valid=False, reason=reason, retries=retries,
             window_nm=(p["WVL_MIN"] * 1e3, p["WVL_MAX"] * 1e3))
    return m


def figure_of_merit(metrics, penalty=-2.0):
    """
    把指標變成「一個分數」（越大越好），給最佳化器用。
    無效的設計回傳固定的懲罰分數（而不是 nan），讓最佳化器避開那一區；
    原因仍保留在 metrics["reason"]，之後才分得出「真的差」與「量不到」。

    下面的加權只是示範（OMA 越大越好、插入損耗越小越好），
    請依專題的實際目標修改，例如改成「ER 大於 10 dB 才算合格」。
    """
    if not metrics["valid"]:
        return penalty
    return metrics["OMA"] - 0.05 * metrics["IL_dB"]


def example_scan_R(R_list):
    """
    示範（main 不會自動執行）：掃描不同的環半徑 R，並用上一組有效結果
    預測下一組的共振位置，讓掃描窗口跟著共振走。
    用法：rows = example_scan_R([2.9, 3.0, 3.05])
    """
    cache, rows, last = {}, [], None            # last = (R, OFF 共振波長 µm)
    for R in R_list:
        p = dict(PARAMS, R=R)
        center = predict_center(R, *last) if last else None
        m = evaluate(p, ref_cache=cache, center_um=center)
        m["R"], m["score"] = R, figure_of_merit(m)
        rows.append(m)
        if m["valid"]:                          # 只有可信的結果才拿來當下一次的預測基準
            last = (R, m["lam0_off_nm"] / 1e3)
        print(f"R={R}: valid={m['valid']}, reason={m['reason']}")
    return rows


def main():
    metrics = evaluate(PARAMS, make_plots=True)
    if mp.am_master():
        print("=== Micro-ring modulator metrics ===")
        for k, v in metrics.items():
            print(f"{k:>14s} : {v}")
        print("Saved: mrm_structure.png, mrm_transmission.png, mrm_spectra.npz")


if __name__ == "__main__":
    main()
