"""
micro_ring_modulator_meep.py
============================
用 Meep (FDTD) 模擬「微環調變器」(2D 等效折射率版本)。

做的事：
  1. 建立 直波導 + 環 的 2D 結構
  2. 跑三次模擬：
       (a) 參考：只有直波導（用來歸一化）
       (b) 電壓 OFF（零偏壓）：環是摻雜矽（有基線吸收），接面兩側有一圈窄的空乏區
       (c) 電壓 ON（反向偏壓）：同一種摻雜矽，但空乏區變寬。空乏區內載子被掃光，
           折射率上升 Δn（Soref 效應）、吸收消失，共振往長波長移
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

# scipy 只用在洛倫茲擬合；沒安裝時擬合結果為 nan，其他功能照常運作
try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ============================================================
# 1. 參數（全部集中在這裡，之後逆向設計只要改這個 dict）
# ============================================================
# 單位約定：Meep 的長度單位在這裡設為 1 µm，所以下面所有「長度」都是 µm，
#           波長 1.55 µm 對應頻率 1/1.55（c = 1）。折射率、吸收係數另有標註單位。
PARAMS = dict(
    # ================= 幾何（設計參數，逆向設計要調的旋鈕）=================
    R=3.0,            # 環「中心線」半徑 (µm)。環的內緣 = R - W/2、外緣 = R + W/2。
                      #   R 越大：FSR 越小（FSR = λ²/(n_g·2πR)）、彎曲輻射損耗越小，但計算區域面積約隨 R² 增加、模擬越慢；
                      #   共振波長約與 R 成正比（同一個階數 m），R 改變 1% 共振約移 15 nm。
    W=0.5,            # 波導寬度 (µm)，環與直波導相同。影響 n_eff、是否單模，也決定空乏區可以占多大比例。
    GAP=0.20,         # 直波導邊緣到環外緣的最短距離 (µm)（不是中心距；中心距 = R + W + GAP）。
                      #   越小：光越容易耦合進環（耦合係數近似指數式變大），凹谷變深變寬、Q 變低；
                      #   太小會變成過耦合，凹谷反而變淺。解析度 40 時 gap 0.08 µm 只有約 3 個像素，不要再往下。
    N_SI=2.8,         # 矽的「等效折射率」（2D 近似，由有效折射率法對 220 nm 厚的矽板算出，見 derive_n_eff.py）。
                      #   真實矽的折射率約 3.48，2D 模擬不能直接用，因為垂直方向的限制要折算進這個值。
    N_CLAD=1.44,      # 包覆層 SiO2 的折射率（也是計算區域的背景材料）。

    # ================= 掃描窗口與取樣（數值參數）=================
    WVL_MIN=1.525,    # 掃描波長下限 (µm)：要涵蓋 OFF 與 ON 的共振（解析度 40、R=3 時 OFF 共振約 1535 nm，放在窗口中央）
    WVL_MAX=1.545,    # 掃描波長上限 (µm)：範圍要小於 FSR（約 41~44 nm），避免抓到別的共振；換解析度或 R 要重設
                      #   （共振位置會隨解析度漂移數 nm，所以窗口不能一直沿用；見 README 11.1）
    NFREQ=4000,       # 頻率取樣點數（20 nm / 4000 = 0.005 nm 間隔）；PN 位移只有約 0.1 nm，間隔要夠細。
                      #   點數多只增加記憶體，不增加模擬時間；太少會漏掉窄的凹谷。
    RESOLUTION=40,    # 每 µm 的像素數（40 -> 格點邊長 25 nm）；空乏區只有 0.06~0.12 µm，解析度太低會畫不準。
                      #   耗時約隨解析度的 2.8 次方增加；共振波長會隨解析度漂移（格點色散），所以同一批資料要固定。
    DPML=1.0,         # PML（完美匹配層，吸收邊界）厚度 (µm)。要大於約半個波長，太薄會有反射。

    # ================= PN 反向偏壓調變（設計參數）=================
    # 做法：環內有一圈 p-n 接面，接面兩側有「空乏區」（載子被掃光的區域）。
    #   OFF = 零偏壓，空乏區窄（W_DEPL_OFF）；ON = 反向偏壓，空乏區變寬（W_DEPL_ON）。
    #   空乏區內沒有載子：折射率比摻雜矽高 DN_DEPL（共振往長波長移），而且沒有自由載子吸收。
    #   Meep 的吸收不能是負的，所以「損耗變少」用「摻雜矽有基線吸收 ALPHA_DOPE、空乏區歸零」來表示。
    PN_FROM_PHYSICS=True,  # True：由下面的 V_REV、NA、ND、TAU_MP、TAU_MN 依「遷移率 -> 有效質量 -> 載子變化 -> Δn」
                           #   算出 DN_DEPL、W_DEPL_OFF、W_DEPL_ON（會蓋掉下面手動填的這三個值）；
                           #   False：直接用下面手動填的 DN_DEPL、W_DEPL_OFF、W_DEPL_ON（不經過物理計算）。
    V_REV=2.0,        # 反向偏壓大小 (V)，填正數（>= 0）。越大空乏區越寬、位移越大；寬度約與 sqrt(Vbi + V) 成正比。
    NA=5e17,          # p 側受體摻雜濃度 (cm^-3)。常見矽光調變器約 1e17 ~ 1e18。
    ND=5e17,          # n 側施體摻雜濃度 (cm^-3)。空乏區往濃度「較低」的一側延伸得比較多。
    TAU_MP=3.13e-14,  # 電洞的動量散射（弛豫）時間 τ_mp (s)，視為定值。預設值讓 NA+ND=1e18 時算出的電洞傳導有效質量
                      #   剛好是常用的 0.39 m0（m = qτ/μ 反推）。
    TAU_MN=3.77e-14,  # 電子的動量散射時間 τ_mn (s)，視為定值。預設值讓 NA+ND=1e18 時算出的電子傳導有效質量 = 0.26 m0。
    LAMBDA_PHYS_UM=1.55,  # 計算 Δn 用的工作波長 (µm)（Drude 公式裡的 λ，與掃描窗口分開，只影響 Δn 的大小）
    N_SI_BULK=3.476,  # 計算 Δn 用的「體材矽」折射率（公式裡的 n）；不是 2D 模擬用的等效折射率 N_SI
    N_G=2.9,          # 環的群折射率 n_g（無因次），只用來由共振位移換算相位調變效率：
                      #   FSR = λ²/(n_g·2πR)，環一圈的相位變化 Δφ = 2π·位移/FSR。
                      #   2.9 是 R=3 µm 時由模擬的 FSR（約 41~44 nm）估出來的值；Meep 窗口只含一個共振時量不到 FSR，
                      #   所以用這個估計值；n_g 差 10%，相位效率就差 10%。可用 derive_n_eff.py 或兩個相鄰共振的間距重新估。
    ALPHA_DOPE=7.0,   # （PN_FROM_PHYSICS=False 時才用；True 時由摻雜算出）摻雜矽的基線吸收係數 (cm^-1)，OFF 與 ON 的環都有；換算：dB/cm = 4.34 × cm^-1，
                      #   所以 7 cm^-1 約 30 dB/cm。數值來自摻雜 5e17 cm^-3 的 Soref 估算（示意值，使用前請對照原文）。
                      #   越大：環自身損耗越大（Q_i 越低）、凹谷越寬；它也決定 OFF/ON 的吸收差有多大。
    ALPHA_DEPL=0.0,   # （PN_FROM_PHYSICS=False 時才用）空乏區的吸收係數 (cm^-1)，不可為負；0 = 載子被掃光、沒有吸收。
                      #   PN_FROM_PHYSICS=True 時會被「基線吸收 + Δα」算出的值取代，且 ALPHA_DOPE 也改由摻雜算出。
    DN_DEPL=0.0017,   # （PN_FROM_PHYSICS=False 時才用）完全空乏的區域，折射率比摻雜矽高多少（無因次，正值）。同樣來自 5e17 cm^-3、電子電洞都被掃光的
                      #   Soref 估算（示意值）。它乘上「空乏區增加的那一圈所含的光場比例」才是真正的 Δn_eff。
    W_DEPL_OFF=0.06,  # （PN_FROM_PHYSICS=False 時才用）OFF（零偏壓）的空乏區寬度 (µm)，以接面為中心、往兩側各展開一半。
                      #   真實的空乏寬度 ≈ sqrt(2·ε·(Vbi + V)/q · (1/Na + 1/Nd))，與摻雜濃度、偏壓有關（約與 √V 成正比）；
                      #   本程式不模擬電學，直接由你輸入。
    W_DEPL_ON=0.12,   # （PN_FROM_PHYSICS=False 時才用）ON（反向偏壓）的空乏區寬度 (µm)，必須比 W_DEPL_OFF 寬；
                      #   位移來自「多出來的這一圈」。
                      #   兩個寬度的差越大、且越靠近光場最強的地方，位移越大；但空乏區在 2D 圖上最多只能到環的內外緣。
    X_JUNC=0.0,       # 接面位置 (µm)：相對環中心線的「徑向偏移」，正值往外（遠離環心）、負值往內。
                      #   接面內側（靠環心）是 p 型、外側是 n 型（示意）；空乏環帶的中心位於半徑 R + X_JUNC。
                      #   0 = 接面剛好在環的中心線（基模光場最強處附近），位移效率最高；
                      #   偏離中心線時，空乏區與光場的重疊變小，位移變小；
                      #   偏移太大（超過 (W - 空乏寬度)/2）時，空乏環帶會被夾到環的內緣或外緣，有效寬度變窄。

    # ================= 模擬收斂與量測（數值參數）=================
    DECAY=1e-6,       # 停止條件：環內（或參考時的出口）的場強衰減到峰值的多少倍就停止模擬。
                      #   越小越準（高 Q 的光要等它漏光漏完）但越慢；太鬆會讓光譜出現假起伏，甚至 T > 1（not_converged）。
    USE_HARMINV=True, # True：在 OFF 的環模擬中加 Harminv，直接從時域衰減量 Q（比半高寬法可靠）。
                      #   尚未在真實 Meep 上驗證（目前每組都量不到值）；出錯時改成 False，不影響其他功能。
    ISOLATE_DIP=False,# True：窗口可以比 FSR 寬，分析時只取最深那個凹谷附近（掃 R 時由 sweep_resolution_gap.py 設定）。
    FSR_EST_UM=0.0,   # ISOLATE_DIP=True 時用的 FSR 估計 (µm)；擷取範圍為凹谷兩側各 0.45 倍 FSR，
                      #   取 0.45 而不是 0.5，是為了不把鄰近共振的邊緣包進來。
)

# 穿透率放大圖的 x 軸範圍：兩個凹谷外側各多留幾倍 FWHM（越小畫面越放大；只影響畫圖，不影響數值）
PLOT_ZOOM = 1.5


# ============================================================
# 1b. 偏壓、摻雜 -> 遷移率 -> 有效質量 -> 載子變化 -> 折射率變化
# ============================================================
# 目的：不要手動猜 DN_DEPL（空乏區折射率變化）與空乏區寬度，而是由「反向偏壓 V、摻雜 Na、Nd、
#       散射時間 τ」依下面四個步驟算出來。V、Na、Nd、τ 都視為定值。
#   步驟一  由摻雜濃度算遷移率 μp、μn                （經驗公式）
#   步驟二  由 μ = q·τ/m 反推有效質量 m_p、m_n          （τ 是定值）
#   步驟三  算出這個偏壓下的空乏區寬度，與空乏區內載子濃度的變化 ΔNa、ΔNd
#   步驟四  用 Drude 模型（Soref 論文的理論式）由 ΔN 與 m 算出折射率變化 Δn
# 單位：這一區塊用 SI（m、kg、s、C），但摻雜濃度與遷移率照文獻習慣用 cm^-3 與 cm^2/(V·s)，
#       進公式前才換算（1 cm^-3 = 1e6 m^-3；1 cm^2/(V·s) = 1e-4 m^2/(V·s)）。

Q_E = 1.602176634e-19        # 電子電荷 q (C)
C_LIGHT = 2.99792458e8       # 真空光速 c (m/s)
EPS0 = 8.8541878128e-12      # 真空介電常數 ε0 (F/m)
M0 = 9.1093837015e-31        # 自由電子質量 m0 (kg)，只用來把有效質量換算成「幾倍 m0」方便閱讀
EPS_SI_REL = 11.7            # 矽的相對介電常數（算空乏區寬度用）
NI_SI_CM3 = 1.0e10           # 300 K 時矽的本徵載子濃度 ni (cm^-3)（算內建電位 Vbi 用）
KT_Q_300K = 0.025852         # 300 K 的熱電壓 kT/q (V)


def pn_carrier_physics(p):
    """
    由 p 裡的 V_REV、NA、ND、TAU_MP、TAU_MN、LAMBDA_PHYS_UM、N_SI_BULK，依四個步驟算出一個 dict，
    每個中間量都留著，方便檢查與印出。為了方便對照，鍵名後面標了單位。
    """
    na, nd, v = p["NA"], p["ND"], p["V_REV"]          # 摻雜 (cm^-3)、反向偏壓 (V)
    if na <= 0 or nd <= 0:
        raise ValueError("NA、ND 必須大於 0")
    if v < 0:
        raise ValueError("V_REV 必須 >= 0（這裡只處理反向偏壓；順向偏壓不適用空乏近似）")

    # ---------- 步驟一：遷移率 μp、μn (cm^2/(V·s)) ----------
    # 經驗公式：摻雜越濃，載子被雜質散射越多，遷移率越低；最後的 +50、+92 是高摻雜時的下限。
    n_tot = na + nd                                                    # 公式裡用 (Na + Nd)
    mu_p = 420.0 / (1.0 + (n_tot / 1.6e17) ** 0.7) + 50.0              # 電洞遷移率 μp
    mu_n = 1318.0 / (1.0 + (n_tot / 1.0e17) ** 0.85) + 92.0            # 電子遷移率 μn

    # ---------- 步驟二：有效質量 m_p、m_n (kg) ----------
    # 遷移率與散射時間的關係 μ = q·τ/m，所以 m = q·τ/μ。τ 是定值；μ 要先換成 m^2/(V·s)。
    m_ch = Q_E * p["TAU_MP"] / (mu_p * 1e-4)                           # 電洞的傳導有效質量 m*_ch
    m_ce = Q_E * p["TAU_MN"] / (mu_n * 1e-4)                           # 電子的傳導有效質量 m*_ce

    # ---------- 步驟三：偏壓下的空乏區與載子變化 ΔNa、ΔNd ----------
    # (a) 內建電位 Vbi = (kT/q)·ln(Na·Nd / ni^2)：p、n 兩邊自然形成的電位差
    vbi = KT_Q_300K * np.log(na * nd / NI_SI_CM3 ** 2)
    # (b) 空乏近似的總寬度 W(V) = sqrt( 2·ε·(Vbi + V)/q · (1/Na + 1/Nd) )，單位 m；
    #     反向偏壓 V 加在內建電位上，所以越大越寬；濃度換成 m^-3 再代入
    eps_si = EPS_SI_REL * EPS0
    def depl_width_m(volt):
        return np.sqrt(2.0 * eps_si * (vbi + volt) / Q_E * (1.0 / (na * 1e6) + 1.0 / (nd * 1e6)))
    w_off, w_on = depl_width_m(0.0), depl_width_m(v)                   # 零偏壓（OFF）與偏壓 V（ON）的總寬度
    # (c) 空乏區往兩側延伸，電荷守恆（Na·xp = Nd·xn），所以往濃度較低的一側延伸得比較多：
    #     xp = W·Nd/(Na+Nd)（p 側）、xn = W·Na/(Na+Nd)（n 側）
    xp_on, xn_on = w_on * nd / (na + nd), w_on * na / (na + nd)      # 偏壓 V 下 p、n 側的空乏深度
    xp_off, xn_off = w_off * nd / (na + nd), w_off * na / (na + nd)  # 0 V 下 p、n 側的空乏深度
    # (d) Na、Nd 是原本就有的摻雜（定值）；偏壓只是把空乏區往外推，新增的那一層載子被掃光。
    #     ΔN 以「0 V 為基準」：新增那一層（xp_on - xp_off）內的載子全部消失，
    #     再把它平均到偏壓下整個空乏深度（xp_on），得到「偏壓 V 造成的平均載子變化」（cm^-3）。
    #     V = 0 時 xp_on = xp_off，所以 ΔN = 0。減少取負號：ΔNh < 0、ΔNe < 0。
    d_nh = -na * (xp_on - xp_off) / xp_on                            # p 側電洞變化 ΔNh
    d_ne = -nd * (xn_on - xn_off) / xn_on                            # n 側電子變化 ΔNe

    # ---------- 步驟四：折射率變化 Δn（Drude 模型，Soref 論文的理論式）----------
    #   Δn = -( e²λ² / (8π²c²ε0 n) ) · [ ΔNe/m*_ce + ΔNh/m*_ch ]
    # 載子「增加」(ΔN>0) 使折射率下降；這裡載子被掃光 (ΔN<0)，所以 Δn 為正。
    lam = p["LAMBDA_PHYS_UM"] * 1e-6                                   # 波長 (m)
    k_drude = Q_E ** 2 * lam ** 2 / (8 * np.pi ** 2 * C_LIGHT ** 2 * EPS0 * p["N_SI_BULK"])
    dn_e = -k_drude * (d_ne * 1e6) / m_ce                              # n 側（電子）的貢獻（ΔN 先換成 m^-3）
    dn_h = -k_drude * (d_nh * 1e6) / m_ch                              # p 側（電洞）的貢獻
    # Meep 的空乏環帶只能放「一個」折射率，所以取兩側的寬度加權平均：
    # n 側寬度占 Na/(Na+Nd)、p 側占 Nd/(Na+Nd)（與偏壓大小無關）。
    # 這個 Δn 已經是「相對 0 V」的平均變化，所以模擬時 OFF（0 V）不放環帶、ON 放整個寬度 w_on 的環帶。
    dn_strip = (na * dn_e + nd * dn_h) / (na + nd)

    # ---------- 損耗（自由載子吸收，同樣是 Drude 模型）----------
    #   Δα = ( e³λ² / (4π²c³ε0 n) ) · [ ΔNe/(m*_ce² μe) + ΔNh/(m*_ch² μh) ]      （單位 m^-1）
    # 載子增加 (ΔN>0) 損耗變大；載子被掃光 (ΔN<0) 損耗變小，所以空乏區的 Δα 為負。
    k_abs = Q_E ** 3 * lam ** 2 / (4 * np.pi ** 2 * C_LIGHT ** 3 * EPS0 * p["N_SI_BULK"])
    mu_e_si, mu_h_si = mu_n * 1e-4, mu_p * 1e-4                        # 遷移率換成 m^2/(V·s)
    def dalpha_cm(dne_cm3, dnh_cm3):
        # ΔN 先換成 m^-3；結果 m^-1 再換成 cm^-1（除以 100）
        return k_abs * (dne_cm3 * 1e6 / (m_ce ** 2 * mu_e_si) + dnh_cm3 * 1e6 / (m_ch ** 2 * mu_h_si)) / 100.0
    da_e = dalpha_cm(d_ne, 0.0)                                        # n 側（電子）的 Δα (cm^-1)，為負
    da_h = dalpha_cm(0.0, d_nh)                                        # p 側（電洞）的 Δα (cm^-1)，為負
    da_strip = (na * da_e + nd * da_h) / (na + nd)                     # 與 Δn 相同的寬度加權平均
    # 摻雜矽本身的基線吸收：把「全部載子 Nd 個電子、Na 個電洞」代進同一條式子；環的 p、n 各占一半寬度，取平均
    a_e0, a_h0 = dalpha_cm(nd, 0.0), dalpha_cm(0.0, na)                # 電子、電洞各自的吸收 (cm^-1)
    alpha_dope = 0.5 * (a_e0 + a_h0)                                   # 環的基線吸收（OFF 與環的其餘部分）
    # Meep 的吸收不能是負的：空乏環帶的吸收 = 基線 + Δα，下限 0
    alpha_depl = max(alpha_dope + da_strip, 0.0)

    return dict(
        mu_p_cm2Vs=mu_p, mu_n_cm2Vs=mu_n,                              # 步驟一
        m_ch_m0=m_ch / M0, m_ce_m0=m_ce / M0,                          # 步驟二（以 m0 為單位）
        vbi_V=vbi, w_off_um=w_off * 1e6, w_on_um=w_on * 1e6,           # 步驟三：寬度 (µm)
        xp_on_um=xp_on * 1e6, xn_on_um=xn_on * 1e6,
        dNh_cm3=d_nh, dNe_cm3=d_ne,                                    # 步驟三：載子變化 (cm^-3)
        dn_e=dn_e, dn_h=dn_h, dn_strip=dn_strip,                       # 步驟四：Δn（無因次）
        dalpha_e_cm=da_e, dalpha_h_cm=da_h, dalpha_strip_cm=da_strip,  # 損耗變化 Δα (cm^-1)
        alpha_dope_cm=alpha_dope, alpha_depl_cm=alpha_depl)            # 基線吸收與空乏環帶吸收 (cm^-1)


def resolve_pn_params(p):
    """
    PN_FROM_PHYSICS=True 時，用 pn_carrier_physics 的結果取代 p 裡的 DN_DEPL、ALPHA_DOPE、ALPHA_DEPL、W_DEPL_OFF、W_DEPL_ON，
    回傳新的 dict（不改到原本的 p）；False 時原樣回傳。可重複呼叫，結果不變。
    """
    if not p.get("PN_FROM_PHYSICS", False):
        return p
    r = pn_carrier_physics(p)
    q = dict(p)
    q["DN_DEPL"] = r["dn_strip"]               # 空乏環帶的折射率變化
    q["ALPHA_DOPE"] = r["alpha_dope_cm"]       # 摻雜矽的基線吸收（由摻雜、遷移率、有效質量算出）
    q["ALPHA_DEPL"] = r["alpha_depl_cm"]       # 空乏環帶的吸收 = 基線 + Δα（下限 0）
    q["W_DEPL_OFF"] = 0.0                      # 0 V 是 Δn 的基準，OFF 不放環帶（Δn 已是相對 0 V 的變化）
    q["W_DEPL_ON"] = r["w_on_um"]              # 偏壓 V_REV 下的整個空乏區寬度 (µm)，平均 Δn 涵蓋這整段
    return q


def print_pn_physics(p):
    """把四個步驟的中間結果印出來，方便你檢查每一步的數字合不合理。"""
    r = pn_carrier_physics(p)
    print(f"=== PN 載子物理（V_REV = {p['V_REV']} V，NA = {p['NA']:.2e}，ND = {p['ND']:.2e} cm^-3）===")
    print(f"步驟一  μp = {r['mu_p_cm2Vs']:.1f}，μn = {r['mu_n_cm2Vs']:.1f} cm^2/(V·s)")
    print(f"步驟二  m_p = {r['m_ch_m0']:.3f} m0（τ_mp = {p['TAU_MP']:.2e} s），"
          f"m_n = {r['m_ce_m0']:.3f} m0（τ_mn = {p['TAU_MN']:.2e} s）")
    print(f"步驟三  Vbi = {r['vbi_V']:.3f} V；空乏區寬度 {r['w_off_um']:.4f} µm（0 V）-> {r['w_on_um']:.4f} µm"
          f"（{p['V_REV']} V），其中 p 側 {r['xp_on_um']:.4f}、n 側 {r['xn_on_um']:.4f} µm")
    print(f"        相對 0 V 的平均載子變化：ΔNh = {r['dNh_cm3']:.2e}，ΔNe = {r['dNe_cm3']:.2e} cm^-3"
          f"（V = 0 時為 0；Na、Nd 是固定的摻雜）")
    print(f"步驟四  Δn_電子 = {r['dn_e']:+.3e}，Δn_電洞 = {r['dn_h']:+.3e}，"
          f"寬度加權平均 Δn = {r['dn_strip']:+.3e}")
    print(f"損耗    基線吸收 α_dope = {r['alpha_dope_cm']:.2f} cm^-1（{4.343 * r['alpha_dope_cm']:.1f} dB/cm）；"
          f"Δα_電子 = {r['dalpha_e_cm']:+.2f}，Δα_電洞 = {r['dalpha_h_cm']:+.2f}，平均 Δα = {r['dalpha_strip_cm']:+.2f} cm^-1；"
          f"空乏環帶吸收 = {r['alpha_depl_cm']:.2f} cm^-1")


# ============================================================
# 2. 幾何與材料
# ============================================================
def cell_geometry(p):
    """回傳 (直波導 y 位置, 計算區域寬 sx, 高 sy)。環的圓心放在 (0, 0)。"""
    # 直波導中心到環中心的距離 = R + W/2 (環外緣) + GAP + W/2 (直波導半寬)
    bus_y = -(p["R"] + p["W"] + p["GAP"])
    # sx（計算區寬）= 環直徑 2R + 6.0 µm：左右各留 3 µm，放 PML（DPML）、光源與監測器（各距 PML 內緣 0.5 µm）
    #   與足夠的直波導長度，讓光源與監測器離環的耦合區夠遠。
    sx = 2 * p["R"] + 6.0
    # sy（計算區高）：以環心為中心上下對稱，涵蓋直波導（bus_y 加半個波導寬）、PML（DPML）
    #   以及再多 0.5 µm 的餘裕；乘 2 是因為上下對稱（上方空白區用來容納環的上緣與 PML）。
    sy = 2 * (abs(bus_y) + p["W"] / 2 + p["DPML"] + 0.5)
    return bus_y, sx, sy


def si_medium(n0, dn=0.0, dalpha_cm=0.0, wvl=1.55):
    """
    建立矽材料：折射率 n0 + dn，並加入吸收 dalpha_cm (cm^-1)。
    參數：n0 = 基準折射率；dn = 折射率變化量（無因次）；dalpha_cm = 吸收係數（cm^-1，不可為負）；
          wvl = 換算吸收用的參考波長 (µm)，取 1.55（吸收在掃描窗口內視為不隨波長變）。
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


def depletion_medium(p):
    """空乏區的材料：折射率比摻雜矽高 DN_DEPL；吸收 = ALPHA_DEPL（載子被掃光所以比 ALPHA_DOPE 小，最低 0）。"""
    return si_medium(p["N_SI"], p["DN_DEPL"], max(p.get("ALPHA_DEPL", 0.0), 0.0))


def depletion_edges(p, depl_width):
    """
    空乏環帶的內、外半徑 (µm)：以接面 (R + X_JUNC) 為中心、寬度 depl_width，
    並夾在環的內外緣之間（空乏區不可能超出波導）。寬度為 0 或被夾到沒有厚度時回傳 None。
    """
    r_in, r_out = p["R"] - p["W"] / 2, p["R"] + p["W"] / 2       # 環的內緣、外緣半徑 (µm)
    r_junc = p["R"] + p["X_JUNC"]                                 # 接面半徑 = 中心線半徑 + 徑向偏移 X_JUNC
    d_in = max(r_junc - depl_width / 2, r_in)                     # 空乏區內緣：接面往內退半個寬度，但不超出環內緣
    d_out = min(r_junc + depl_width / 2, r_out)                   # 空乏區外緣：接面往外進半個寬度，但不超出環外緣
    return (d_in, d_out) if (depl_width > 0 and d_out > d_in) else None


def build_geometry(p, with_ring, ring_medium, depl_width=0.0):
    """
    建立幾何物件清單：直波導，以及（選擇性）環。
    depl_width > 0 時（PN 模式），在環內加一圈空乏環帶，其材料為 depletion_medium。
    """
    bus_y, _, _ = cell_geometry(p)
    clad = mp.Medium(index=p["N_CLAD"])         # 包覆層（用來把環的內圓挖空）
    bus_medium = mp.Medium(index=p["N_SI"])     # 直波導不受電壓影響，也沒有摻雜吸收，用原本的矽（無損耗）

    # 直波導：沿 x 方向無限延伸（會一路延伸進 PML，避免邊界反射）
    geometry = [mp.Block(size=mp.Vector3(mp.inf, p["W"], mp.inf),
                         center=mp.Vector3(0, bus_y),
                         material=bus_medium)]

    if with_ring:
        # 環 = 外圓（矽）減去內圓（包覆層）。後放的物件會蓋過先放的。
        geometry += [mp.Cylinder(radius=p["R"] + p["W"] / 2, material=ring_medium)]
        edges = depletion_edges(p, depl_width)
        if edges is not None:
            d_in, d_out = edges
            # 先蓋一個「空乏材料」的外圓，再用「環材料」的內圓把 d_in 以內還原，
            # 剩下 d_in 到 d_out 之間就是空乏環帶
            geometry += [mp.Cylinder(radius=d_out, material=depletion_medium(p)),
                         mp.Cylinder(radius=d_in, material=ring_medium)]
        geometry += [mp.Cylinder(radius=p["R"] - p["W"] / 2, material=clad)]
    return geometry


# ============================================================
# 3. 跑一次模擬，回傳各頻率的穿透功率
# ============================================================
def run_spectrum(p, with_ring, ring_medium=None, want_eps=False, want_harminv=False,
                 depl_width=0.0):
    """
    with_ring=False -> 參考模擬（只有直波導）
    with_ring=True  -> 有環，環的（摻雜矽）材料由 ring_medium 決定
    depl_width      -> 空乏區寬度 (µm)；OFF 與 ON 的差別就在這（OFF 窄、ON 寬）
    want_harminv=True 且有環時，會在環內放一個 Harminv 監測點，直接由時域的衰減量出 Q。
    回傳 (頻率陣列, 穿透功率 flux 陣列, 介電常數圖或 None, Harminv 模態清單)
    Harminv 模態清單的每一項是 (波長 µm, Q, 誤差估計, 振幅)；沒用 Harminv 時是空清單。
    """
    bus_y, sx, sy = cell_geometry(p)

    # --- 頻率設定 ---
    f_min, f_max = 1 / p["WVL_MAX"], 1 / p["WVL_MIN"]       # 波長 -> 頻率（大小會反過來）
    fcen = (f_min + f_max) / 2                               # 中心頻率
    df = f_max - f_min                                       # 監測的頻寬
    # 光源頻寬 = 2.5 倍監測頻寬：高斯脈衝的頻譜像鐘形，邊緣很弱；取比監測範圍寬一些，
    # 才能讓整個監測範圍內的頻率都有足夠的能量，歸一化時不會除到很小的數字。
    src_fwidth = 2.5 * df

    # --- 光源：直波導左端的寬頻高斯脈衝 ---
    # 這裡用最單純的 Ez 點源（沿波導截面鋪開），主要激發基模。
    # 之後可改用 mp.EigenModeSource 只激發乾淨的基模。
    # 光源 x 位置：離 PML 內緣 0.5 µm（太靠近 PML 會被吸收、也會激發出怪模態）
    src_x = -sx / 2 + p["DPML"] + 0.5
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=src_fwidth),
                         component=mp.Ez,
                         center=mp.Vector3(src_x, bus_y),
                         size=mp.Vector3(0, p["W"]))]        # 光源沿 y 鋪開，長度 = 波導寬 W，只激發波導內

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy),
        geometry=build_geometry(p, with_ring, ring_medium, depl_width),
        sources=sources,
        boundary_layers=[mp.PML(p["DPML"])],
        resolution=p["RESOLUTION"],
        default_material=mp.Medium(index=p["N_CLAD"]),       # 背景 = 包覆層
    )

    # --- 監測器：直波導右端（through port），在環的耦合區之外 ---
    # 監測器 x 位置：右端離 PML 內緣 0.5 µm（與光源左右對稱），遠離環的耦合區，只收直波導傳出來的光
    mon_x = sx / 2 - p["DPML"] - 0.5
    through = sim.add_flux(
        fcen, df, p["NFREQ"],
        mp.FluxRegion(center=mp.Vector3(mon_x, bus_y),
                      size=mp.Vector3(0, 3 * p["W"])))   # 監測線長 3W：包住波導與兩側的消逝場，才收得到全部功率

    # --- 決定「什麼時候可以停」---
    # 有環：看環裡的場衰減完（共振光要等它漏光漏完，Q 越高越久）
    # 沒環：看直波導出口的場衰減完
    if with_ring:
        decay_pt = mp.Vector3(p["R"], 0)                     # 環右側中心線上的一個點（在矽內），看環內場有沒有衰減完
    else:
        decay_pt = mp.Vector3(mon_x, bus_y)

    # --- Harminv：光源關掉後，分析環內一個點的訊號，找出共振頻率與衰減率（Q）---
    # 光譜法要靠「凹谷夠深」才量得準；Harminv 看的是時域衰減，不受凹谷深淺影響。
    harminv, step_funcs = None, []
    if want_harminv and with_ring:
        try:
            # Harminv(分量, 取樣位置, 中心頻率, 頻寬)：在環右側中心線上取樣，
            # 在 [fcen - df/2, fcen + df/2] 內找衰減的振盪模態（頻率、Q、誤差、振幅）
            harminv = mp.Harminv(mp.Ez, mp.Vector3(p["R"], 0), fcen, df)
            step_funcs.append(mp.after_sources(harminv))   # 只在光源關掉之後才取樣
        except Exception:                                  # noqa: BLE001（建立失敗就不用 Harminv）
            harminv = None

    # stop_when_fields_decayed(dt, 分量, 位置, 衰減比例)：每隔 dt = 50 個時間單位檢查一次，
    # 該位置的 Ez 振幅衰減到「峰值的 DECAY 倍」以下就停止（光源關掉之後才開始計）。
    sim.run(*step_funcs, until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, decay_pt, p["DECAY"]))

    flux = np.array(mp.get_fluxes(through))
    freqs = np.array(mp.get_flux_freqs(through))

    # 取出介電常數分布，用來畫 2D 結構圖
    eps = None
    if want_eps:
        eps = sim.get_array(center=mp.Vector3(),
                            size=mp.Vector3(sx, sy),
                            component=mp.Dielectric)
    # 整理 Harminv 找到的模態（取不到就當作沒有，不讓整個模擬失敗）
    modes = []
    if harminv is not None:
        try:
            for m in harminv.modes:
                if m.Q > 0 and m.freq > 0:                 # 排除成長（Q<0）或負頻率的假模態；存 (波長, Q, 誤差, 振幅)
                    modes.append((1.0 / m.freq, float(m.Q), float(m.err), float(abs(m.amp))))
        except Exception:                                  # noqa: BLE001
            modes = []
    return freqs, flux, eps, modes


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


# ------------------------------------------------------------
# 4a. 更可靠的凹谷量測：洛倫茲擬合 + Harminv
# ------------------------------------------------------------
# 為什麼要改：半高寬法只用「最亮點、最暗點、半深度交叉點」幾個點，
# 凹谷只有 2~3% 深、漣漪約 0.3% 時，最亮點（取最大值）會被漣漪拉高，
# 半高寬因此偏差，實測同一組設計重跑 Q 差了約 34%。
# 擬合用凹谷附近全部的點，雜訊會互相平均；並且能給出不確定度。

def lorentz_dip(x, t0, a, x0, g):
    """洛倫茲凹谷：T = t0 - a / (1 + ((x - x0) / (g/2))^2)。t0 基線、a 深度、x0 中心、g 半高寬。"""
    return t0 - a / (1.0 + ((x - x0) / (g / 2.0)) ** 2)


def fit_dip(wvl, T, m):
    """
    對最深的凹谷做洛倫茲擬合。m 是 analyze_dip 的結果（只用來給初始值）。
    回傳 dict：lam0、fwhm（µm）、Q、ER（dB）、Q_err（Q 的相對 1σ 不確定度）、
    rms（殘差）、x/y（擬合曲線，畫圖用）。失敗（沒有 scipy、擬合不收斂）時數值為 nan。
    注意：Q_err 只反映雜訊的隨機部分，漣漪造成的系統偏差不在內。
    """
    bad = dict(lam0=np.nan, fwhm=np.nan, Q=np.nan, ER=np.nan, Q_err=np.nan,
               rms=np.nan, x=None, y=None)
    if not HAVE_SCIPY:
        return bad

    i = int(np.argmin(T))
    lam0 = wvl[i]
    g0 = m["fwhm"] if np.isfinite(m["fwhm"]) else 5e-4         # 半高寬初始值 (µm)：量不到就假設 0.5 nm（5e-4 µm）
    # 只用凹谷附近（兩側各 8 倍半高寬）的點；太少點就用全部
    sel = np.abs(wvl - lam0) <= 8 * g0
    if sel.sum() < 15:                                         # 點數少於 15 擬合不穩，改用全部資料
        sel = np.ones_like(wvl, dtype=bool)
    x_nm = (wvl[sel] - lam0) * 1e3                             # 以 nm 為單位、以凹谷為原點，數值較穩定
    y = T[sel]
    far = np.abs(x_nm) > 4 * g0 * 1e3                          # 離凹谷超過 4 倍半高寬的點，用來估基線（凹谷的尾巴已很小）
    t0_init = np.median(y[far]) if far.sum() >= 3 else np.max(y)
    # 初始值 p0 = [基線 t0, 深度 a, 中心偏移 x0（相對 lam0，nm）, 半高寬 g（nm）]；深度至少給 1e-4 避免從 0 出發
    p0 = [t0_init, max(t0_init - T[i], 1e-4), 0.0, g0 * 1e3]
    dx = np.median(np.diff(x_nm))
    # 擬合的上下界：基線限制在 0.8~1.2（T 本來就該接近 1）；深度 0~1.5；中心限制在資料範圍內；
    # 半高寬至少 0.5 倍取樣間隔（比取樣還窄的凹谷無法分辨）、至多整個資料範圍
    lower = [0.8, 0.0, x_nm.min(), 0.5 * dx]
    upper = [1.2, 1.5, x_nm.max(), x_nm.max() - x_nm.min()]
    try:
        # maxfev = 最多呼叫模型函式 5000 次；超過還沒收斂就放棄（回傳 nan）
        popt, pcov = curve_fit(lorentz_dip, x_nm, y, p0=p0, bounds=(lower, upper), maxfev=5000)
    except Exception:                                          # noqa: BLE001（擬合失敗就回傳 nan）
        return bad

    t0, a, x0, g = popt
    lam0_fit = lam0 + x0 * 1e-3                                # 回到 µm
    fwhm = g * 1e-3
    err = np.sqrt(pcov[3, 3]) / g if np.isfinite(pcov[3, 3]) else np.nan
    t_min = max(t0 - a, 1e-12)
    resid = y - lorentz_dip(x_nm, *popt)
    return dict(lam0=lam0_fit, fwhm=fwhm, Q=lam0_fit / fwhm,
                ER=10 * np.log10(t0 / t_min), Q_err=float(err),
                rms=float(np.sqrt(np.mean(resid ** 2))),
                x=lam0 + x_nm * 1e-3, y=lorentz_dip(x_nm, *popt))


def pick_harminv_mode(modes, lam0_um, tol_um=5e-4):   # tol_um = 容許誤差 (µm)，5e-4 µm = 0.5 nm
    """
    從 Harminv 找到的模態中，挑出離光譜凹谷位置 lam0_um 最近（容許 tol_um）的那一個，
    多個時取振幅最大的。回傳它的 Q；找不到回傳 nan。
    """
    near = [md for md in modes if abs(md[0] - lam0_um) <= tol_um]
    if not near:
        return np.nan
    return max(near, key=lambda md: md[3])[1]


def isolate_main_dip(wvl, T_off, T_on, half_width_um):
    """
    只保留「OFF 最深的那個凹谷」兩側各 half_width_um 以內的資料。
    用在窗口比 FSR 寬（窗口內有好幾個共振）的情況：只分析其中一個，其餘丟掉。
    half_width_um 一般取 0.45 倍 FSR，確保鄰近的共振不會被包進來。
    """
    i = int(np.argmin(T_off))
    keep = np.abs(wvl - wvl[i]) <= half_width_um
    return wvl[keep], T_off[keep], T_on[keep]


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


def phase_efficiency(shift_um, lam0_um, p):
    """
    由共振位移換算「相位調變效率」。回傳 (Δφ, 效率 rad/V, VπL V·cm)。
    原理：環共振時一圈的相位 = 2π 的整數倍；共振波長移動 Δλ，等於一圈的相位多了
          Δφ = 2π·Δλ/FSR      （FSR = λ²/(n_g·L)，L = 2πR 是環一圈的長度）
    指標：
      效率 η    = Δφ / V_REV              (rad/V)  一圈的相位變化 / 偏壓，越大越好
      VπL       = V_REV · L · π / |Δφ|    (V·cm)   讓相位變 π 所需的「電壓 × 長度」，越小越好；
                  這裡 L 取環周長 2πR（因為光在環裡繞很多圈，這是單圈的長度；文獻若用有效長度請另行換算）
    位移為 0（或量不到）時 VπL 沒有意義，回傳 nan。
    """
    if not np.isfinite(shift_um) or not np.isfinite(lam0_um):
        return np.nan, np.nan, np.nan                           # 量不到位移或共振波長
    circ_um = 2 * np.pi * p["R"]                                # 環周長 L (µm)
    fsr_um = lam0_um ** 2 / (p["N_G"] * circ_um)                # 自由頻譜範圍 FSR (µm)
    dphi = 2 * np.pi * shift_um / fsr_um                        # 一圈的相位變化 Δφ (rad)；位移為正（紅移）時為正
    v = p["V_REV"]                                              # 偏壓 (V)
    eta = dphi / v if v > 0 else np.nan                         # 效率 (rad/V)；0 V 沒有意義
    vpil = v * (circ_um * 1e-4) * np.pi / abs(dphi) if (v > 0 and dphi != 0) else np.nan   # 1 µm = 1e-4 cm
    return dphi, eta, vpil


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
    "no_dip":        "凹谷深度低於門檻（可能耦合太弱而太淺，或共振完全在窗口外；可用 center_um 預測窗口）",
    "edge_low":      "凹谷靠近窗口短波長端（共振可能在窗口外）",
    "edge_high":     "凹谷靠近窗口長波長端（共振可能在窗口外）",
    "incomplete":    "凹谷不完整，量不到半高寬",
    "multi":         "窗口內有多個共振（請縮小 WVL 範圍或改 R）",
}


def check_dip(wvl, T, min_er_db=0.1, edge_frac=0.05, max_T=1.02, multi_frac=0.3):
    """
    檢查一條穿透率曲線的凹谷是否「可信」。回傳錯誤代碼（見 REASONS），"ok" 代表合格。
    wvl 必須由小到大排序。檢查順序很重要：先排除模擬本身的問題，再看凹谷。

    min_er_db : 凹谷深度（dB）低於此值視為「沒有凹谷」。0.1 dB 約等於深度 2.3%。
                實測參考：收斂後的雜訊約 ±0.3%（約 0.013 dB），一個真實但很淺的凹谷
                （深度約 3%）約 0.13 dB，所以 0.1 dB 能分開兩者。想更嚴格可調高。
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

    # 5. 有幾個凹谷：數「進入凹谷」的次數。
    #    用遲滯（兩條線）而不是單一門檻：只用一條線時，雜訊在凹谷側壁的門檻附近來回穿越，
    #    一個凹谷會被數成好幾個（假的 multi）。這裡要先低於 enter 線才算進入，
    #    再回升到 leave 線以上才算離開，雜訊小於兩線的間距就不會重複計數。
    #    淺凹谷（深度 2~3%）和雜訊（約 0.2~0.3%）差不到 10 倍，所以先做 5 點移動平均，
    #    並用中位數當基線：最大值會被雜訊的正向極端值拉高，門檻跟著抬高，基線上的雜訊就會被當成凹谷。
    k = 5 if n >= 50 else 1
    Ts = np.convolve(np.pad(T, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    base = np.median(T)                         # 基線（凹谷只佔一小部分，中位數不受影響）
    depth = base - Ts.min()
    enter = base - multi_frac * depth           # 低於這條線 -> 進入一個凹谷
    leave = base - 0.1 * depth                  # 回升到這條線以上 -> 離開凹谷
    n_dips, inside = 0, False
    for t in Ts:
        if not inside and t < enter:
            n_dips, inside = n_dips + 1, True
        elif inside and t > leave:
            inside = False
    if n_dips > 1:
        return "multi"

    return "ok"


def set_window(p, center_um=None, shift_um=0.0):
    """
    回傳新的參數 dict，掃描窗口寬度不變，但平移或重設中心。
      center_um : 指定新的窗口中心 (µm)
      shift_um  : 在現有位置上平移多少 (µm)
    """
    q = dict(p)                                                    # 複製，不改到呼叫者的參數
    width = p["WVL_MAX"] - p["WVL_MIN"]                            # 窗口寬度 (µm)，平移時維持不變
    center = (p["WVL_MIN"] + p["WVL_MAX"]) / 2 if center_um is None else center_um   # 沒指定就用現有中心
    center += shift_um                                             # 再往長（+）或短（-）波長平移
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
        q["DECAY"] = max(p["DECAY"] * 0.01, 1e-10)      # 縮小 100 倍；1e-10 是下限（再小也不會更準，只會跑很久）
        return q, f"DECAY 加嚴為 {q['DECAY']:.0e}"

    # --- no_dip、multi 等無法靠平移修正 ---
    return p, None


# ============================================================
# 5. 畫圖
# ============================================================
def plot_structure(eps, p, filename="mrm_structure.png"):
    """畫 2D 結構平面圖：顏色越深 = 折射率越高（矽）。"""
    bus_y, sx, sy = cell_geometry(p)
    fig, ax = plt.subplots(figsize=(7, 6))                   # 圖的大小（英吋）；存檔時 dpi=200，所以約 1400x1200 像素

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

    # ---------- 在圖上標示結構參數（每次畫圖都從 p 自動帶入，不用手動改） ----------
    R, W, gap = p["R"], p["W"], p["GAP"]
    # 標註箭頭與文字的共同樣式：橘色雙箭頭
    arrow = dict(arrowstyle="<->", color="tab:orange", lw=1.6, shrinkA=0, shrinkB=0)
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    label = dict(color="tab:orange", fontsize=10, fontweight="bold", bbox=box)

    # R：從環心 (0,0) 畫到環的中心線（45 度方向），文字標在線段旁
    c45 = np.cos(np.pi / 4)
    ax.annotate("", xy=(R * c45, R * c45), xytext=(0, 0), arrowprops=arrow)
    ax.text(R * c45 / 2 - 0.15, R * c45 / 2 + 0.25, f"R = {R} µm", ha="right", **label)   # 放在箭頭左上方（環內空白處）

    # W：環最上方的截面，從內緣量到外緣；文字放在右上側避免遮住環
    ax.annotate("", xy=(0, R + W / 2), xytext=(0, R - W / 2), arrowprops=arrow)
    ax.text(0.4, R + W / 2 + 0.3, f"W = {W} µm", va="bottom", **label)

    # gap：環外緣 (y = -(R+W/2)) 到直波導上緣的距離；gap 很小，文字用引線拉到旁邊
    y_ring_out = -(R + W / 2)
    ax.annotate("", xy=(0, y_ring_out - gap), xytext=(0, y_ring_out), arrowprops=arrow)
    ax.annotate(f"gap = {gap} µm", xy=(0, y_ring_out - gap / 2),
                xytext=(R + 0.9, y_ring_out + 0.45), ha="center", arrowprops=dict(arrowstyle="-", color="tab:orange", lw=1.0),
                **label)

    # 材料與折射率：矽標在環旁、SiO2 標在環心（背景區）
    ax.text(0, -1.0, f"SiO$_2$ cladding\n$n_{{SiO_2}}$ = {p['N_CLAD']}", ha="center", va="center",
            fontsize=9, bbox=box)
    ax.annotate(f"Si ring / bus\n$n_{{Si}}$ = {p['N_SI']}", xy=(-R - W / 2, 0), xytext=(-5.0, -1.5),
                ha="center", va="center", fontsize=9, bbox=box,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # 空乏區只有幾十 nm 厚、折射率差只有 0.001 左右，在介電常數圖上看不出來，
    # 所以用圓弧畫出接面（虛線）與 OFF、ON 的空乏區邊界（細線、點線）
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot((R + p["X_JUNC"]) * np.cos(th), (R + p["X_JUNC"]) * np.sin(th),
            color="tab:purple", ls="--", lw=1.0, label="pn junction")
    for wd, ls, nm in ((p["W_DEPL_OFF"], "-", "depletion OFF"), (p["W_DEPL_ON"], ":", "depletion ON")):
        e = depletion_edges(p, wd)
        if e is not None:
            for k, rr in enumerate(e):                 # 內、外兩條邊界；只在第一條加圖例
                ax.plot(rr * np.cos(th), rr * np.sin(th), color="tab:cyan", ls=ls, lw=0.9,
                        label=nm if k == 0 else None)
    # 內側（環中央的白色區）標 p，外側（環外的白色背景）標 n，不蓋在黑色的矽上
    ax.text(-(R - W / 2) + 0.15, 0.0, "p", color="tab:purple", fontsize=13,
            fontweight="bold", ha="left", va="center")
    ax.text(R + W / 2 + 0.15, 0.0, "n", color="tab:purple", fontsize=13,
            fontweight="bold", ha="left", va="center")
    # PN 參數：放左下角的空白處
    pn_info = (f"PN depletion: {p['W_DEPL_OFF']:.3f} → {p['W_DEPL_ON']:.3f} µm (OFF → ON)\n"
               f"$\\Delta n_{{depl}}$ = +{p['DN_DEPL']:.2e},  "
               f"$\\alpha_{{dope}}$ = {p['ALPHA_DOPE']:.2f},  $\\alpha_{{depl}}$ = {p.get('ALPHA_DEPL', 0.0):.2f} cm$^{{-1}}$")
    ax.text(0.02, 0.02, pn_info, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # 參數總表：放在左上角（計算區的空角落，不會擋到結構）
    info = (f"R = {R} µm\nW = {W} µm\ngap = {gap} µm\n"
            f"$n_{{Si}}$ = {p['N_SI']}\n$n_{{SiO_2}}$ = {p['N_CLAD']}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Micro-ring structure")               # 數值已標在圖上，標題不再重複
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)                           # dpi = 每英吋像素數，越大圖越細
    plt.close(fig)                                           # 關掉圖，避免批次呼叫時記憶體越用越多


def plot_transmission(wvl, T_off, T_on, m_off, m_on, mod, filename="mrm_transmission.png",
                      zoom=True, xlim=None, fit=None):
    """
    畫穿透率 vs 波長（OFF / ON）。
    zoom=True : 自動放大到兩個凹谷附近（兩側各留 PLOT_ZOOM 倍 FWHM）；False 則畫完整掃描範圍。
    xlim      : 手動指定 x 軸範圍 (nm)，例如 (1541, 1547)；優先於 zoom。
    fit       : fit_dip 的結果；給了就把 OFF 凹谷的洛倫茲擬合曲線疊上去，方便肉眼檢查擬合是否合理。
    上圖：自動放大到兩個凹谷附近，才看得出位移與深度差異。
    下圖：ΔT = T_on - T_off，亮暗差最大的地方就是雷射最佳位置。
    """
    w_nm = wvl * 1000                                        # µm -> nm
    l_off, l_on = m_off["lam0"] * 1000, m_on["lam0"] * 1000  # 兩個共振波長 (nm)

    # --- 決定 x 軸範圍 ---
    if xlim is not None:                                     # 手動指定（最優先）
        lo, hi = xlim
    elif zoom:                                               # 自動放大：共振兩側各留 PLOT_ZOOM 倍 FWHM
        fw = np.nanmax([m_off["fwhm"], m_on["fwhm"]]) * 1000
        pad = PLOT_ZOOM * fw if np.isfinite(fw) else 2.0     # 量不到 FWHM 就固定留 2 nm
        lo, hi = min(l_off, l_on) - pad, max(l_off, l_on) + pad
    else:                                                    # 完整掃描範圍
        lo, hi = w_nm.min(), w_nm.max()
    keep = (w_nm >= lo) & (w_nm <= hi)                       # 範圍內的資料點
    if not keep.any():                                       # 手動範圍內沒有資料時退回全部
        keep = np.ones_like(w_nm, dtype=bool)
    y_low = min(T_off[keep].min(), T_on[keep].min())

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,                 # 上下兩張圖共用 x 軸（波長）
        gridspec_kw=dict(height_ratios=[3, 1.3]))            # 上圖（穿透率）與下圖（ΔT）高度比 3 : 1.3

    # --- 上圖：穿透率 ---
    ax.plot(w_nm, T_off, color="tab:red", lw=1.8, label="OFF (0 V)")
    ax.plot(w_nm, T_on, color="tab:green", lw=1.8, ls="--", label="ON (reverse bias)")
    if fit is not None and fit.get("x") is not None:        # 疊上洛倫茲擬合曲線（OFF）
        ax.plot(fit["x"] * 1000, fit["y"], color="k", lw=1.0, ls=":",
                label="Lorentz fit (OFF)")
    ax.axvline(l_off, color="tab:red", ls=":", lw=0.8)       # OFF 共振位置
    ax.axvline(l_on, color="tab:green", ls=":", lw=0.8)      # ON 共振位置
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(0.0, y_low - 0.05), 1.03)                # y 軸下緣比最低點再低 0.05；上緣 1.03 讓 T≈1 的線不貼框
    ax.set_ylabel("Transmission T")
    ax.set_title(f"Resonance shift = {l_on - l_off:+.3f} nm   (ER = {mod['ER_mod']:.2f} dB)",
                 pad=48)                                      # pad 留給兩列圖例
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
                   label=f"laser {lam_L:.2f} nm (ER = {mod['ER_mod']:.2f} dB)")
        ax2.axvline(lam_L, color="k", ls="-.", lw=1)

    # 圖例放在標題下方、圖框外，避免擋住凹谷
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


# ============================================================
# 6. 完整流程（包成函式，逆向設計直接呼叫 evaluate）
# ============================================================
# evaluate 回傳的欄位（無效時數值欄位為 nan，欄位固定，方便存成表格）
# 目前只保留兩個指標：ER_dB（調變消光比）與相位調變效率（phase_eff_rad_per_V、VpiL_Vcm）。
# lam0_off_nm（OFF 共振波長，掃描時用來追蹤窗口）與 shift_nm（共振位移，相位效率由它算出）是中間量，一併保留。
METRIC_KEYS = ("lam0_off_nm", "shift_nm", "ER_dB", "phase_eff_rad_per_V", "VpiL_Vcm")


def _simulate_once(p, make_plots, ref_cache):
    """
    跑完「參考 + OFF + ON」三次模擬，回傳 (波長, T_off, T_on, 介電常數圖, Harminv 模態)。
    參考光譜只和直波導、解析度、掃描窗口有關，所以用 ref_cache 重用：
    只要這些沒變，就不用重跑。（R、GAP 只影響計算區域大小，
    對直波導的功率頻譜影響很小，視為可重用。）
    """
    key = (p["W"], p["N_SI"], p["N_CLAD"], p["RESOLUTION"], p["DPML"],
           p["NFREQ"], round(p["WVL_MIN"], 6), round(p["WVL_MAX"], 6))
    if key not in ref_cache:
        freqs, flux, _, _ = run_spectrum(p, with_ring=False)
        ref_cache[key] = (freqs, flux)
    freqs, ref_flux = ref_cache[key]

    # PN 反向偏壓：OFF 與 ON 用同一種摻雜矽（有基線吸收），差別只在空乏區寬度
    ring_medium = si_medium(p["N_SI"], 0.0, p["ALPHA_DOPE"])
    w_off, w_on = p["W_DEPL_OFF"], p["W_DEPL_ON"]

    # 電壓 OFF（零偏壓，空乏區窄）
    _, off_flux, eps, modes = run_spectrum(
        p, True, ring_medium, want_eps=make_plots,
        want_harminv=p.get("USE_HARMINV", False),          # Harminv 只在 OFF 的環模擬中使用
        depl_width=w_off)
    # 電壓 ON（反向偏壓，空乏區寬）
    _, on_flux, _, _ = run_spectrum(p, True, ring_medium, depl_width=w_on)

    # 穿透率 = 有環 / 參考；頻率轉波長後，排成由短到長
    wvl = 1 / freqs
    order = np.argsort(wvl)
    return (wvl[order], (off_flux / ref_flux)[order],
            (on_flux / ref_flux)[order], eps, modes)


def evaluate(p, make_plots=False, ref_cache=None, center_um=None, max_retry=2, spectra_out=None):
    """
    給一組參數 p，回傳指標 dict。逆向設計時每組設計都呼叫這個函式。

    p          : 參數 dict（不會被修改）
    make_plots : True 時存圖與 npz（平常批次掃描請關掉）
    ref_cache  : 空 dict；多次呼叫時傳同一個，可重用參考光譜省時間
    center_um  : 預測的共振位置 (µm)，會把掃描窗口移到以它為中心（見 predict_center）
    max_retry  : 檢查不合格時，最多自動修正並重跑幾次
    spectra_out: 檔名（.npz）；給了就把完整光譜存起來，之後可以不重跑 Meep 而重新分析

    回傳 dict 內固定有：
      valid  : True 才代表數字可信；False 時其他數值僅供除錯，不要拿去訓練
      reason : 無效原因（中文），有效時為 "ok"
      retries: 實際重試次數
      window_nm : 最後使用的掃描窗口 (nm)
    以及 METRIC_KEYS 列出的各項指標。
    """
    p = dict(p)                                 # 複製一份，避免改到呼叫者的參數
    p = resolve_pn_params(p)                    # PN_FROM_PHYSICS=True：由偏壓、摻雜、τ 算出 DN_DEPL 與空乏區寬度
    if ref_cache is None:
        ref_cache = {}
    if center_um is not None:
        p = set_window(p, center_um=center_um)  # 窗口移到預測的共振位置

    for attempt in range(max_retry + 1):
        # --- 第一層：模擬失敗（當機、記憶體不足等）也要當成「無效」，不要讓批次中斷 ---
        try:
            wvl_full, Toff_full, Ton_full, eps, modes = _simulate_once(p, make_plots, ref_cache)
        except Exception as e:                  # noqa: BLE001（刻意攔下所有錯誤）
            return _invalid_result(p, attempt, f"模擬失敗: {e}")

        # --- 窗口比 FSR 寬時，只取最深的那個凹谷附近來分析（掃 R 用）---
        if p.get("ISOLATE_DIP") and p.get("FSR_EST_UM", 0) > 0:
            wvl, T_off, T_on = isolate_main_dip(wvl_full, Toff_full, Ton_full,
                                                0.45 * p["FSR_EST_UM"])
        else:
            wvl, T_off, T_on = wvl_full, Toff_full, Ton_full

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
    fit = fit_dip(wvl, T_off, m_off)                         # 洛倫茲擬合（OFF 的凹谷）
    fit_on = fit_dip(wvl, T_on, m_on)                        # 洛倫茲擬合（ON 的凹谷），只為了量位移
    # 位移：PN 的位移只有約 0.1 nm，和頻率取樣間隔（0.005~0.02 nm）同量級，
    # 取離散最低點會有量化誤差；兩邊擬合都成功時，改用兩個擬合中心的差（可以細過取樣間隔）
    if np.isfinite(fit["lam0"]) and np.isfinite(fit_on["lam0"]):
        shift_um = fit_on["lam0"] - fit["lam0"]
    else:
        shift_um = m_on["lam0"] - m_off["lam0"]
    # 相位調變效率：優先用擬合的 OFF 共振波長，擬合失敗就用離散最低點
    lam0_ref = fit["lam0"] if np.isfinite(fit["lam0"]) else m_off["lam0"]
    dphi, eta, vpil = phase_efficiency(shift_um, lam0_ref, p)
    metrics = dict(
        valid=valid, reason=reason, retries=attempt,
        window_nm=(p["WVL_MIN"] * 1e3, p["WVL_MAX"] * 1e3),
        lam0_off_nm=m_off["lam0"] * 1e3,                   # 中間量：窗口追蹤用
        shift_nm=shift_um * 1e3,                           # 中間量：相位效率由它算出
        ER_dB=mod["ER_mod"],                               # 指標一：調變消光比（雷射放在 OFF/ON 差最大的波長）
        phase_eff_rad_per_V=eta, VpiL_Vcm=vpil)            # 指標二：相位調變效率

    # 只讓主程序存檔（用 MPI 平行時避免每個 rank 都寫檔）
    if make_plots and mp.am_master():
        plot_structure(eps, p)
        plot_transmission(wvl, T_off, T_on, m_off, m_on, mod, fit=fit)
        plot_transmission(wvl, T_off, T_on, m_off, m_on, mod,
                          filename="mrm_transmission_full.png", zoom=False)   # 完整範圍總覽
        np.savez("mrm_spectra.npz", wvl_um=wvl_full, T_off=Toff_full, T_on=Ton_full)
    if spectra_out and mp.am_master():                      # 完整光譜另存，之後可離線重新分析
        np.savez(spectra_out, wvl_um=wvl_full, T_off=Toff_full, T_on=Ton_full)
    return metrics


def _invalid_result(p, retries, reason):
    """模擬根本沒跑出結果時，回傳欄位一致的無效結果（數值全為 nan）。"""
    m = {k: np.nan for k in METRIC_KEYS}
    m.update(valid=False, reason=reason, retries=retries,
             window_nm=(p["WVL_MIN"] * 1e3, p["WVL_MAX"] * 1e3))
    return m


def figure_of_merit(metrics, penalty=-2.0):   # penalty = 無效設計的固定分數，要比任何有效設計的分數都低
    """
    把指標變成「一個分數」（越大越好），給最佳化器用。
    無效的設計（或量不到相位效率）回傳固定的懲罰分數（而不是 nan），讓最佳化器避開那一區；
    原因仍保留在 metrics["reason"]，之後才分得出「真的差」與「量不到」。

    下面的加權只是示範（ER 越大越好、VπL 越小越好），請依專題的實際目標修改，
    例如改成「ER 大於 3 dB 且 VπL 小於 1 V·cm 才算合格」。
    """
    if not metrics["valid"] or not np.isfinite(metrics["VpiL_Vcm"]):
        return penalty
    # ER 以 3 dB 為一單位、VπL 以 1 V·cm 為一單位，兩者大致同一量級再相減
    return metrics["ER_dB"] / 3.0 - metrics["VpiL_Vcm"] / 1.0


def example_scan_R(R_list):
    """
    示範（main 不會自動執行）：掃描不同的環半徑 R，並用上一組有效結果
    預測下一組的共振位置，讓掃描窗口跟著共振走。
    注意：predict_center 假設「同一個階數 m」，R 只能小幅變動（約 1% 內）。
    R 變動很大時（例如 3 -> 4 -> 5 µm）同一階的共振會跑到幾百 nm 外，
    請改用 sweep_resolution_gap.py 的 --Rs 選項（窗口取一個 FSR 寬）。
    用法：rows = example_scan_R([2.9, 3.0, 3.05])
    """
    cache, rows, last = {}, [], None            # cache = 參考光譜快取；rows = 每個 R 的結果；last = (R, OFF 共振波長 µm)
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
    if mp.am_master() and PARAMS.get("PN_FROM_PHYSICS", False):
        print_pn_physics(PARAMS)                # 先印出四步驟的計算結果，確認數字合理再等模擬
    metrics = evaluate(PARAMS, make_plots=True)
    if mp.am_master():
        print("=== Micro-ring modulator metrics ===")
        for k, v in metrics.items():
            print(f"{k:>14s} : {v}")
        print("Saved: mrm_structure.png, mrm_transmission.png, mrm_spectra.npz")


if __name__ == "__main__":
    main()
