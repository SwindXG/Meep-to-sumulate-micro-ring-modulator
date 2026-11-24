"""
sweep_resolution_gap.py
=======================
批次掃描「環半徑 R × 間隙 gap × 解析度」，每跑完一組就把結果寫進 CSV。
CSV 的欄位順序與試算表 mrm_sweep_analysis.xlsx 的「掃描資料」分頁一致，
可以直接複製貼上，由試算表自動算出收斂判定與各項衍生數值。

三種典型用法
------------
1) 解析度收斂測試（gap、R 固定，掃解析度）：
       python sweep_resolution_gap.py --resolutions 25 40 50 --gaps 0.20
   目的：看位移、Q、ER 會不會隨解析度提高而穩定下來。

2) gap 掃描（解析度、R 固定，掃 gap）：
       python sweep_resolution_gap.py --resolutions 40 --gaps 0.10 0.12 0.15 0.20
   目的：找出凹谷最深（ER 最大）的耦合強度。

3) 環半徑 R 掃描（新增）：
       python sweep_resolution_gap.py --Rs 3 4 5 --gaps 0.12 --resolutions 25
   目的：看環自身的損耗（Q）會不會隨 R 變大而降低，使凹谷變深。
   R 一變，共振位置會跑得很遠，所以這個模式不用「預測共振位置」，而是：
     - 掃描窗口取約 1.1 個 FSR 寬（窗口內至少有一個完整的共振）；
     - 分析時只取最深的那個凹谷附近（ISOLATE_DIP），其餘共振丟掉。
   模擬時間主要由環內的光漏完要多久決定，與窗口寬度幾乎無關，所以放寬窗口不會變慢。
   注意：R 越大，計算區域越大、Q 通常也越高，每組都會比 R=3 慢很多；
         建議先用低解析度（25）從 R=4、5 開始。

特色
----
- 檢查點（checkpoint）：每跑完一組就寫入 CSV；中途中斷後重新執行同一個指令，
  已經在 CSV 裡的組合會自動跳過，不會重算。
- 每組的完整光譜另存成 .npz（預設放在 sweep_spectra/），之後可以不重跑 Meep 而重新分析。
- 無效結果（valid=False）也會寫入 CSV 並附原因，方便事後檢查。

執行：
    python sweep_resolution_gap.py --Rs 4 5 --gaps 0.12 --resolutions 25
    mpirun -np 4 python sweep_resolution_gap.py ...        # 平行加速
"""

import argparse
import csv
import os
import time

import numpy as np
import meep as mp

from micro_ring_modulator_meep import PARAMS, evaluate, resolve_pn_params

# CSV 欄位（順序必須與試算表「掃描資料」分頁的 A~W 欄一致）
FIELDS = ["resolution", "gap", "R", "valid", "reason", "window_min_nm", "window_max_nm",
          "lam0_off_nm", "shift_nm", "ER_dB", "phase_eff_rad_per_V", "VpiL_Vcm",
          "retries", "elapsed_s"]

# 直接從 evaluate 的回傳值複製的欄位
COPY_KEYS = ["valid", "reason", "lam0_off_nm", "shift_nm", "ER_dB", "phase_eff_rad_per_V",
             "VpiL_Vcm", "retries"]


def make_key(resolution, gap, R):
    """把 (解析度, gap, R) 轉成可比較的鍵，避免 0.2 與 0.2000001 被當成不同組合。"""
    # 解析度取整數；gap 四捨五入到 0.1 nm（4 位小數）；R 四捨五入到 1 nm（3 位小數）
    return int(resolution), round(float(gap), 4), round(float(R), 3)


def load_done(path):
    """
    讀取既有 CSV，回傳 {鍵: 該列內容}；檔案不存在就回傳空 dict。
    若標題列和現在的欄位不一致（例如舊版沒有 R 欄的檔案），直接停止並提示，
    以免把不同格式的資料混寫在同一個檔案裡。
    """
    done = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise SystemExit(
                    f"{path} 的欄位和目前版本不同（可能是舊版格式）。\n"
                    f"請把它改名備份，或用 --out 指定新的檔名，再重新執行。")
            for row in reader:
                done[make_key(row["resolution"], row["gap"], row["R"])] = row
    return done


def to_cell(v):
    """寫入 CSV 前的格式整理：nan 變空白（試算表比較好處理），numpy 數值轉成一般浮點數。"""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else float(v)
    return v


def append_row(path, row):
    """把一列結果附加到 CSV；檔案是新的就先寫標題列。utf-8-sig 讓 Excel 正確顯示中文。"""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def fsr_window(R, target_um, ng_est, window_fsr, spacing_nm):
    """
    依環半徑 R 決定掃描窗口（掃 R 的模式用）。
    FSR = λ^2 / (n_g * 2πR)；窗口寬度取 window_fsr 倍 FSR（預設 1.1），
    這樣不管共振落在哪裡，窗口內至少有一個完整的共振。
    回傳 (WVL_MIN, WVL_MAX, NFREQ, FSR)，長度單位 µm。
    NFREQ 依「頻率間隔約 spacing_nm」決定（頻率點多只增加記憶體，不增加模擬時間）。
    """
    fsr = target_um ** 2 / (ng_est * 2 * np.pi * R)    # FSR = λ² / (n_g · 環周長)，單位 µm
    width = window_fsr * fsr                           # 窗口寬度 = window_fsr 倍 FSR（預設 1.1，確保至少有一個完整共振）
    # 頻率點數 = 窗口寬度 (nm) / 想要的間隔 (nm)，再限制在 600~5000 之間：
    # 少於 600 點光譜太粗；多於 5000 點記憶體與 flux 監測的計算量變大，但解析度已不再有幫助
    nfreq = int(np.clip(np.ceil(width * 1e3 / spacing_nm), 600, 5000))
    return target_um - width / 2, target_um + width / 2, nfreq, fsr   # 窗口以 target_um 為中心左右對稱


def main():
    ap = argparse.ArgumentParser(description="R x gap x 解析度 批次掃描")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[25, 40, 50],
                    help="要測試的解析度（每 µm 像素數）")
    ap.add_argument("--gaps", nargs="+", type=float, default=[0.20], help="要測試的 gap (µm)")
    ap.add_argument("--Rs", nargs="+", type=float, default=None,
                    help="要測試的環半徑 (µm)；有給就進入「掃 R」模式（窗口取 1.1 個 FSR 寬）")
    ap.add_argument("--out", default="sweep_results_pn.csv", help="輸出的 CSV 檔名")
    ap.add_argument("--spectra-dir", default="sweep_spectra", help="存放每組完整光譜 (.npz) 的資料夾")
    ap.add_argument("--center-nm", type=float, default=1540.4,
                    help="（不掃 R 時）第一組的掃描窗口中心 (nm)；預設取自先前實測的 OFF 共振位置")
    ap.add_argument("--target-nm", type=float, default=1540.0,
                    help="（掃 R 時）窗口中心的目標波長 (nm)")
    ap.add_argument("--ng-est", type=float, default=2.9,
                    help="（掃 R 時）估算 FSR 用的群折射率；2D 模型 N_SI=2.8 約為 2.9")
    ap.add_argument("--window-fsr", type=float, default=1.1, help="（掃 R 時）窗口寬度是 FSR 的幾倍")
    ap.add_argument("--spacing-nm", type=float, default=0.02, help="（掃 R 時）頻率取樣間隔 (nm)")
    ap.add_argument("--decay", type=float, default=None,
                    help="覆寫 DECAY；不給就用 micro_ring_modulator_meep.PARAMS 的值")
    args = ap.parse_args()

    sweep_R = args.Rs is not None
    R_list = args.Rs if sweep_R else [PARAMS["R"]]

    done = load_done(args.out)          # 讀一次就好（MPI 時避免各 rank 讀到不同內容）
    if mp.am_master():
        # CSV 沒有記錄 PN 參數；提醒使用者：同一個 CSV 內所有列要用同一組 PN 參數
        # （舊的 sweep_results_v2.csv 是用整個環均勻 Δn=-0.004 的舊版簡化跑的，不要接著寫）
        rp = resolve_pn_params(PARAMS)           # PN_FROM_PHYSICS=True 時，這裡是由偏壓與摻雜算出來的值
        print(f"PN 參數：V_REV={PARAMS['V_REV']} V, NA={PARAMS['NA']:.1e}, ND={PARAMS['ND']:.1e} cm^-3, "
              f"ON 環帶寬 {rp['W_DEPL_ON']:.4f} µm（OFF 不放環帶）, DN_DEPL={rp['DN_DEPL']:.3e}, "
              f"ALPHA_DOPE={PARAMS['ALPHA_DOPE']}, X_JUNC={PARAMS['X_JUNC']}"
              f"（{args.out} 若是用別的 PN 參數或舊版簡化跑的，請改用新的 --out 檔名）")
    cache = {}                          # 參考光譜快取，傳給 evaluate 共用
    center_um = args.center_nm / 1000   # 不掃 R 時：目前的窗口中心 (µm)
    if mp.am_master():
        os.makedirs(args.spectra_dir, exist_ok=True)

    # 迴圈順序：外層 R、中層 gap、內層解析度由小到大。
    # 這樣 R 與 gap 都相同的列會相鄰且解析度遞增，試算表才能逐列比較收斂。
    for R in R_list:
        for gap in args.gaps:
            for res in sorted(args.resolutions):
                key = make_key(res, gap, R)

                # --- 檢查點：已經跑過的組合直接跳過 ---
                if key in done:
                    row = done[key]
                    if (not sweep_R and str(row["valid"]).upper() == "TRUE"
                            and row["lam0_off_nm"] != ""):
                        center_um = float(row["lam0_off_nm"]) / 1000   # 仍用它更新窗口中心
                    if mp.am_master():
                        print(f"[skip] R={R}, gap={gap}, resolution={res}（CSV 已有結果）")
                    continue

                # --- 組合這一組的參數 ---
                p = dict(PARAMS, RESOLUTION=res, GAP=gap, R=R, N_G=args.ng_est)   # 群折射率：窗口與相位效率共用
                if args.decay is not None:
                    p["DECAY"] = args.decay
                if sweep_R:
                    # 掃 R：窗口 = 1.1 個 FSR，並讓 evaluate 只分析最深的那個凹谷
                    wmin, wmax, nfreq, fsr = fsr_window(
                        R, args.target_nm / 1000, args.ng_est, args.window_fsr, args.spacing_nm)
                    p.update(WVL_MIN=wmin, WVL_MAX=wmax, NFREQ=nfreq,
                             ISOLATE_DIP=True, FSR_EST_UM=fsr)
                    center = None
                else:
                    center = center_um              # 不掃 R：窗口跟著上一組有效結果走

                # --- 跑評估並計時；完整光譜另存 ---
                spectra = os.path.join(args.spectra_dir, f"R{R}_gap{gap}_res{res}.npz")
                t0 = time.perf_counter()
                m = evaluate(p, ref_cache=cache, center_um=center, spectra_out=spectra)
                elapsed = time.perf_counter() - t0

                # --- 整理成 CSV 的一列 ---
                row = dict(resolution=res, gap=gap, R=R, elapsed_s=round(elapsed, 1),
                           window_min_nm=m["window_nm"][0], window_max_nm=m["window_nm"][1])
                for k in COPY_KEYS:
                    row[k] = m[k]
                row = {k: to_cell(v) for k, v in row.items()}

                # --- 只讓主程序寫檔並印進度 ---
                if mp.am_master():
                    append_row(args.out, row)
                    print(f"[done] R={R}, gap={gap}, resolution={res}: valid={m['valid']}, "
                          f"ER={m['ER_dB']:.3f} dB, shift={m['shift_nm']:+.4f} nm, "
                          f"VπL={m['VpiL_Vcm']:.3g} V·cm, "
                          f"耗時 {elapsed / 60:.1f} 分鐘")

                # --- 不掃 R 時，只有可信的結果才拿來更新下一組的窗口中心 ---
                if m["valid"] and not sweep_R:
                    center_um = m["lam0_off_nm"] / 1000

    if mp.am_master():
        print(f"完成。結果在 {args.out}，可複製進試算表「掃描資料」分頁（A5 起，不含標題列）。")


if __name__ == "__main__":
    main()
