"""
sweep_resolution_gap.py
=======================
批次掃描「解析度 (RESOLUTION) × 間隙 (GAP)」，每跑完一組就把結果寫進 CSV。
CSV 的欄位順序與試算表 mrm_sweep_analysis.xlsx 的「掃描資料」分頁一致，
可以直接複製貼上，由試算表自動算出收斂判定與各項衍生數值。

兩種典型用法
------------
1) 解析度收斂測試（gap 固定，掃解析度）：
       python sweep_resolution_gap.py --resolutions 25 40 50 --gaps 0.20
   目的：看 Q 與共振波長會不會隨解析度提高而穩定下來。
        若解析度提高後 Q 明顯上升，代表原本的損耗有一部分是數值誤差。

2) gap 掃描（解析度固定，掃 gap）：
       python sweep_resolution_gap.py --resolutions 40 --gaps 0.10 0.12 0.15 0.20
   目的：找出凹谷最深（ER 最大）的耦合強度。

兩者也可同時給多個值，程式會跑完整個「解析度 × gap」格點。

特色
----
- 檢查點（checkpoint）：每跑完一組就寫入 CSV；中途中斷後重新執行同一個指令，
  已經在 CSV 裡的組合會自動跳過，不會重算。
- 窗口追蹤：每組的掃描窗口以「上一組有效結果的共振波長」為中心，
  因為解析度與 gap 改變時共振位置會移動。
- 無效結果（valid=False）也會寫入 CSV 並附原因，方便事後檢查。

執行：
    python sweep_resolution_gap.py --resolutions 25 40 50 --gaps 0.20
    mpirun -np 4 python sweep_resolution_gap.py ...        # 平行加速
"""

import argparse
import csv
import os
import time

import numpy as np
import meep as mp

from micro_ring_modulator_meep import PARAMS, evaluate

# CSV 欄位（順序必須與試算表「掃描資料」分頁的 A~R 欄一致）
FIELDS = ["resolution", "gap", "valid", "reason", "window_min_nm", "window_max_nm",
          "lam0_off_nm", "lam0_on_nm", "shift_nm", "fwhm_nm", "Q", "ER_dB",
          "OMA", "ER_mod_dB", "IL_dB", "lam_L_nm", "retries", "elapsed_s"]


def make_key(resolution, gap):
    """把 (解析度, gap) 轉成可比較的鍵，避免 0.2 與 0.2000001 被當成不同組合。"""
    return int(resolution), round(float(gap), 4)


def load_done(path):
    """讀取既有 CSV，回傳 {鍵: 該列內容}；檔案不存在就回傳空 dict。"""
    done = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done[make_key(row["resolution"], row["gap"])] = row
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


def main():
    ap = argparse.ArgumentParser(description="解析度 x gap 批次掃描")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[25, 40, 50],
                    help="要測試的解析度（每 µm 像素數）")
    ap.add_argument("--gaps", nargs="+", type=float, default=[0.20],
                    help="要測試的 gap (µm)")
    ap.add_argument("--out", default="sweep_results.csv", help="輸出的 CSV 檔名")
    ap.add_argument("--center-nm", type=float, default=1540.4,
                    help="第一組的掃描窗口中心 (nm)；預設取自先前實測的 OFF 共振位置")
    ap.add_argument("--decay", type=float, default=None,
                    help="覆寫 DECAY；不給就用 micro_ring_modulator_meep.PARAMS 的值")
    args = ap.parse_args()

    done = load_done(args.out)          # 讀一次就好（MPI 時避免各 rank 讀到不同內容）
    cache = {}                          # 參考光譜快取，傳給 evaluate 共用
    center_um = args.center_nm / 1000   # 目前的窗口中心 (µm)

    # 迴圈順序：外層 gap、內層解析度由小到大。
    # 這樣同一個 gap 的列會相鄰且解析度遞增，試算表才能逐列比較收斂。
    for gap in args.gaps:
        for res in sorted(args.resolutions):
            key = make_key(res, gap)

            # --- 檢查點：已經跑過的組合直接跳過，但仍用它更新窗口中心 ---
            if key in done:
                row = done[key]
                if str(row["valid"]).upper() == "TRUE" and row["lam0_off_nm"] != "":
                    center_um = float(row["lam0_off_nm"]) / 1000
                if mp.am_master():
                    print(f"[skip] resolution={res}, gap={gap}（CSV 已有結果）")
                continue

            # --- 組合這一組的參數，只改 RESOLUTION 與 GAP（與選填的 DECAY）---
            p = dict(PARAMS, RESOLUTION=res, GAP=gap)
            if args.decay is not None:
                p["DECAY"] = args.decay

            # --- 跑評估並計時 ---
            t0 = time.perf_counter()
            m = evaluate(p, ref_cache=cache, center_um=center_um)
            elapsed = time.perf_counter() - t0

            # --- 整理成 CSV 的一列 ---
            row = dict(resolution=res, gap=gap, elapsed_s=round(elapsed, 1),
                       window_min_nm=m["window_nm"][0], window_max_nm=m["window_nm"][1])
            for k in ("valid", "reason", "lam0_off_nm", "lam0_on_nm", "shift_nm",
                      "fwhm_nm", "Q", "ER_dB", "OMA", "ER_mod_dB", "IL_dB",
                      "lam_L_nm", "retries"):
                row[k] = m[k]
            row = {k: to_cell(v) for k, v in row.items()}

            # --- 只讓主程序寫檔並印進度 ---
            if mp.am_master():
                append_row(args.out, row)
                print(f"[done] resolution={res}, gap={gap}: valid={m['valid']}, "
                      f"Q={m['Q']:.0f}, ER={m['ER_dB']:.3f} dB, "
                      f"耗時 {elapsed / 60:.1f} 分鐘")

            # --- 只有可信的結果才拿來更新下一組的窗口中心 ---
            if m["valid"]:
                center_um = m["lam0_off_nm"] / 1000

    if mp.am_master():
        print(f"完成。結果在 {args.out}，可複製進試算表「掃描資料」分頁（A5 起，不含標題列）。")


if __name__ == "__main__":
    main()
