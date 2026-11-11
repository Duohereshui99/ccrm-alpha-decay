"""逐体系判定 α 衰变准束缚共振态（默认 252Cf），计算其波函数、Q、Γ，并做分层检验。

运行（在本目录下）：python3 check_ccrm.py [核素] [--rank k] [--ranks lo hi] [--check]，例如 check_ccrm.py 244Cf。
默认只做判定、计算和输出（第 2、3、6 步）；加 --check 时再做第 1、4、5 步的数值检验。
输入优先读 data/<核素>.json（实验数据，格式同 252Cf.json）；没有时取 data/cases.json 中 Ni–Ren 表 I 的输入。
本目录自成一体，不依赖目录外的文件。终端输出同时写入 results/<核素>_output.txt；另写 <核素>.json（全部数值）、
<核素>_wavefunctions.csv（波函数数值表）、<核素>_wavefunction.png（波函数分区总览图）及 8 张子图单图。
1. 单元（--check；各用另一种算法独立核对）：渐近出射库仑波 vs mpmath；外区 H(a)、B vs 自适应 DOP853 直接积分；
   内区 B=0 本征值 vs 打靶法；判定用的 V0（广义本征法）处直接对角化，E_k 应等于 Q_exp。
2. 判定：Ni–Ren 几何下逐个位次求 V0 使 E_k = Q_exp，列出各道节点数和 Wildermuth 权重；取各道节点数
   满足 n_ℓ = (G-ℓ)/2 且 G 壳总权重 ≥ 0.5 的态（README「共振态判定」）。各候选的观测量只作展示。
   判据没有给出唯一态时停止；人工判断后用 --rank 指定位次。
3. 结果：所选态的 Q、总 α 半衰期，以及各道（分支）的运动学量、内区概率、出射振幅、Γ_ℓ、分支比和
   分半衰期，与实验值对照（缺失的实验值记为 —；实验值不参与判定）。
4. 数值收敛（--check）：a、N、Rout、外区步长 dr 单独变化，并核对所选态的节点和权重不变。
5. 跟踪（--check）：V0、R0、a 小幅变化时，以参考本征矢跟踪同一个态，记录重叠、位次和观测量；并核对只求参考
   位次附近本征对的窗口选态与全范围选态给出同一结果。
6. 波函数：各道 u_ℓ 与总波函数 (Σ_ℓ|u_ℓ|²)^(1/2) 的数值表、分区总览图和各子图单图（内区、全空间、势垒衰减区、
   渐近区）。
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from functools import partial
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import brentq

try:                                                                     # 可选：限制 BLAS 线程数（这些矩阵只有 ~500 阶，单线程最快）
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(limits=None):
        """没装 threadpoolctl 时不限制线程数，其余照常运行。"""
        return nullcontext()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))           # matplotlib 缓存放在本目录
from ccrm import Numerics, Potential, System, candidates, coulomb_h, exterior, hamiltonian, potential, solve  # noqa: E402

def parse_args(argv=None):
    """命令行参数：核素名（默认 252Cf）、人工指定的位次、候选位次范围。"""
    ap = argparse.ArgumentParser(description="逐体系判定 α 衰变准束缚共振态，并计算、检验、输出。")
    ap.add_argument("nucleus", nargs="?", default="252Cf",
                    help="核素名，如 244Cf：读 data/<核素>.json；没有时取 data/cases.json 中 Ni–Ren 表 I 的输入")
    ap.add_argument("--rank", type=int, help="人工指定共振态位次（判据没有给出唯一态、人工判断之后使用）")
    ap.add_argument("--ranks", type=int, nargs=2, default=(30, 49), metavar=("LO", "HI"), help="候选位次范围，默认 30 49")
    ap.add_argument("--check", action="store_true", help="同时做单元检验、数值收敛和跟踪（多用几秒）")
    return ap.parse_args(argv)


def load_input(name):
    """读核素 name 的输入，键名同 data/252Cf.json；缺失的实验量不出现或为 None。

    优先读 data/<name>.json（实验数据，附来源）；没有时由 data/cases.json 中 Ni–Ren 表 I 的一行构造：
    Q = q₀，ε_ℓ = q₀ - q_ℓ（1 keV 精度），实验分支比取表中所引值；表中的计算值另存为理论参照 paper。
    """
    f = HERE / "data" / f"{name}.json"
    if f.exists():
        return dict(json.loads(f.read_text()), source=f"data/{name}.json")
    cases = {c["name"]: c for c in json.loads((HERE / "data" / "cases.json").read_text())["cases"]}
    if name not in cases:
        raise SystemExit(f"缺少输入：没有 data/{name}.json，data/cases.json 中也没有 {name}")
    c = cases[name]
    return dict(nucleus=name, Z=c["Z"], A=c["A"], beta=[c["beta2"], c["beta4"]], Q_alpha_MeV=c["q"][0],
                levels=[dict(jp=f"{2 * i}+", E_MeV=round(c["q"][0] - q, 6)) for i, q in enumerate(c["q"])],
                exp_branches_percent=c["exp_branches"], paper=dict(branches=c["paper_branches"], half_life_s=c["paper_half_life"]),
                source="data/cases.json（Ni–Ren 2010 表 I；q 精度 1 keV，没有实验半衰期）")


def nan_array(x):
    """含 None 的列表 → 浮点数组（None 记为 nan）。"""
    return np.array([np.nan if v is None else v for v in x], float)


ARGS = parse_args(None if __name__ == "__main__" else [])                 # 作为模块导入时取默认值
DATA = load_input(ARGS.nucleus)
SYS = System(DATA["Z"], DATA["A"], tuple(DATA["beta"]), [lv["E_MeV"] for lv in DATA["levels"]])
Q_EXP = DATA["Q_alpha_MeV"]                                              # 判定时调 V0 的目标值
T_EXP = (DATA["half_life_s"] / (DATA["alpha_branch_percent"] / 100)      # 实验 α 分半衰期 (s)；缺失为 nan
         if "half_life_s" in DATA and "alpha_branch_percent" in DATA else np.nan)
if "alpha_intensity_per_100_decays" in DATA:                             # 实验分支比 (%)，四道归一；缺失为 nan
    INT = nan_array(DATA["alpha_intensity_per_100_decays"])
    BR_EXP = 100 * INT / np.nansum(INT)
else:
    BR_EXP = nan_array(DATA.get("exp_branches_percent", [None] * len(SYS.l)))
GEOM = Potential.ni_ren(SYS, 0.0)                                        # Ni–Ren 几何：只用其 R0、a
WILDERMUTH = [(SYS.G - l) // 2 for l in SYS.l]                            # 各道应有的节点数 [11, 10, 9, 8]
HBAR_LN2 = 6.582119569e-22 * np.log(2)
P_ALPHA = 0.39                                                           # 预形成因子：Ni–Ren 对所有偶偶核取 0.39
YEAR = 365.2422 * 86400                                                  # 年 (s)，与 ENSDF/LiveChart 的换算一致
NAME = DATA["nucleus"]                                                   # 输出文件名前缀
RESULTS = HERE / "results"


def summary(res):
    """一个 Result 的主要观测量与诊断（键名 T_half_P039 对应 P_ALPHA = 0.39）。"""
    i = res.info
    return dict(Q=res.Q, rank=res.rank, gamma=res.gamma.tolist(), branches=(100 * res.gamma / res.gamma.sum()).tolist(),
                T_half_P039=res.half_life(P_ALPHA), P_eff=HBAR_LN2 / (res.gamma.sum() * T_EXP), nodes=i["nodes"],
                w_l0=i["w_l0"], w_G=i["w_G"], gaps=i["gaps"], tail=i["tail"], overlap=i["overlap"],
                overlap_next=i["overlap_next"], passes=i["passes"], A_abs=abs(res.amp).tolist())


def unit_checks():
    """用另一种算法独立核对 ccrm 的三个数值部件（V0 = 155 MeV，Ni–Ren 几何，E = Q_exp）。"""
    num, p, l, m = Numerics(), Potential.ni_ren(SYS, 155.0), SYS.l, len(SYS.l)
    k = np.sqrt((Q_EXP - SYS.ex) / SYS.c)
    eta, out = SYS.zz / (2 * SYS.c * k), {}

    def mp_h(L, et, rho):                                            # mpmath 的 H⁺ 及 dH⁺/dρ（递推 u'_L = S u_L - R u_L+1）
        with mp.workdps(30):
            h0, h1 = (mp.coulombg(j, et, rho) + 1j * mp.coulombf(j, et, rho) for j in (L, L + 1))
            return complex(h0), complex(((L + 1) / rho + et / (L + 1)) * h0 - mp.sqrt(1 + (et / (L + 1)) ** 2) * h1)

    for R in (120.0, 360.0, 1000.0):                                 # 1. 渐近库仑波
        h, dh = coulomb_h(l, eta, k * R)
        ref = np.array([mp_h(int(L), e, q * R) for L, e, q in zip(l, eta, k)])
        out[f"coulomb_h_max_rel Rout={R:g}"] = float(max(abs(h / ref[:, 0] - 1).max(), abs(dh / ref[:, 1] - 1).max()))

    def rhs(E):                                                      # [U; U']' = [U'; K U]，K = (V + ε + cℓ(ℓ+1)/r² - E)/c
        def f(r, y):
            K = potential(SYS, p, r)[0]
            K[np.diag_indices(m)] += SYS.ex - E + SYS.c * l * (l + 1) / r ** 2
            return np.concatenate((y[m * m:], (K / SYS.c @ y[:m * m].reshape(m, m)).ravel()))
        return f

    hp, dhp = coulomb_h(l, eta, k * num.Rout)                       # 2. 外区：自适应 DOP853 直接积分
    ya = solve_ivp(rhs(Q_EXP), (num.Rout, num.rm), np.r_[np.diag(hp).ravel(), np.diag(k * dhp).ravel()],
                   method="DOP853", rtol=1e-12, atol=1e-30).y[:, -1]
    Ha, dHa = ya[:m * m].reshape(m, m), ya[m * m:].reshape(m, m)
    La = np.linalg.solve(Ha.T, dHa.T).T.real                         # Re[H'(a) H(a)⁻¹]
    ext = exterior(SYS, p, num, Q_EXP)
    out["H_a_max_rel"] = float(abs(ext.Ha - Ha).max() / abs(Ha).max())
    out["B_max_rel"] = float(abs(ext.B - (La + La.T) / 2).max() / abs(La).max())

    e44 = eigh(hamiltonian(SYS, p, num), subset_by_index=(44, 44), eigvals_only=True)[0]   # 3. 内区：打靶法
    r0 = 1e-3                                                        # 正则解 u_ℓ ≈ r^(ℓ+1) 起步
    y0 = np.r_[np.diag(r0 ** (l + 1.0)).ravel(), np.diag((l + 1.0) * r0 ** l).ravel()]

    def det_du(E):                                                   # B = 0 的本征条件：det U'(a) = 0（列归一）
        du = solve_ivp(rhs(E), (r0, num.rm), y0, method="DOP853", rtol=1e-10, atol=1e-60).y[m * m:, -1].reshape(m, m)
        return np.linalg.det(du / abs(du).max(0))
    out["interior_E44_dev_MeV"] = float(e44 - brentq(det_du, e44 - 0.01, e44 + 0.01, xtol=1e-10))

    rows = candidates(SYS, p.R0, p.a, Q_EXP, ranks=range(40, 49))   # 4. 判定：广义本征法给出的 V0 处直接对角化
    out["candidates_E_dev_MeV"] = float(max(abs(eigh(hamiltonian(SYS, Potential(r["V0"], p.R0, p.a), num), eigvals_only=True,
                                                     subset_by_index=(r["rank"],) * 2)[0] - Q_EXP) for r in rows))
    return out


def identify():
    """逐个位次调到 Q_exp，标出满足判据的态；附各候选的观测量（只作展示）。"""
    rows = candidates(SYS, GEOM.R0, GEOM.a, Q_EXP, ranks=range(ARGS.ranks[0], ARGS.ranks[1] + 1))
    for r in rows:
        r["passes_rule"] = r["nodes"] == WILDERMUTH and r["w_G"] >= 0.5
        r.update({k: v for k, v in summary(solve(SYS, Potential(r["V0"], GEOM.R0, GEOM.a), r["rank"])).items()
                  if k in ("T_half_P039", "branches", "P_eff")})
    return rows


def convergence(p, rank, ref):
    variants = {"a=12, N=100": Numerics(rm=12.0, N=100), "a=16, N=140": Numerics(rm=16.0, N=140),
                "N=100": Numerics(N=100), "N=160": Numerics(N=160), "Rout=600": Numerics(Rout=600.0),
                "Rout=1000": Numerics(Rout=1000.0), "dr=0.35": Numerics(dr=0.35), "dr=0.7": Numerics(dr=0.7)}
    rows = {}
    for key, num in variants.items():
        r = summary(solve(SYS, p, rank, num))
        rows[key] = dict(r, dQ=r["Q"] - ref["Q"], T_rel=r["T_half_P039"] / ref["T_half_P039"] - 1,
                         br_rel=(np.array(r["branches"]) / ref["branches"] - 1).tolist(),
                         same_state=r["nodes"] == ref["nodes"] and abs(r["w_G"] - ref["w_G"]) < 1e-3)
    return rows


def tracking(p, ref):
    """从所选态 ref（Result）出发小幅改变 V0、R0、a，按本征矢重叠跟踪同一个态。窗口选态（只求 ref.rank ± window
    的本征对）与全范围选态（传 ref.c）各算一遍，记录二者位次是否相同及宽度的最大相对差。"""
    rows = []
    shifts = [("V0", d) for d in (-3.0, -1.0, 1.0, 3.0)] + [("R0", f) for f in (0.95, 0.98, 1.02, 1.05)] + \
             [("a", f) for f in (0.90, 0.95, 1.05, 1.10)]
    for name, x in shifts:
        q = Potential(p.V0 + x, p.R0, p.a) if name == "V0" else \
            Potential(p.V0, p.R0 * x, p.a) if name == "R0" else Potential(p.V0, p.R0, p.a * x)
        label = f"V0 {x:+g}" if name == "V0" else f"{name} x{x:g}"
        try:
            res, full = solve(SYS, q, ref), solve(SYS, q, ref.c)
            rows.append(dict(change=label, **summary(res), window_same_rank=res.rank == full.rank,
                             window_dG=float(np.max(abs(res.gamma / full.gamma - 1)))))
        except (ValueError, RuntimeError) as err:
            rows.append(dict(change=label, error=str(err)))
    return rows


def exterior_check(res):
    """外转折点以外的 ℓ=0 外区解：振荡半周期与振幅对照纯库仑单极的 WKB 预期。"""
    r_t, k0 = SYS.zz / Q_EXP, res.info["k"][0]
    sel = (res.r_ext >= 80) & (res.r_ext <= 120)
    r, u = res.r_ext[sel], res.u_ext[0, sel].real
    i = np.where(u[:-1] * u[1:] < 0)[0]
    zeros = r[i] - u[i] * (r[i + 1] - r[i]) / (u[i + 1] - u[i])           # 线性插值求零点
    mid = (zeros[:-1] + zeros[1:]) / 2
    half = np.diff(zeros) * k0 * np.sqrt(1 - r_t / mid) / np.pi            # 实测半周期 / (π / k_loc)
    j = np.argmin(abs(res.r_ext - 100))
    amp = abs(res.u_ext[0, j]) / (abs(res.amp[0]) * (1 - r_t / res.r_ext[j]) ** -0.25)   # |H⁺| ≈ (1 - r_t/r)^(-1/4)
    return dict(turning_point_fm=r_t, zeros_80_120fm=len(zeros), half_period_ratio_mean=half.mean(),
                half_period_ratio_max_dev=abs(half - 1).max(), amplitude_ratio_100fm=amp, u0_abs_100fm=abs(res.u_ext[0, j]))


class Tee:
    """把 print 同时写到终端和输出文件（配合 redirect_stdout 使用）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


class Timer:
    """各步墙钟用时：lap(名称) 记下距上一次 lap（或开始）的秒数，并返回"（用时 x s）"供打印。"""

    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.begin, self.laps = datetime.now(), {}

    def lap(self, name):
        now = time.perf_counter()
        self.laps[name], self.last = now - self.last, now
        return f"（用时 {self.laps[name]:.2f} s）"

    def total(self):
        return time.perf_counter() - self.start

    def summary(self):
        """各步用时与合计，一行文字。"""
        return "，".join(f"{k} {v:.2f} s" for k, v in self.laps.items()) + f"；合计 {self.total():.2f} s"


def channels(res):
    """所选态各道（分支）的量，均在 E = Q 处取值，按 ℓ = 0, 2, 4, 6 排列。"""
    l, q = SYS.l, res.Q - SYS.ex                                         # q_ℓ = Q - ε_ℓ：各道相对运动能 (MeV)
    return dict(
        l=l, jp=[lv["jp"] for lv in DATA["levels"]], eps=SYS.ex, q=q, k=res.info["k"], eta=res.info["eta"],
        r_t=(SYS.zz + np.sqrt(SYS.zz ** 2 + 4 * q * SYS.c * l * (l + 1))) / (2 * q),   # 外转折点：zz/r + cℓ(ℓ+1)/r² = q_ℓ
        nodes=res.info["nodes"],
        P=(res.c.reshape(len(l), -1) ** 2).sum(1),             # 内区道概率 P_ℓ = ∫₀^a u_ℓ² dr = Σ_n c_ℓn²（网格求积），Σ_ℓ = 1
        P_ext=res.info["p_ext"],                               # 外区模方积分 ∫_a^Rout |u_ℓ|² dr，Σ_ℓ = N_ext
        u_a=res.u_ext[:, 0].real, A=res.amp, gamma=res.gamma,                      # u_ℓ(a)、出射振幅 A_ℓ、宽度 Γ_ℓ
        br=100 * res.gamma / res.gamma.sum(), br_exp=BR_EXP,                       # 分支比 (%)；实验值四道归一
        T=HBAR_LN2 / (P_ALPHA * res.gamma), T_exp=T_EXP / (BR_EXP / 100))         # 分半衰期 T½,ℓ = ħ ln2/(P_α Γ_ℓ)


def fmt(x, spec):
    """格式化数值；缺失（None 或 nan）时显示为 —。"""
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def report(res, p, note=""):
    """打印所选态的结果：Q、总 α 半衰期和各道（分支）的量，并与实验对照；note 接在标题后；返回 channels(res)。"""
    ch, i, tot, T = channels(res), res.info, res.gamma.sum(), res.half_life(P_ALPHA)
    unc = DATA.get("Q_alpha_unc_MeV")
    print(f"== 结果：{NAME} → {DATA.get('daughter', f'Z={SYS.Z - 2}, A={SYS.A - 4}')} + α，位次 {res.rank} 的准束缚态{note}")
    print(f"  势参数     V0 = {p.V0:.6f} MeV，R0 = {p.R0:.6f} fm，a = {p.a:.6f} fm")
    print(f"  选态       各道节点 {i['nodes']}，w_G = {i['w_G']:.4f}，与上下相邻本征值的间距 {-i['gaps'][0]:+.3f} / "
          f"{i['gaps'][1]:+.3f} MeV，外区自洽 {i['passes']} 遍")
    print(f"  Q          {res.Q:.7f} MeV（模型）；实验 {Q_EXP:.7f}{f'({1e7 * unc:.0f})' if unc else ''} MeV")
    print(f"  ΣΓ         {tot:.4e} MeV；外区范数 N_ext = {i['n_ext']:.2e}（内区归一 ∫₀^a Σ u² dr = 1）")
    texp = (f"{T_EXP:.4e} s = {T_EXP / YEAR:.4g} y（总 T½ = {DATA['half_life_s'] / YEAR:.4g} y，"
            f"α 分支 {DATA['alpha_branch_percent']}%）" if np.isfinite(T_EXP) else "—")
    print(f"  α 半衰期   {T:.4e} s = {T / YEAR:.4g} y（P_α = {P_ALPHA}）；实验 α 分半衰期 {texp}")
    if np.isfinite(T_EXP):
        print(f"  计算/实验  {T / T_EXP:.4f}；使两者相等的 P_α = {HBAR_LN2 / (tot * T_EXP):.4f}")
    if "paper" in DATA:
        print(f"  论文计算值 T½ = {DATA['paper']['half_life_s']:.4e} s，分支比 {DATA['paper']['branches']} %"
              "（Ni–Ren 的理论结果，只作参照，不是实验值）")
    print("  各道运动学与边界量（E = Q；r_t 为单极库仑加离心势的外转折点）：")
    print("     l  J^pi  eps_l(MeV)  q_l(MeV)  k_l(1/fm)    eta_l  r_t(fm)  nodes  u_l(a)(fm^-1/2)")
    for j, l in enumerate(ch["l"]):
        print(f"    {l:2d}  {ch['jp'][j]:4s}  {ch['eps'][j]:10.6f}  {ch['q'][j]:8.6f}  {ch['k'][j]:9.6f}  "
              f"{ch['eta'][j]:7.4f}  {ch['r_t'][j]:7.3f}  {ch['nodes'][j]:5d}  {ch['u_a'][j]:+.5e}")
    print("  中间量：各道波函数的模方积分（道概率）。内区 P_l = ∫₀^a u_l² dr = Σ_n c_ln²（Lagrange 网格求积），Σ_l P_l = 1；")
    print("          外区 P_ext,l = ∫_a^Rout |u_l|² dr（与外区 RK 步同阶的求积），Σ_l P_ext,l = N_ext")
    print("     l         P_l     P_l(%)    P_ext,l")
    for j, l in enumerate(ch["l"]):
        print(f"    {l:2d}    {ch['P'][j]:.6f}  {100 * ch['P'][j]:9.4f}  {ch['P_ext'][j]:.3e}")
    print(f"   sum    {ch['P'].sum():.6f}  {100 * ch['P'].sum():9.4f}  {ch['P_ext'].sum():.3e}")
    print("  各道衰变量（分支）：T1/2_l = ħ ln2 / (P_α Γ_l)，即只有该分支时的半衰期；ratio = BR / BR_exp")
    print("     l  |A_l|(fm^-1/2)  Gamma_l(MeV)      BR(%)  BR_exp(%)   ratio     T1/2_l(s)  T1/2_l_exp(s)")
    for j, l in enumerate(ch["l"]):
        print(f"    {l:2d}  {abs(ch['A'][j]):14.5e}  {ch['gamma'][j]:12.5e}  {ch['br'][j]:9.5f}  "
              f"{fmt(ch['br_exp'][j], '9.5f'):>9s}  {fmt(ch['br'][j] / ch['br_exp'][j], '6.3f'):>6s}  "
              f"{ch['T'][j]:12.5e}  {fmt(ch['T_exp'][j], '13.5e'):>13s}", flush=True)
    return ch


def clean(o):
    """写 JSON 前转换：numpy 类型 → Python 类型，nan → None（标准 JSON 没有 NaN）。"""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and np.isnan(o) else o


def save_wavefunctions(res, path):
    """波函数数值表（CSV）：各道 u_ℓ 的实部、虚部与总波函数 u_tot = (Σ_ℓ|u_ℓ|²)^(1/2)；只含实际计算的点。"""
    r, u = np.r_[res.r, res.r_ext], np.c_[res.u, res.u_ext]              # 内区网格点（实数）+ 外区等步长网格（复数）
    cols = [r] + [f(u[i]) for i in range(len(SYS.l)) for f in (np.real, np.imag)] + [np.sqrt((abs(u) ** 2).sum(0))]
    names = ["r_fm"] + [f"{part}_u{l}" for l in SYS.l for part in ("Re", "Im")] + ["u_tot"]
    head = (",".join(names) + "\n"                                       # 列名放第一行，'#' 说明行在其后，便于直接读取
            f"# {NAME} alpha decay, rank {res.rank}, Q = {res.Q:.7f} MeV. Units: r in fm, u in fm^-1/2; "
            "interior normalization int_0^a sum_l |u_l|^2 dr = 1.\n"
            f"# r < {res.r_ext[0]:g} fm: interior Lagrange mesh points (u real); r >= {res.r_ext[0]:g} fm: exterior grid "
            f"(uniform step {res.r_ext[1] - res.r_ext[0]:.4f} fm). u_tot = (sum_l |u_l|^2)^(1/2).")
    np.savetxt(path, np.column_stack(cols), delimiter=",", fmt="%.10e", header=head, comments="")


# 作图样式：各道颜色与线型，墨色、次要色、网格色、底色；渐近区作图范围 (fm)
COLORS, STYLES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], ["-", "--", "-.", ":"]
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ASYM = (30.0, 130.0)


def wf_data(res):
    """作图数据：r、各道 u_ℓ（内区实数 + 外区复数）、总波函数 u_tot = (Σ_ℓ|u_ℓ|²)^(1/2) 和各道量。"""
    r, u = np.r_[res.r, res.r_ext], np.c_[res.u, res.u_ext]
    return dict(res=res, ch=channels(res), a=res.r_ext[0], r=r, u=u, tot=np.sqrt((abs(u) ** 2).sum(0)))


def panel_interior(ax, d, legend=True):
    """(a) 内区 u_ℓ(r)（线性坐标）与 u_tot。"""
    res, m = d["res"], len(d["res"].r)
    for i, l in enumerate(SYS.l):
        ax.plot(res.r, res.u[i], color=COLORS[i % 4], ls=STYLES[i % 4], lw=1.5,
                label=f"l = {l} ({d['ch']['nodes'][i]} nodes)")
    ax.plot(res.r, d["tot"][:m], color=INK, lw=1.0, label=r"total $(\Sigma_\ell|u_\ell|^2)^{1/2}$")
    ax.set(xlim=(0, d["a"]), ylim=(None, 1.4 * d["tot"][:m].max()), xlabel="r (fm)",
           ylabel=r"$u_\ell(r)$ (fm$^{-1/2}$)", title=f"(a) Interior, 0 < r < a = {d['a']:g} fm")
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)


def log_panel(ax, d, legend):
    """(b)(c) 共用：各道 |u_ℓ| 与 u_tot 的对数图，虚线标出匹配半径 a。"""
    for i, l in enumerate(SYS.l):
        ax.semilogy(d["r"], abs(d["u"][i]), color=COLORS[i % 4], ls=STYLES[i % 4], lw=1.5, label=f"l = {l}")
    ax.semilogy(d["r"], d["tot"], color=INK, lw=0.9, label="total")
    ax.axvline(d["a"], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(d["a"], 0.97, " a", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8.5, va="top")
    ax.set(ylim=(1e-19, 3), xlabel="r (fm)", ylabel=r"$|u_\ell(r)|$ (fm$^{-1/2}$)")
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)


def panel_full(ax, d, legend=False):
    """(b) 全空间 0 < r < R_out 总览，点线标出 ℓ=0 外转折点。"""
    log_panel(ax, d, legend)
    rt, r_out = d["ch"]["r_t"][0], d["res"].r_ext[-1]
    ax.axvline(rt, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(rt, 0.97, r" $r_t$ (l=0)", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8.5, va="top")
    ax.text(0.97, 0.66, r"outgoing waves: $|u_\ell| \to |A_\ell|$", transform=ax.transAxes, ha="right", color=MUTED,
            fontsize=8.5)
    ax.set(xlim=(0, r_out), title=f"(b) Full range, 0 < r < R_out = {r_out:g} fm")


def panel_barrier(ax, d, legend=False):
    """(c) 势垒衰减区，点线标出各道外转折点 r_t,ℓ（颜色同各道）。"""
    log_panel(ax, d, legend)
    rt = d["ch"]["r_t"]
    for i in range(len(SYS.l)):
        ax.axvline(rt[i], color=COLORS[i % 4], lw=0.9, ls=(0, (1, 2)))
    ax.text(rt.min(), 0.03, r"$r_{t,\ell}$ ", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8.5,
            va="bottom", ha="right")
    ax.set(xlim=(6, rt.max() + 12), title="(c) Barrier (decay) region, a and outer turning points")


def panel_asymptotic(ax, d, legend=False, j=0):
    """(d) 起：第 j 道（j 等于道数时为总波函数）在外转折点以外的出射波。纵轴乘 10^n 使振幅为 O(1)；
    虚线为 r → ∞ 时 |u| 的极限：各道为 |A_ℓ|，总波函数为 (Σ_ℓ|A_ℓ|²)^(1/2)。"""
    res, r, last = d["res"], d["r"], j == len(SYS.l)
    lim = np.sqrt((abs(res.amp) ** 2).sum()) if last else abs(res.amp[j])
    n = -int(np.round(np.log10(lim)))
    s, sel = 10.0 ** n, (r >= ASYM[0]) & (r <= ASYM[1])
    if last:
        ax.plot(r[sel], s * d["tot"][sel], color=INK, lw=1.2, label=r"$u_{tot}$")
        ax.axhline(s * lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), label=r"$(\Sigma_\ell|A_\ell|^2)^{1/2}$ ($r\to\infty$)")
        ax.set(ylim=(0, 2.5 * s * lim), ylabel=rf"$u_{{tot}}\times10^{{{n}}}$ (fm$^{{-1/2}}$)")
    else:
        y, c, l = d["u"][j], COLORS[j % 4], SYS.l[j]
        ax.plot(r[sel], s * y.real[sel], color=c, lw=1.1, label="Re u")
        ax.plot(r[sel], s * y.imag[sel], color=c, lw=1.0, ls="--", label="Im u")
        ax.plot(r[sel], s * abs(y[sel]), color=INK, lw=0.9, ls=":", label="|u|")
        ax.axhline(s * lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), label=rf"$|A_{{{l}}}|$ ($r\to\infty$)")
        ax.set(ylim=(-2.5 * s * lim, 2.5 * s * lim), ylabel=rf"$u_{{{l}}}\times10^{{{n}}}$ (fm$^{{-1/2}}$)")
    ax.axvline(d["ch"]["r_t"][0 if last else j], color=MUTED, lw=0.8, ls=(0, (1, 2)))
    tag = "total" if last else f"l = {SYS.l[j]}"
    ax.set(xlim=ASYM, xlabel="r (fm)", title=f"({'defghijk'[j]}) {tag}, asymptotic region")
    if legend:
        ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)


def panels():
    """各子图：(文件名后缀, 画图函数)，顺序与总览图 (a)–(h) 一致。"""
    tags = [f"l{l}" for l in SYS.l] + ["total"]
    return [("a_interior", panel_interior), ("b_full_range", panel_full), ("c_barrier", panel_barrier)] + \
           [(f"{'defghijk'[j]}_{tag}_asymptotic", partial(panel_asymptotic, j=j)) for j, tag in enumerate(tags)]


def plot(res, path):
    """波函数图：总览图写到 path，每个子图另存一张（同目录，文件名加子图后缀）；返回写出的全部图片。

    总览图上排为 (a) 内区 u_ℓ（线性）、(b) 全空间 |u_ℓ|、(c) 势垒衰减区 |u_ℓ|（对数），下排 (d) 起为各道
    与总波函数的渐近区。总波函数取 u_tot = (Σ_ℓ |u_ℓ|²)^(1/2)，即对角度和子核内部坐标积分后的径向几率幅。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": BG, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.titlesize": 10, "axes.titlecolor": INK})
    d, nch = wf_data(res), len(SYS.l)
    head = rf"$^{{{SYS.A}}}${NAME.lstrip('0123456789')} $\alpha$ decay, rank {res.rank}: Q = {res.Q:.4f} MeV"
    fig = plt.figure(figsize=(17, 9.2), facecolor=BG, layout="constrained")
    top, bottom = fig.subfigures(2, 1)                                   # 上下两排各自排版
    for sf in (top, bottom):
        sf.set_facecolor(BG)
    axes = list(top.subplots(1, 3)) + list(bottom.subplots(1, nch + 1))
    for ax, (key, draw) in zip(axes, panels()):
        draw(ax, d, legend=key[0] in "ad")                               # 总览图只在 (a)、(d) 放图例
    fig.suptitle(head + rf", $\Sigma\Gamma$ = {res.gamma.sum():.3e} MeV, $T_{{1/2}}$($P_\alpha$ = {P_ALPHA}) = "
                 rf"{res.half_life(P_ALPHA):.3e} s;  total $u_{{tot}} = (\Sigma_\ell|u_\ell|^2)^{{1/2}}$",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    files = [path]
    for key, draw in panels():                                           # 每个子图单独一张，带图例
        fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=BG, layout="constrained")
        draw(ax, d, legend=True)
        fig.suptitle(head, fontsize=9.5, color=MUTED)
        files.append(path.with_name(f"{path.stem}_{key}.png"))
        fig.savefig(files[-1], dpi=150, facecolor=BG)
        plt.close(fig)
    return files


def main():
    clock = Timer()                                                      # 各步用时（不含 Python 启动和库导入）
    RESULTS.mkdir(exist_ok=True)
    files = [RESULTS / f"{NAME}{tail}" for tail in ("_output.txt", ".json", "_wavefunctions.csv", "_wavefunction.png")]
    # 终端输出同时写入 files[0]
    with open(files[0], "w", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), threadpool_limits(limits=1):
        print(f"== 开始 {clock.begin:%Y-%m-%d %H:%M:%S}；体系 {NAME}：输入取自 {DATA['source']}")
        out = {"input": dict(source=DATA["source"], Q_exp=Q_EXP, T_alpha_exp_s=T_EXP, branches_exp=BR_EXP, ex=SYS.ex,
                             beta=list(SYS.beta), R0=GEOM.R0, a=GEOM.a, wildermuth_nodes=WILDERMUTH,
                             ranks=list(ARGS.ranks), rank_override=ARGS.rank)}
        if ARGS.check:
            out["unit"] = unit_checks()
            print("== 单元检验（另一种算法独立核对：库仑波与外区为最大相对差，内区与判定为能量差 (MeV)）" + clock.lap("单元检验"))
            print(json.dumps(out["unit"], indent=1), flush=True)
        out["candidates"] = rows = identify()
        print("== 候选（每个位次调 V0 使 E = Q_exp）：位次  V0  各道节点  w_l0n11  w_G22  判据  T½(P=0.39)  分支比"
              + clock.lap("判定"))
        for r in rows:
            print(f"  {r['rank']:3d} {r['V0']:8.3f} {r['nodes']} {r['w_l0']:.3f} {r['w_G']:.3f} {'选中' if r['passes_rule'] else '  - '} "
                  f"{r['T_half_P039']:.3e} {np.round(r['branches'], 4).tolist()}", flush=True)
        chosen = [r for r in rows if r["passes_rule"]]
        if ARGS.rank is not None:                                        # 人工指定位次，覆盖判据
            chosen = [r for r in rows if r["rank"] == ARGS.rank]
            print(f"== 人工指定位次 {ARGS.rank}（判据{'也选中' if chosen and chosen[0]['passes_rule'] else '没有选中'}该态）")
        if len(chosen) != 1:
            print(f"== 判据给出 {len(chosen)} 个态，需要人工判定；判定后用 --rank 指定位次" if ARGS.rank is None else
                  f"== 位次 {ARGS.rank} 不在候选表中（范围 {ARGS.ranks[0]}–{ARGS.ranks[1]}，且须能调到 E = Q_exp）")
            print(f"== 运行时长：{clock.summary()}（不含 Python 启动和库导入）")
            raise SystemExit(1)
        p = Potential(chosen[0]["V0"], GEOM.R0, GEOM.a)
        res = solve(SYS, p, chosen[0]["rank"])
        out["chosen"] = ref = dict(summary(res), V0=p.V0, R0=p.R0, a=p.a)
        ch = report(res, p, clock.lap("所选态计算"))
        out["channels"] = {k: v for k, v in ch.items() if k != "A"} | dict(A_re=ch["A"].real, A_im=ch["A"].imag)
        out["exterior_check"] = x = exterior_check(res)
        print(f"== 外区核对（ℓ=0）{clock.lap('外区核对')}：外转折点 {x['turning_point_fm']:.2f} fm；"
              f"80–120 fm 内 {x['zeros_80_120fm']} 个零点，半周期/WKB {x['half_period_ratio_mean']:.4f}"
              f"（最大偏差 {x['half_period_ratio_max_dev']:.1e}）；100 fm 处 |u0| = {x['u0_abs_100fm']:.3e}，"
              f"与 |A0|(1-r_t/r)^(-1/4) 之比 {x['amplitude_ratio_100fm']:.4f}", flush=True)
        if ARGS.check:
            num = Numerics()
            out["convergence"] = convergence(p, res.rank, ref)
            print(f"== 数值收敛（相对默认 a={num.rm:g}, N={num.N}, Rout={num.Rout:g}, dr={num.dr:g}）" + clock.lap("数值收敛"))
            for key, r in out["convergence"].items():
                print(f"  {key:12s} 同一态 {r['same_state']}  dQ {r['dQ']:+.1e}  T½ {r['T_rel']:+.1e}  分支比 {np.round(r['br_rel'], 6).tolist()}", flush=True)
            out["tracking"] = tracking(p, res)
            print(f"== 跟踪（参考本征矢；只求位次 {res.rank} ± {num.window} 的本征对，另以全范围选态核对）：变化  Q  位次  "
                  "重叠/次大  节点  w_G22  T½(P=0.39)  分支比  窗口与全范围的 ΔΓ/Γ" + clock.lap("跟踪"))
            for r in out["tracking"]:
                if "error" in r:
                    print(f"  {r['change']:9s} 失败：{r['error']}", flush=True)
                    continue
                print(f"  {r['change']:9s} {r['Q']:.4f} {r['rank']:3d} {r['overlap']:.4f}/{r['overlap_next']:.4f} {r['nodes']} "
                      f"{r['w_G']:.3f} {r['T_half_P039']:.3e} {np.round(r['branches'], 3).tolist()} "
                      f"{r['window_dG']:.0e}{'' if r['window_same_rank'] else '（位次不同！）'}", flush=True)
        save_wavefunctions(res, files[2])
        figs = plot(res, files[3])                                       # 总览图 + 各子图单图
        note = clock.lap("写出数据与图")
        out["timing_s"] = dict(clock.laps, 合计=clock.total(), 开始=f"{clock.begin:%Y-%m-%d %H:%M:%S}")
        files[1].write_text(json.dumps(clean(out), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"== 主要结果：Q = {res.Q:.7f} MeV，T½(α，P_α = {P_ALPHA}) = {res.half_life(P_ALPHA):.4e} s，"
              f"分支比 {np.round(ref['branches'], 4).tolist()} %（完整结果见上方「结果」一节）")
        print(f"== 输出文件{note}：")
        for f in files[:3] + figs:
            print(f"  {f.relative_to(ROOT)}", flush=True)
        print(f"== 运行时长：{clock.summary()}（不含 Python 启动和库导入）；结束 {datetime.now():%H:%M:%S}", flush=True)


if __name__ == "__main__":
    main()
