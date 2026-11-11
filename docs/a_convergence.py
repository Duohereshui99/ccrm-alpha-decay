"""匹配半径 r_m（Numerics.rm；ccrm 注释沿用 R 矩阵习惯记作 a）的收敛检验：可观测量与波函数随 r_m 的变化（252Cf，位次 44）。

运行（在 quasibound/ 下）：python3 docs/a_convergence.py，约 11 s。输出：
    docs/a_convergence_abs.png     可观测量的绝对数值随 r_m 的变化
    docs/a_convergence_wf_abs.png  各 r_m 的波函数 u_tot(r) 叠画（全空间与边界附近放大）
    docs/a_convergence.png         可观测量相对平台中位数的偏差（对数坐标）
    docs/a_convergence_wf.png      波函数随 r 的偏差（各 r_m 与平台参考解比较）
    docs/a_convergence.json        全部数值
做法：每个 r_m 取两种网格密度（N = 120·r_m/14 及其 1.5 倍），调 V0 使自洽能量（含 B 修正）等于 Q_exp；波函数和出射
振幅按全空间归一（除以 sqrt(1 + N_ext)）。可观测量与平台 r_m = 11–20 fm（较密网格）的中位数比较，不以默认的
14 fm 为参考；波函数与平台内的 r_m = 12 fm（较密网格）比较，它的 u(r_m) 为峰值的 2e-3，舍入影响最小。
读出检验（rematch）：r_m > 14 fm 的同一个解，改在 r = 14 fm（u 尚大处）与外区匹配读出振幅。在 r_m 处读出的宽度
随 r_m 增大而失准、改在 14 fm 读出的不失准，说明失准来自 u(r_m) 太小时的舍入，而不是解本身随 r_m 变化。
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ccrm import Numerics, Potential, System, exterior, mesh, solve

d = json.load(open(HERE.parent / "data" / "252Cf.json"))
s = System(d["Z"], d["A"], tuple(d["beta"]), [lv["E_MeV"] for lv in d["levels"]])
Q, g = d["Q_alpha_MeV"], Potential.ni_ren(s, 0.0)
AS = (10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 30)            # 匹配半径 r_m (fm)
PLATEAU = (11, 12, 13, 14, 15, 16, 18, 20)
DENS, RANK, V0_START, R_FIX = 120 / 14, 44, 155.146671, (14.0, 100.0)
R_READ = 14.0                                                        # 读出检验：r_m 更大的解改在此处读出振幅
RR = np.arange(0.5, 360.25, 0.5)                                     # 比较波函数的公共网格（与外区网格对齐）


def at_q(num, v=V0_START):
    """调 V0 使 solve 的自洽能量等于 Q_exp。Newton 步的导数为 -cᵀFc，F 由形状因子直接构造。"""
    N = num.N
    F = np.zeros((4 * N, 4 * N))
    F.reshape(4, N, 4, N)[:, range(N), :, range(N)] = np.einsum(
        "is,ns,js->nij", s.U, expit(-(mesh(N, num.rm)[0][:, None] - g.R0 - s.shift) / g.a), s.U)
    for _ in range(10):
        res = solve(s, Potential(v, g.R0, g.a), RANK, num)
        step = (res.Q - Q) / (res.c @ F @ res.c)
        v += step
        if abs(step) < 1e-10:
            break
    return solve(s, Potential(v, g.R0, g.a), RANK, num), v


def u_at(res, num, rr):
    """全空间归一的波函数 u_ℓ(r)：r < r_m 用内区 Lagrange 展开，r ≥ r_m 取外区网格上的值。"""
    a, N = num.rm, num.N
    x, c = mesh(N, a)[0] / a, res.c.reshape(4, N)
    out = np.zeros((4, rr.size), complex)
    inn = rr < a - 1e-9
    r1 = rr[inn][None, :] + 1e-9                                     # 避开恰在网格点上的 0/0（奇数 N 时 r = r_m/2）
    out[:, inn] = c @ (((-1.0) ** (N + np.arange(1, N + 1)))[:, None] * r1 / (a * x[:, None])
                       * np.sqrt(a * x * (1 - x))[:, None] * legval(2 * r1 / a - 1, [0] * N + [1]) / (r1 - a * x[:, None]))
    j = np.rint((rr[~inn] - res.r_ext[0]) / (res.r_ext[1] - res.r_ext[0])).astype(int)
    out[:, ~inn] = res.u_ext[:, j]
    return out / np.sqrt(1 + res.info["n_ext"])


def rematch(res, num, v, rr=R_READ):
    """同一个解改在 rr（< r_m）处读出出射振幅：u(rr) 取内区展开，外区在同一能量、同一 V0 下由 Rout 传播到 rr，
    A = H(rr)⁻¹ u(rr)。精确算术下与在 r_m 处读出相同（内区实解在势垒区等于出射解的实部，虚部 ~1e-20 可略）。
    返回全空间归一的 |A_ℓ| 与 Γ_ℓ = 2 c k_ℓ |A_ℓ|²。"""
    ext = exterior(s, Potential(v, g.R0, g.a), replace(num, rm=rr), res.Q)
    A = abs(np.linalg.solve(ext.Ha, u_at(res, num, np.array([rr]))[:, 0]))
    return A, 2 * s.c * ext.k * A ** 2


runs, wf = [], {}
for fac in (1.0, 1.5):
    for a in AS:
        num = Numerics(rm=float(a), N=int(round(DENS * fac * a)))
        res, v = at_q(num)
        u = u_at(res, num, RR)
        E_fix = solve(s, Potential(V0_START, g.R0, g.a), RANK, num).Q     # 固定 V0 时的能量
        G = res.gamma
        run = dict(a=a, fac=fac, N=num.N, V0=v, E_fixV0=E_fix, T=res.half_life(), b=(G / G.sum()).tolist(),
                   A=(abs(res.amp) / np.sqrt(1 + res.info["n_ext"])).tolist(), tail=res.info["tail"],
                   n_ext=res.info["n_ext"], u_fix=[float(np.sqrt((abs(u[:, RR == r]) ** 2).sum())) for r in R_FIX])
        if a > R_READ:
            A_r, G_r = rematch(res, num, v)
            run.update(T_r=res.half_life() * G.sum() / G_r.sum(), b_r=(G_r / G_r.sum()).tolist(), A_r=A_r.tolist())
        runs.append(run)
        wf[(fac, a)] = u
        print(f"r_m = {a:2d} fm, N = {num.N:3d}: V0 = {v:.7f} MeV, T½ = {res.half_life():.6e} s, u(r_m)/max|u| = {res.info['tail']:.1e}",
              flush=True)

# 平台参考值：较密网格、r_m = 11–20 fm 的中位数
P = [r for r in runs if r["fac"] == 1.5 and r["a"] in PLATEAU]
ref = {k: np.median([np.asarray(r[k]) for r in P], axis=0) for k in ("V0", "E_fixV0", "T", "b", "A", "u_fix")}
rel = lambda x, y: np.abs(np.asarray(x) / y - 1).tolist()
for r in runs:
    r["dev"] = dict(V0=abs(r["V0"] - ref["V0"]), E_fixV0=abs(r["E_fixV0"] - ref["E_fixV0"]), T=rel(r["T"], ref["T"]),
                    b=rel(r["b"], ref["b"]), A=rel(r["A"], ref["A"]), u_fix=rel(r["u_fix"], ref["u_fix"]))
    if "T_r" in r:
        r["dev"].update(T_r=rel(r["T_r"], ref["T"]), b_r=rel(r["b_r"], ref["b"]), A_r=rel(r["A_r"], ref["A"]))
print(f"读出检验（T½ 与分支比相对平台中位数的最大偏差）：在 r_m 处读出 → 改在 r = {R_READ:g} fm 读出")
for r in runs:
    if "T_r" in r:
        print(f"  N = {r['fac']:g}×120·r_m/14, r_m = {r['a']:2d} fm: T½ {r['dev']['T']:.1e} → {r['dev']['T_r']:.1e}，"
              f"分支比 {max(r['dev']['b']):.1e} → {max(r['dev']['b_r']):.1e}")
json.dump(dict(note=__doc__.split("\n")[0], Q_exp=Q, plateau=PLATEAU, r_read=R_READ,
               reference={k: np.asarray(v).tolist() for k, v in ref.items()}, runs=runs),
          open(HERE / "a_convergence.json", "w"), indent=1, ensure_ascii=False)

# ---------- 图 1：可观测量随 r_m（相对偏差） ----------
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # 分类色，按固定顺序使用
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": BG, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlecolor": INK})
fig, axs = plt.subplots(3, 2, figsize=(10.0, 10.5), sharex=True, facecolor=BG, layout="constrained")
fine, coarse = ([r for r in runs if r["fac"] == f] for f in (1.5, 1.0))
read = [r for r in fine if "T_r" in r]                               # 有读出检验的较密网格计算
A_ = np.array(AS)
FLOOR = 1e-10                                                        # 0 偏差画在下限处
HOLLOW = dict(ls="none", marker="o", ms=7.5, mfc="none", mew=1.0)
READ_LABEL = f"same solutions, read out at r = {R_READ:g} fm"


def series(ax, y, color, label, ls="-"):
    ax.semilogy(A_, np.maximum(y, FLOOR), ls, color=color, lw=1.2, marker="o", ms=4.5, mec=BG, mew=1.0, label=label)


def reread(ax, key, i=None, color=INK, label=None):
    y = [r["dev"][key] if i is None else r["dev"][key][i] for r in read]
    ax.semilogy([r["a"] for r in read], np.maximum(y, FLOOR), mec=color, label=label, **HOLLOW)


def frame(ax, title, ylabel, ylim=(1e-9, 1)):
    ax.axvspan(10.5, 20.5, color=C[0], alpha=0.07, lw=0)
    ax.axvline(14, color=MUTED, lw=0.7)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)


ax = axs[0, 0]
series(ax, [r["dev"]["T"] for r in fine], C[0], r"N = 180·$r_m$/14, read out at $r_m$")
series(ax, [r["dev"]["T"] for r in coarse], C[1], r"N = 120·$r_m$/14 (default density)", ls="--")
reread(ax, "T_r", label=READ_LABEL + " (N = 180·$r_m$/14)")
frame(ax, r"(a) half-life $T_{1/2}$ at $E=Q_\alpha$", "relative deviation", (1e-9, 1e2))
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.text(14.2, 1e-8, r"default $r_m$ = 14 fm", color=MUTED, fontsize=7.5)

ax = axs[0, 1]
for i, l in enumerate(s.l):
    series(ax, [r["dev"]["b"][i] for r in fine], C[i], rf"$\ell={l}$")
    reread(ax, "b_r", i, C[i])
ax.plot([], [], mec=MUTED, label=f"hollow: read out at r = {R_READ:g} fm", **HOLLOW)
frame(ax, "(b) branching ratios (N = 180·$r_m$/14)", "relative deviation", (1e-9, 1e2))
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)

ax = axs[1, 0]
series(ax, [r["dev"]["V0"] for r in fine], C[0], r"$V_0$ giving $E=Q_\alpha$")
series(ax, [r["dev"]["E_fixV0"] for r in fine], C[1], r"$E$ at fixed $V_0$")
frame(ax, "(c) potential depth and energy (N = 180·$r_m$/14)", "absolute deviation (MeV)", (1e-9, 1e-2))
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

ax = axs[1, 1]
for i, l in enumerate(s.l):
    series(ax, [r["dev"]["A"][i] for r in fine], C[i], rf"$|A_{l}|$")
    reread(ax, "A_r", i, C[i])
ax.plot([], [], mec=MUTED, label=f"hollow: read out at r = {R_READ:g} fm", **HOLLOW)
frame(ax, r"(d) outgoing amplitudes $|A_\ell|$, full-space normalized", "relative deviation", (1e-9, 1e2))
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)

ax = axs[2, 0]
for i, r0 in enumerate(R_FIX):
    series(ax, [r["dev"]["u_fix"][i] for r in fine], C[i], rf"$u_{{\rm tot}}$ at $r$ = {r0:g} fm")
frame(ax, r"(e) wave function at fixed radii (N = 180·$r_m$/14)", "relative deviation")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_xlabel(r"matching radius $r_m$ (fm)")

ax = axs[2, 1]
ax.semilogy(A_, [r["tail"] for r in fine], "-", color=INK, lw=1.2, marker="o", ms=4.5, mec=BG, mew=1.0)
ax.axhspan(2e-13, 3e-12, color=MUTED, alpha=0.12, lw=0)                # 由读出误差反推：|δu(r_m)| ≈ 峰值的 2e-13–3e-12
ax.text(10.3, 8e-13, "round-off level of the eigenvector (~1e-12 of the peak)", color=MUTED, fontsize=7.5, va="center")
frame(ax, r"(f) boundary value $|u(r_m)|/\max|u|$", "ratio", (1e-14, 1))
ax.set_xlabel(r"matching radius $r_m$ (fm)")
fig.suptitle(r"$^{252}$Cf, rank 44: convergence in the matching radius $r_m$" "\n"
             r"(deviations from the median over $r_m$ = 11–20 fm; shaded: plateau; line: default $r_m$ = 14 fm)",
             color=INK, fontsize=10)
fig.savefig(HERE / "a_convergence.png", dpi=150, facecolor=BG)
plt.close(fig)

# ---------- 图 2：波函数随 r 的偏差（参考：r_m = 12 fm，较密网格） ----------
ref_u = wf[(1.5, 12)]
nr = np.sqrt((abs(ref_u) ** 2).sum(0))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True, facecolor=BG, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.35]})
ax.loglog(RR, nr, color=INK, lw=1.3)
ax.set_ylabel(r"$u_{\rm tot}(r)$ (fm$^{-1/2}$)")
ax.set_ylim(1e-16, 5)
ax.set_title(r"reference: $r_m$ = 12 fm, N = 154 (full-space normalized, $E=Q_\alpha$)", loc="left")
for i, a in enumerate((10, 14, 16, 20, 25)):
    u = wf[(1.0, a)] * np.sign(np.real(np.vdot(ref_u[:, :40].ravel(), wf[(1.0, a)][:, :40].ravel())))   # 对齐整体符号
    dv = np.sqrt((abs(u - ref_u) ** 2).sum(0)) / nr
    bx.loglog(RR, np.maximum(dv, 1e-12), color=C[i], lw=1.1, label=rf"$r_m$ = {a} fm" + (" (default)" if a == 14 else ""))
    k = np.argmin(abs(RR - a))
    bx.plot(RR[k], max(dv[k], 1e-12), "o", ms=5, color=C[i], mec=BG, mew=1.0)
bx.axhline(1e-4, color=MUTED, lw=0.7)
bx.text(0.55, 1.4e-4, "1e-4", color=MUTED, fontsize=7.5)
bx.text(40, 3e-1, "dot: r = $r_m$, where each run switches from the\ninterior expansion (left) to exterior propagation (right)",
        color=MUTED, fontsize=7.5)
bx.set_ylim(1e-8, 30)
bx.set_xlim(0.5, 360)
bx.set_xlabel(r"$r$ (fm)")
bx.set_ylabel(r"$\|u^{(r_m)}(r)-u^{\rm ref}(r)\| / \|u^{\rm ref}(r)\|$")
bx.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
fig.savefig(HERE / "a_convergence_wf.png", dpi=150, facecolor=BG)
plt.close(fig)

# ---------- 图 3：可观测量的绝对数值随 r_m ----------
# 每格：(取值函数, 改在 R_READ 读出的取值函数或 None, 标签)。纵轴以平台中位数为中心，范围只按 r_m ≤ YLIM_RM 的点定：
# 取其最大偏离的 1.25 倍与平台散布的 15 倍中较大者；更远的点画在边缘（三角，尖端指向真实值所在方向）
YLIM_RM = 25
ITEMS = [(lambda r: r["T"] / 1e8, lambda r: r["T_r"] / 1e8, r"$T_{1/2}$ at $E=Q_\alpha$ ($10^8$ s)"),
         (lambda r: r["V0"], None, r"$V_0$ giving $E=Q_\alpha$ (MeV)")] + \
        [(lambda r, i=i: 100 * r["b"][i], lambda r, i=i: 100 * r["b_r"][i], rf"branching ratio, $\ell={l}$ (%)")
         for i, l in enumerate(s.l)] + \
        [(lambda r: r["A"][0] * 1e16, lambda r: r["A_r"][0] * 1e16, r"$|A_0|$, full-space normalized ($10^{-16}$ fm$^{-1/2}$)"),
         (lambda r: r["u_fix"][0] * 1e5, None, r"$u_{\rm tot}$ at $r$ = 14 fm ($10^{-5}$ fm$^{-1/2}$)")]
fig, axs = plt.subplots(4, 2, figsize=(10.0, 12.0), sharex=True, facecolor=BG, layout="constrained")
for ax, (get, get_r, label) in zip(axs.flat, ITEMS):
    yf, yc = np.array([get(r) for r in fine]), np.array([get(r) for r in coarse])
    m = np.median([get(r) for r in fine if r["a"] in PLATEAU])
    spread = max(abs(get(r) - m) for r in fine if r["a"] in PLATEAU)
    near = A_ <= YLIM_RM
    half = max(1.25 * np.max(abs(np.r_[yf[near], yc[near]] - m)), 15 * spread)
    lo, hi = m - half, m + half
    ax.axvspan(10.5, 20.5, color=C[0], alpha=0.07, lw=0)
    ax.axvline(14, color=MUTED, lw=0.7)
    ax.axhline(m, color=MUTED, lw=0.7)
    for y, color, ls in ((yf, C[0], "-"), (yc, C[1], "--")):
        yp = np.clip(y, lo, hi)
        ax.plot(A_, yp, ls, color=color, lw=1.2)
        for x, y0, y1 in zip(A_, y, yp):
            ax.plot(x, y1, "^" if y0 > hi else "v" if y0 < lo else "o", ms=4.5 if lo <= y0 <= hi else 7, color=color,
                    mec=BG, mew=1.0, clip_on=False, zorder=4)
    if get_r:
        ax.plot([r["a"] for r in read], [get_r(r) for r in read], mec=INK, zorder=5, **HOLLOW)
    ax.set_ylim(lo, hi)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_title(label, loc="left")
    ax.text(0.02, 0.95, f"plateau median {m:.8g}\nspread ±{spread / abs(m):.0e} (relative)", transform=ax.transAxes,
            ha="left", va="top", color=MUTED, fontsize=7.5)
for ax in axs[-1]:
    ax.set_xlabel(r"matching radius $r_m$ (fm)")
axs[0, 0].legend(handles=[plt.Line2D([], [], color=C[0], marker="o", mec=BG, label=r"N = 180·$r_m$/14, read out at $r_m$"),
                          plt.Line2D([], [], color=C[1], ls="--", marker="o", mec=BG, label=r"N = 120·$r_m$/14 (default density)"),
                          plt.Line2D([], [], mec=INK, label=READ_LABEL + " (N = 180·$r_m$/14)", **HOLLOW),
                          plt.Line2D([], [], ls="none", marker="v", ms=7, color=MUTED, label="off scale, beyond the axis edge")],
                 frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle(r"$^{252}$Cf, rank 44: observables vs matching radius $r_m$ "
             r"(shaded: plateau; vertical line: default $r_m$ = 14 fm; horizontal line: plateau median)", color=INK, fontsize=10)
fig.savefig(HERE / "a_convergence_abs.png", dpi=150, facecolor=BG)
plt.close(fig)

# ---------- 图 4：各 r_m 的波函数叠画 ----------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.8), facecolor=BG, layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
for i, a in enumerate((10, 14, 16, 20, 25)):                      # 默认 14 fm 画成宽的半透明底带，其余细线压在上面
    u = wf[(1.0, a)]
    ut = np.sqrt((abs(u) ** 2).sum(0))
    k = np.argmin(abs(RR - a))
    for axx in (ax, bx):
        axx.plot(RR, ut, color=C[i], lw=4.5 if a == 14 else 1.0, alpha=0.45 if a == 14 else 1.0, zorder=1 if a == 14 else 2,
                 label=rf"$r_m$ = {a} fm" + (" (default, wide band)" if a == 14 else ""))
        axx.plot(RR[k], ut[k], "o", ms=5.5, color=C[i], mec=BG, mew=1.0, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.5, 360); ax.set_ylim(1e-16, 3)
ax.set_title(r"$u_{\rm tot}(r)$, full-space normalized, $E=Q_\alpha$ (dot: $r=r_m$)", loc="left")
ax.set_xlabel(r"$r$ (fm)"); ax.set_ylabel(r"$u_{\rm tot}(r)$ (fm$^{-1/2}$)")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
bx.set_yscale("log")
bx.set_xlim(8, 30); bx.set_ylim(1e-12, 1)
bx.set_title("zoom: barrier region around the matching radii", loc="left")
bx.set_xlabel(r"$r$ (fm)")
bx.annotate("$r_m$ = 25 fm run: interior expansion\nwhere u is ~1e-10 of the peak", xy=(23.5, 2.5e-9), xytext=(21.5, 3e-4),
            color=MUTED, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
fig.savefig(HERE / "a_convergence_wf_abs.png", dpi=150, facecolor=BG)
plt.close(fig)
print("写出 docs/ 下的 a_convergence_abs.png、a_convergence_wf_abs.png、a_convergence.png、a_convergence_wf.png、a_convergence.json")
