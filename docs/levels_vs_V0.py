"""ccrm_formulas.md 第 6 节的插图：B=0 能级 E_k(V0) 随势深的变化，及其与广义本征值 ν_k 的对应。

运行（在 quasibound/ 下）：python3 docs/levels_vs_V0.py，输出 docs/levels_vs_V0.png（252Cf，Ni–Ren 几何，默认网格）。
上图：H(V0) = H0 - V0·F 的本征值随 V0 严格下降，各穿过 E = Q 一次；交点的横坐标就是广义本征问题
      (H0 - Q) x = V0 F x 的本征值 ν_k（candidates 的结果，圆点）。
下图：低于 Q 的能级数 N(V0)（直接对角化计数）恰在每个 ν_k 处加一，即 Sylvester 惯性定理给出的
      N(V0) = #{k : ν_k < V0}。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ccrm import Numerics, Potential, System, candidates, hamiltonian

INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"     # 与 check_ccrm.py 的图一致
CONTEXT, ACCENT, SEL = "#b9b8b2", "#2a78d6", 44                      # 其余位次（灰）、所选位次 44（蓝）

d = json.load(open(HERE.parent / "data" / "252Cf.json"))
s = System(d["Z"], d["A"], tuple(d["beta"]), [lv["E_MeV"] for lv in d["levels"]])
Q, g, num = d["Q_alpha_MeV"], Potential.ni_ren(s, 0.0), Numerics()
H0 = hamiltonian(s, Potential(0.0, g.R0, g.a), num)
F = H0 - hamiltonian(s, Potential(1.0, g.R0, g.a), num)              # F = -∂H/∂V0
nu = {r["rank"]: r["V0"] for r in candidates(s, g.R0, g.a, Q)}       # 位次 30–49 的广义本征值 ν_k
K0, K1 = 20, 60                                                      # 画出的位次
V = np.linspace(95.0, 195.0, 401)
E = np.array([eigh(H0 - v * F, eigvals_only=True, subset_by_index=(K0, K1)) for v in V])
assert (E[:, 0] < Q).all() and (E[:, -1] > Q).all()                 # 位次 K0 始终低于 Q、K1 始终高于 Q，计数才完整
N = K0 + (E < Q).sum(1)                                              # 低于 Q 的能级数 N(V0)

plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.facecolor": BG, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, facecolor=BG, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 2]})
ks = sorted(nu)

for j in range(E.shape[1]):                                          # 上图：能级曲线
    if K0 + j != SEL:
        ax.plot(V, E[:, j], color=CONTEXT, lw=0.9, zorder=1)
ax.plot(V, E[:, SEL - K0], color=ACCENT, lw=1.6, zorder=2)
ax.axhline(Q, color=MUTED, lw=0.8, zorder=1)
ax.plot([nu[k] for k in ks if k != SEL], [Q] * (len(ks) - 1), "o", ms=4.5, color=INK, mec=BG, mew=1.0, zorder=3)
ax.plot(nu[SEL], Q, "o", ms=6.5, color=ACCENT, mec=BG, mew=1.2, zorder=4)
ax.text(96.0, Q + 0.35, r"$E = Q_\alpha$ = %.4f MeV" % Q, color=MUTED, fontsize=8.5, va="bottom",
        bbox=dict(facecolor=BG, edgecolor="none", pad=1.0))
for k, dy, ha in ((30, -0.9, "center"), (36, -0.9, "center"), (41, -0.9, "center"), (49, -0.9, "center")):
    ax.text(nu[k], Q + dy, f"{k}", color=MUTED, fontsize=8, ha=ha, va="top")
ax.annotate(r"$k=44$:  $\nu_{44}$ = %.3f MeV" % nu[SEL], (nu[SEL], Q), xytext=(nu[SEL] + 4, Q - 5.0), color=INK,
            fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
ax.set_ylim(Q - 10, Q + 12)
ax.set_ylabel(r"$E_k(V_0)$ at $B=0$ (MeV)")
ax.set_title(r"$^{252}$Cf, Ni–Ren geometry: levels $E_k$ of $\mathbf{H}(V_0)=\mathbf{H}^{(0)}-V_0\mathbf{F}$", color=INK,
             fontsize=10, loc="left")
ax.legend(handles=[plt.Line2D([], [], color=CONTEXT, lw=0.9, label=f"$E_k(V_0)$, ranks {K0}–{K1}"),
                   plt.Line2D([], [], color=ACCENT, lw=1.6, label="rank 44 (selected state)"),
                   plt.Line2D([], [], ls="", marker="o", ms=4.5, color=INK, mec=BG,
                              label=r"$E_k(V_0)=Q$: $V_0=\nu_k$ (generalized eigenvalue, $k$ = 30–49)")],
          frameon=False, fontsize=8, loc="upper right")

bx.plot(V, N, drawstyle="steps-post", color=INK, lw=1.0, zorder=2)  # 下图：计数阶梯
bx.plot([nu[k] for k in ks if k != SEL], [k + 0.5 for k in ks if k != SEL], "o", ms=4.5, color=INK, mec=BG, mew=1.0,
        zorder=3)
bx.plot(nu[SEL], SEL + 0.5, "o", ms=6.5, color=ACCENT, mec=BG, mew=1.2, zorder=4)
bx.set_ylim(K0 + 7, K0 + 33)
bx.set_xlim(V[0], V[-1])
bx.set_xlabel(r"$V_0$ (MeV)")
bx.set_ylabel(r"$N(V_0)=\#\{k: E_k(V_0)<Q\}$")
bx.legend(handles=[plt.Line2D([], [], color=INK, lw=1.0, label="count by direct diagonalization"),
                   plt.Line2D([], [], ls="", marker="o", ms=4.5, color=INK, mec=BG,
                              label=r"$\nu_k$ ($k$ = 30–49) from $(\mathbf{H}^{(0)}-Q)\mathbf{x}=V_0\mathbf{F}\mathbf{x}$")],
          frameon=False, fontsize=8, loc="upper left")
fig.savefig(HERE / "levels_vs_V0.png", dpi=150, facecolor=BG)
print("写出", HERE / "levels_vs_V0.png")
