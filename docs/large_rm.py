"""匹配半径越过外转折点（r_m 很大）时特征值法给出什么：外转折点之外是驻波，既不指数衰减，也不是模固定的出射波（252Cf）。

运行（在 quasibound/ 下）：python3 docs/large_rm.py，约 15 s。输出 docs/large_rm.png 与 docs/large_rm.json。
V0 = 155.146671 MeV 下的两个态：位次 44（本态，E = 6.217 MeV，外转折点 44.5 fm，Γ ~ 1e-29 MeV）与位次 45
（E = 13.55 MeV，外转折点 20.4 fm，势垒薄得多，尾部远高于舍入，可在双精度下看清驻波）。
(a)(b) ℓ=0 分量 |u_0(r)|：r_m = 60 fm 的大盒子本征矢；两段式（r_m = 14 fm）的出射解 H(r)A；本征矢的实数延拓 Re[H(r)A]，
       即 r_m 之外按实解继续积分（与内区实解在 r_m 处值和导数都相同）。
(c) 在 r_m 处读出的 Γ 与 r_m = 14 fm 之比（按参考向量在各 r_m 下选同一个态）。
(d) 实能量近似在 r 处略去的部分 ‖Im L‖/‖Re L‖，L = H'H⁻¹。由复 Wronskian H†H' - H'†H = 2iK（K = diag k_ℓ）守恒，
    Im L = H⁻† K H⁻¹，不经相减，势垒下 1e-22 量级也算得准。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ccrm import Numerics, Potential, System, exterior, mesh, solve

d = json.load(open(HERE.parent / "data" / "252Cf.json"))
s = System(d["Z"], d["A"], tuple(d["beta"]), [lv["E_MeV"] for lv in d["levels"]])
g = Potential.ni_ren(s, 0.0)
V0 = 155.146671
p = Potential(V0, g.R0, g.a)
RANKS, R_BIG = (44, 45), 60.0
RMS = (12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60)
n14 = Numerics()


def basis(N, a, r):
    """正则化 Lagrange–Legendre 基函数 φ_n(r)，形状 (N, len(r))。"""
    x = mesh(N, a)[0] / a
    r1 = np.atleast_1d(r)[None, :] + 1e-9                            # 避开恰在网格点上的 0/0
    return (((-1.0) ** (N + np.arange(1, N + 1)))[:, None] * r1 / (a * x[:, None]) * np.sqrt(a * x * (1 - x))[:, None]
            * legval(2 * r1 / a - 1, [0] * N + [1]) / (r1 - a * x[:, None]))


def on_mesh(res, nsrc, nbig):
    """把 nsrc 网格上的内区解放到 nbig 网格上（r ≥ nsrc.rm 处取 0），作选态的参考向量。"""
    rb, wb = mesh(nbig.N, nbig.rm)[:2]
    u = np.zeros((4, nbig.N))
    inn = rb < nsrc.rm - 1e-9
    u[:, inn] = res.c.reshape(4, nsrc.N) @ basis(nsrc.N, nsrc.rm, rb[inn])
    v = (u * np.sqrt(nbig.rm * wb)).ravel()
    return v / np.linalg.norm(v)


out, cur = {}, {}
for rank in RANKS:
    r14 = solve(s, p, rank, n14)
    E = r14.Q
    k = np.sqrt((E - s.ex) / s.c)
    eta = s.zz / (2 * s.c * k)
    r_t = (eta + np.sqrt(eta ** 2 + s.l * (s.l + 1))) / k            # 各道外转折点
    ratio, energy = [], []
    for a in RMS:                                                    # 大 r_m：按 r_m = 14 fm 的内区解选同一个态
        nb = Numerics(rm=float(a), N=int(round(120 / 14 * a)), kmax=320)
        rb = solve(s, p, on_mesh(r14, n14, nb), nb)
        ratio.append(rb.gamma.sum() / r14.gamma.sum())
        energy.append(rb.Q)
        if a == R_BIG:
            big, nbig = rb, nb
    ext = exterior(s, p, Numerics(rm=10.0, dr=0.05), E)             # 细网格出射解 H(r)，r = 10 → Rout
    r = ext.r[::-1]
    H, dH = ext.Y[::-1, :4], ext.Y[::-1, 4:]
    A = r14.amp / np.sqrt(1 + r14.info["n_ext"])                    # 两段式的出射振幅（全空间归一）
    u_out = H @ A                                                    # 出射解：r_m 之外的模趋于 |A_ℓ|
    Hinv = np.linalg.inv(H)
    imL = np.einsum("rji,j,rjk->rik", Hinv.conj(), k, Hinv).real     # Im L = H⁻† K H⁻¹（实对称）
    reL = np.einsum("rij,rjk->rik", dH, Hinv).real
    crit = np.linalg.norm(imL, 2, axis=(1, 2)) / np.linalg.norm(reL, 2, axis=(1, 2))
    far = r > r_t.max() + 2                                          # 外转折点外 Im L ~ k，直接相除也准：核对公式
    check = np.max(abs(imL[far] - np.einsum("rij,rjk->rik", dH[far], Hinv[far]).imag)) / np.max(abs(imL[far]))
    rr = np.linspace(0.05, R_BIG - 0.01, 6000)
    cur[rank] = dict(E=E, r_t=r_t, r=r, u_out=u_out[:, 0], u_re=u_out[:, 0].real, crit=crit, ratio=np.array(ratio),
                     rr=rr, u_big=(big.c.reshape(4, nbig.N) @ basis(nbig.N, R_BIG, rr))[0] / np.sqrt(1 + big.info["n_ext"]),
                     ri=np.linspace(0.05, 13.99, 1500))
    cur[rank]["u_in"] = (r14.c.reshape(4, n14.N) @ basis(n14.N, 14.0, cur[rank]["ri"]))[0] / np.sqrt(1 + r14.info["n_ext"])
    tail = (cur[rank]["rr"] > 30)
    floor = float(np.median(abs(cur[rank]["u_big"][tail])) / abs(cur[rank]["u_big"]).max())   # 大盒子本征矢 30 fm 外的中位幅度
    out[rank] = dict(E=E, gamma=r14.gamma.sum(), A=abs(A).tolist(), r_t=r_t.tolist(), big_rank=big.rank, big_N=nbig.N,
                     gamma_ratio=dict(zip(map(str, RMS), map(float, ratio))),
                     energy_spread=float(np.ptp(energy)), big_tail_median=floor, imL_formula_check=float(check),
                     imL_over_reL={str(x): float(np.interp(x, r, crit)) for x in (14, 20, 25, 30, 40)})
    print(f"位次 {rank}：E = {E:.6f} MeV，Γ = {r14.gamma.sum():.3e} MeV，|A_0| = {abs(A[0]):.3e}，外转折点 {r_t[0]:.1f} fm；"
          f"各 r_m 的能量之差 ≤ {np.ptp(energy):.1e} MeV；大盒子本征矢 30 fm 外中位幅度 / 峰值 = {floor:.1e}；"
          f"Im L 公式与直接相除之差（外转折点外）{check:.1e}")
    print("  Γ(r_m)/Γ(14 fm)：" + "，".join(f"{a}: {x:.4g}" for a, x in zip(RMS, ratio)))
    print("  ‖Im L‖/‖Re L‖：" + "，".join(f"{x} fm: {v:.1e}" for x, v in out[rank]["imL_over_reL"].items()))
json.dump(dict(note=__doc__.split("\n")[0], V0=V0, r_big=R_BIG, states={str(k): v for k, v in out.items()}),
          open(HERE / "large_rm.json", "w"), indent=1, ensure_ascii=False)

# ---------- 作图 ----------
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # 分类色，按固定顺序使用
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": BG, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlecolor": INK})
fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(10.5, 8.8), facecolor=BG, layout="constrained")


def wave(axx, rank, title, ylim):
    c = cur[rank]
    big_label = rf"eigenvector of the $r_m$ = {R_BIG:g} fm box (rank {out[rank]['big_rank']} there)"
    axx.semilogy(c["rr"], abs(c["u_big"]), color=C[0], lw=1.0, label=big_label)
    sel = c["r"] >= 14
    axx.semilogy(np.r_[c["ri"], c["r"][sel]], np.r_[abs(c["u_in"]), abs(c["u_out"][sel])], color=C[1], lw=1.8,
                 label=r"two-region ($r_m$ = 14 fm): outgoing $H(r)A$, fixed modulus")
    axx.semilogy(c["r"][sel], abs(c["u_re"][sel]), color=INK, lw=0.9, ls="--",
                 label=r"real continuation Re$[H(r)A]$ beyond $r_m$ = 14 fm")
    axx.axvline(c["r_t"][0], color=MUTED, lw=0.7)
    axx.text(c["r_t"][0] + 0.6, ylim[1] / 4, r"$r_{t,0}$", color=MUTED, fontsize=8)
    axx.set_xlim(0, R_BIG); axx.set_ylim(*ylim)
    axx.set_xlabel(r"$r$ (fm)"); axx.set_ylabel(r"$|u_0(r)|$ (fm$^{-1/2}$)")
    axx.set_title(title, loc="left")
    axx.legend(frameon=False, fontsize=7.2, loc="lower left")


wave(ax, 45, rf"(a) rank 45, $E$ = {cur[45]['E']:.2f} MeV, $\Gamma$ = {out[45]['gamma']:.1e} MeV (thin barrier)", (1e-9, 3))
wave(bx, 44, rf"(b) rank 44, $E$ = {cur[44]['E']:.3f} MeV, $\Gamma$ = {out[44]['gamma']:.1e} MeV (this state)", (1e-18, 3))
bx.text(22, 3e-6, f"box eigenvector (N = {out[44]['big_N']}): round-off floor,\n"
        f"median {out[44]['big_tail_median']:.0e} of the peak beyond 30 fm;\nthe true tail (dashed) is ~1e-15",
        color=C[0], fontsize=7.5)

for rank, color in ((45, C[2]), (44, C[0])):
    cx.semilogy(RMS, cur[rank]["ratio"], "-", color=color, lw=1.3, marker="o", ms=4.5, mec=BG, mew=1.0, label=f"rank {rank}")
    cx.axvline(cur[rank]["r_t"][0], color=color, lw=0.8, ls=":")
    dx.semilogy(cur[rank]["r"], cur[rank]["crit"], color=color, lw=1.3, label=f"rank {rank}")
    dx.axvline(cur[rank]["r_t"][0], color=color, lw=0.8, ls=":")
cx.axhline(1, color=MUTED, lw=0.7)
cx.text(11, 3e6, "rank 45: fails at its turning point (standing wave)\n"
        "rank 44: fails at 30 fm, before its turning point\n" r"($u(r_m)$ reaches the round-off level)", color=MUTED,
        fontsize=7.5, bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
cx.set_xlim(10, R_BIG + 1); cx.set_ylim(0.1, 3e11)
cx.set_xlabel(r"matching radius $r_m$ (fm)"); cx.set_ylabel(r"$\Gamma(r_m)\,/\,\Gamma(14\ \mathrm{fm})$")
cx.set_title(r"(c) width read out at $r_m$ (dotted: outer turning points $r_{t,0}$)", loc="left")
cx.legend(frameon=False, fontsize=7.5, loc="upper left")
dx.axhline(1, color=MUTED, lw=0.7)
dx.set_xlim(10, R_BIG); dx.set_ylim(1e-24, 1e3)
dx.set_xlabel(r"$r$ (fm)"); dx.set_ylabel(r"$\|\mathrm{Im}\,L\|\,/\,\|\mathrm{Re}\,L\|$")
dx.set_title(r"(d) part dropped by the real-energy approximation, $L=H'H^{-1}$", loc="left")
dx.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle(r"$^{252}$Cf, $V_0$ = 155.146671 MeV: beyond the outer turning point a real eigenvector is a standing wave, "
             "not an outgoing wave", color=INK, fontsize=10.5)
fig.savefig(HERE / "large_rm.png", dpi=150, facecolor=BG)
print("写出 docs/large_rm.png、docs/large_rm.json")
