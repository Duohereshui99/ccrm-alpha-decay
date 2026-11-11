"""势垒区的指数衰减从何而来（252Cf，位次 44，V0 = 155.146671 MeV）：衰减由方程和边界条件决定，不由基函数决定。

运行（在 quasibound/ 下）：python3 docs/barrier_decay.py，约 2 s。输出 docs/barrier_decay.png 与 docs/barrier_decay.json。
(a) 全空间波函数 |u_ℓ(r)|：只在经典禁区（势垒下）指数衰减；外转折点之外是振幅趋于 |A_ℓ| 的出射波，不再衰减。
(b) ℓ=0 的局部衰减率 -d ln|u_0|/dr 与 WKB 的 κ + κ'/(2κ)：r < r_m 取内区本征矢，r ≥ r_m 取外区 RK8 解，两段同一规律。
(c) 同一组 Lagrange 基、同一个矩阵（B = 0）的三个本征矢与两个基函数：本征矢衰减多快由其能量决定，基函数本身不衰减。
(d) 边界条件只影响 r_m 附近：r_m = 20 fm 时 B = 0 与自洽 B 的本征矢之差（相对值）≈ exp(-2∫_r^{r_m} κ dr')。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ccrm import Numerics, Potential, System, exterior, hamiltonian, mesh, potential, solve

d = json.load(open(HERE.parent / "data" / "252Cf.json"))
s = System(d["Z"], d["A"], tuple(d["beta"]), [lv["E_MeV"] for lv in d["levels"]])
g = Potential.ni_ren(s, 0.0)
V0, RANK = 155.146671, 44
p = Potential(V0, g.R0, g.a)
num = Numerics()                                                     # 默认 r_m = 14 fm，N = 120
res = solve(s, p, RANK, num)
E = res.Q


def basis(N, a, r):
    """正则化 Lagrange–Legendre 基函数 φ_n(r)，形状 (N, len(r))；u(r) = Σ_n c_n φ_n(r)。"""
    x = mesh(N, a)[0] / a
    r1 = np.atleast_1d(r)[None, :] + 1e-9                            # 避开恰在网格点上的 0/0
    return (((-1.0) ** (N + np.arange(1, N + 1)))[:, None] * r1 / (a * x[:, None]) * np.sqrt(a * x * (1 - x))[:, None]
            * legval(2 * r1 / a - 1, [0] * N + [1]) / (r1 - a * x[:, None]))


def kappa(r, e=E):
    """各道对角有效势 W_ℓℓ 下的 WKB 衰减常数 κ_ℓ(r) = sqrt((W_ℓℓ - e)/c)，禁区外取 0；形状 (len(r), nch)。"""
    r = np.atleast_1d(r)
    w = np.einsum("rll->rl", potential(s, p, r)) + s.ex + s.c * s.l * (s.l + 1) / r[:, None] ** 2
    return np.sqrt(np.clip(w - e, 0, None) / s.c)


# ---------- 禁区与全空间波函数 ----------
rg = np.linspace(5, 60, 55001)
forbidden = {int(l): (float(rg[kappa(rg)[:, i] > 0].min()), float(rg[kappa(rg)[:, i] > 0].max())) for i, l in enumerate(s.l)}
norm = np.sqrt(1 + res.info["n_ext"])                               # 全空间归一
ri = np.linspace(0.05, num.rm - 0.02, 2000)
ui = res.c.reshape(4, num.N) @ basis(num.N, num.rm, ri) / norm       # 内区：本征矢的 Lagrange 展开
ext = exterior(s, p, Numerics(dr=0.02), E)                          # 外区：细步长 RK8，直接给出 u 与 u'
A = np.linalg.solve(ext.Ha, res.c.reshape(4, num.N) @ mesh(num.N, num.rm)[2]) / norm
y = ext.Y @ A                                                        # (点数, 2·nch)，Rout → r_m
re, ue, due = ext.r[::-1], y[::-1, :4].T, y[::-1, 4:].T

# ---------- (b) 局部衰减率 ----------
h = 1e-4
ri_b = np.linspace(forbidden[0][0] + 0.5, num.rm - 0.02, 400)
du = (res.c.reshape(4, num.N) @ (basis(num.N, num.rm, ri_b + h) - basis(num.N, num.rm, ri_b - h))) / (2 * h)
rate_int = -(du / (res.c.reshape(4, num.N) @ basis(num.N, num.rm, ri_b)))[0]
sel = (re >= num.rm) & (re <= forbidden[0][1] - 0.5)
rate_ext = (-np.real(due * ue.conj()) / abs(ue) ** 2)[0, sel]
rk = np.linspace(forbidden[0][0] + 0.3, forbidden[0][1] - 0.3, 4000)
k0 = kappa(rk)[:, 0]
wkb = k0 + np.gradient(k0, rk) / (2 * k0)                           # u ≈ κ^(-1/2) exp(-∫κ) 的局部衰减率
TABLE_R = (11, 12, 14, 16, 20, 25, 30, 35, 40)
table = []
for r0 in TABLE_R:
    num_rate = (rate_int[np.argmin(abs(ri_b - r0))] if r0 < num.rm else rate_ext[np.argmin(abs(re[sel] - r0))])
    table.append(dict(r=r0, kappa=float(np.interp(r0, rk, k0)), wkb=float(np.interp(r0, rk, wkb)), numeric=float(num_rate)))
xs = np.linspace(11, 43, 8001)
lnratio = float(np.log(abs(np.interp(11, ri, ui[0])) / abs(np.interp(43, re, abs(ue[0])))))
wkb_int = float(np.trapz(kappa(xs)[:, 0], xs) + 0.5 * np.log(kappa(43.0)[0, 0] / kappa(11.0)[0, 0]))

# ---------- (c) 同一组基、同一个矩阵的三个本征矢 ----------
h0 = hamiltonian(s, p, num)
e_all, v_all = eigh(h0)
picks = {"deep": int(np.argmin(abs(e_all + 20))), "this state": RANK, "above barrier": int(np.argmin(abs(e_all - 40)))}
rc = np.linspace(0.05, num.rm - 0.02, 1500)
B = basis(num.N, num.rm, rc)
eig_u = {k: np.sqrt(((v_all[:, j].reshape(4, num.N) @ B) ** 2).sum(0)) for k, j in picks.items()}
rn = mesh(num.N, num.rm)[0]
show_n = [int(np.argmin(abs(rn - r0))) for r0 in (6.0, 12.0)]
rf = np.linspace(0.01, num.rm - 0.001, 30000)                         # 基函数在其余网格点上为零，取相邻网格点之间的包络
starts = np.unique(np.r_[0, np.searchsorted(rf, rn)])
starts = starts[starts < rf.size]
env_r = (rf[starts] + rf[np.r_[starts[1:], rf.size] - 1]) / 2
env = [np.maximum.reduceat(abs(row), starts) for row in basis(num.N, num.rm, rf)[show_n]]

# ---------- (d) B = 0 与自洽 B 的本征矢（r_m = 20 fm） ----------
n20 = Numerics(rm=20.0, N=171)
r20 = solve(s, p, RANK, n20)
x0 = eigh(hamiltonian(s, p, n20), subset_by_index=(RANK, RANK))[1][:, 0]
x0 *= np.sign(x0 @ r20.c)
rd = np.linspace(10.5, 19.98, 600)
B20 = basis(n20.N, 20.0, rd)
uB, u0 = (r20.c.reshape(4, n20.N) @ B20)[0], (x0.reshape(4, n20.N) @ B20)[0]
dev = abs(u0 - uB) / abs(uB)
kd = kappa(np.linspace(rd[0], 20.0, 20001))[:, 0]
tail = cumulative_trapezoid(kd[::-1], dx=(20.0 - rd[0]) / 20000, initial=0)[::-1]   # ∫_r^{r_m} κ0
pred = np.exp(-2 * np.interp(rd, np.linspace(rd[0], 20.0, 20001), tail))

# ---------- 输出数值 ----------
out = dict(note=__doc__.split("\n")[0], E=E, forbidden=forbidden, rate_table=table,
           integrated_11_43=dict(ln_ratio=lnratio, wkb=wkb_int),
           tail_vs_A={str(r0): (abs(ue[:, np.argmin(abs(re - r0))])).tolist() for r0 in (50, 100, 200, 300)}, A=abs(A).tolist(),
           eigvecs={k: dict(rank=j, E=float(e_all[j]), tail=float(eig_u[k][-1] / eig_u[k].max())) for k, j in picks.items()},
           B0_vs_B={str(r0): dict(dev=float(np.interp(r0, rd, dev)), pred=float(np.interp(r0, rd, pred))) for r0 in (12, 14, 16, 18, 19)})
json.dump(out, open(HERE / "barrier_decay.json", "w"), indent=1, ensure_ascii=False)
print("禁区（对角有效势）：", {l: f"{a:.2f}–{b:.2f} fm" for l, (a, b) in forbidden.items()})
for t in table:
    print(f"r = {t['r']:2d} fm：κ0 = {t['kappa']:.3f}，WKB κ+κ'/(2κ) = {t['wkb']:.3f}，数值 -dln|u0|/dr = {t['numeric']:.3f} fm⁻¹")
print(f"11→43 fm：ln|u0(11)/u0(43)| = {lnratio:.3f}，WKB ∫κ + ½ln(κ2/κ1) = {wkb_int:.3f}")
for k, v in out["eigvecs"].items():
    print(f"本征矢 {k}：位次 {v['rank']}，E = {v['E']:.2f} MeV，|u(r_m)|/max|u| = {v['tail']:.1e}")
for r0, v in out["B0_vs_B"].items():
    print(f"r_m = 20 fm，r = {r0} fm：|u_B0 - u_B|/|u_B| = {v['dev']:.2e}，exp(-2∫κ) = {v['pred']:.2e}")

# ---------- 作图 ----------
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # 分类色，按固定顺序使用
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": BG, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlecolor": INK})
fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(10.5, 8.6), facecolor=BG, layout="constrained")
lo, hi = forbidden[0]

# (a) 全空间波函数
for i, l in enumerate(s.l):
    ax.semilogy(np.r_[ri, re], np.r_[abs(ui[i]), abs(ue[i])], color=C[i], lw=1.2, label=rf"$\ell={l}$")
    ax.axhline(abs(A[i]), color=C[i], lw=0.8, ls=":")
ax.axvspan(lo, hi, color=MUTED, alpha=0.08, lw=0)
ax.axvline(num.rm, color=MUTED, lw=0.7)
ax.text(num.rm + 0.8, 0.3, r"$r_m$", color=MUTED, fontsize=8)
ax.text((lo + hi) / 2, 3e-19, "classically forbidden (ℓ=0)", color=MUTED, fontsize=8, ha="center")
ax.text(118, abs(A[0]) * 2.2, r"dotted: $|A_\ell|$", color=MUTED, fontsize=8, ha="right")
ax.set_xlim(0, 120); ax.set_ylim(1e-19, 3)
ax.set_xlabel(r"$r$ (fm)"); ax.set_ylabel(r"$|u_\ell(r)|$ (fm$^{-1/2}$)")
ax.set_title("(a) decays only under the barrier; outside: outgoing wave, amplitude → $|A_\\ell|$", loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)

# (b) 局部衰减率与 WKB
bx.plot(rk, k0, color=MUTED, lw=1.0, ls="--", label=r"WKB $\kappa_0=\sqrt{(W_{00}-E)/c}$")
bx.plot(rk, wkb, color=INK, lw=1.0, ls=":", label=r"WKB $\kappa_0+\kappa_0'/(2\kappa_0)$")
bx.plot(ri_b, rate_int, color=C[0], lw=2.2, label=r"eigenvector, $r<r_m$ (Lagrange basis)")
bx.plot(re[sel], rate_ext, color=C[1], lw=1.4, label=r"exterior RK8 solution, $r\geq r_m$")
bx.axvline(num.rm, color=MUTED, lw=0.7)
bx.set_xlim(lo, hi); bx.set_ylim(0, 2.2)
bx.set_xlabel(r"$r$ (fm)"); bx.set_ylabel(r"$-d\ln|u_0|/dr$ (fm$^{-1}$)")
bx.set_title(r"(b) local decay rate of $u_0$ vs WKB", loc="left")
bx.legend(frameon=False, fontsize=7.5, loc="upper right")

# (c) 同一组基的三个本征矢与两个基函数
for k, color in zip(picks, (C[2], C[0], C[3])):
    j = picks[k]
    cx.semilogy(rc, eig_u[k] / eig_u[k].max(), color=color, lw=1.3,
                label=f"eigenvector, rank {j}, E = {e_all[j]:.1f} MeV ({k})")
for n, e_n, ls in zip(show_n, env, ("--", ":")):
    cx.semilogy(env_r, e_n / e_n.max(), color=MUTED, lw=1.0, ls=ls,
                label=rf"basis function $|\varphi_n|$, $r_n$ = {rn[n]:.1f} fm (envelope)")
cx.set_xlim(0, num.rm); cx.set_ylim(1e-9, 3)
cx.set_xlabel(r"$r$ (fm)"); cx.set_ylabel("normalized to max 1")
cx.set_title(r"(c) same basis, same matrix ($B=0$, $r_m$ = 14 fm)", loc="left")
cx.legend(frameon=False, fontsize=7.2, loc="lower left")

# (d) B = 0 与自洽 B
dx.semilogy(rd, dev, color=C[0], lw=1.6, label=r"$|u^{B=0}_0-u^{B}_0|\,/\,|u^{B}_0|$")
dx.semilogy(rd, pred, color=INK, lw=1.0, ls="--", label=r"$\exp(-2\int_r^{r_m}\kappa_0\,dr')$")
dx.set_xlim(rd[0], 20); dx.set_ylim(1e-13, 3)
dx.set_xlabel(r"$r$ (fm)"); dx.set_ylabel("relative difference")
dx.set_title(r"(d) boundary condition matters only near $r_m$ ($r_m$ = 20 fm)", loc="left")
dx.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle(rf"$^{{252}}$Cf, rank {RANK}, $E$ = {E:.4f} MeV: exponential decay comes from the equation, not from the basis",
             color=INK, fontsize=10.5)
fig.savefig(HERE / "barrier_decay.png", dpi=150, facecolor=BG)
print("写出 docs/barrier_decay.png、docs/barrier_decay.json")
