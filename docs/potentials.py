"""各项势能的径向形状与大小，以及匹配半径 r_m 是否须在核势力程之外（252Cf，Ni–Ren 几何，位次 44 的 V0）。

运行（在 quasibound/ 下）：python3 docs/potentials.py，约 4 s。输出 docs/potentials.png 与 docs/potentials.json。
(a) 核势：有限位移算符四个本征取向的分量 -V0 f_s(r)（表面在 R0 + δ_s）与 ℓ=0 道的对角矩阵元；
(b) 各道有效势 V_ℓℓ(r) + cℓ(ℓ+1)/r² + ε_ℓ 与 E = Q_α；
(c) 各项的大小（对数坐标）：核势按 exp(-r/a) 衰减（a 为弥散），库仑耦合只按 r^-(λ+1) 衰减；
(d) 调 V0 使 E = Q_α 后 T½ 随 r_m 的变化：外区含全部势（本程序）与外区去掉核势（标准 R 矩阵的假设：r_m 之外核势
    可以忽略）。本程序的外区从 Rout 向内数值传播完整的耦合势，r_m 之外的核势已精确计入，所以不要求 r_m 在核势力程
    之外；须在核势力程之外的是出射边界条件所在的 Rout。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ccrm
from ccrm import Numerics, Potential, System, mesh, potential, solve

d = json.load(open(HERE.parent / "data" / "252Cf.json"))
s = System(d["Z"], d["A"], tuple(d["beta"]), [lv["E_MeV"] for lv in d["levels"]])
Q, g = d["Q_alpha_MeV"], Potential.ni_ren(s, 0.0)
V0, RANK = 155.146671, 44
p = Potential(V0, g.R0, g.a)
k = np.sqrt((Q - s.ex) / s.c)
eta = s.zz / (2 * s.c * k)
r_t = (eta + np.sqrt(eta ** 2 + s.l * (s.l + 1))) / k                # 各道外转折点（点库仑 + 离心）

# ---------- 势的各项（大小，MeV） ----------
nuc = lambda r, sh: V0 * expit(-(r - g.R0 - sh) / g.a)               # 位移为 sh 的取向的核势 V0·f_s
tip = lambda r: nuc(r, s.shift.max())                                # 尖端取向；|V_N,ℓℓ'| = V0|Σ_s U_ℓs U_ℓ's f_s| ≤ V0 max_s f_s
vn00 = lambda r: expit(-(np.atleast_1d(r)[:, None] - g.R0 - s.shift) / g.a) @ s.U[0] ** 2 * V0   # ℓ=0 对角矩阵元的大小
coul0 = lambda r: np.where(r < s.Rc, s.zz / (2 * s.Rc) * (3 - (r / s.Rc) ** 2), s.zz / r)
rho = lambda r, lam: np.where(r < s.Rc, r ** lam / s.Rc ** (lam + 1), s.Rc ** lam / r ** (lam + 1))
quad = lambda r: 3 * s.zz / 5 * s.beta[0] * rho(r, 2) * s.ang[2][0, 1]   # 库仑四极耦合 ℓ=0↔2 矩阵元
hexa = lambda r: 3 * s.zz / 9 * s.beta[1] * rho(r, 4) * s.ang[4][0, 2]   # 库仑十六极耦合 ℓ=0↔4 矩阵元


# ---------- (d) 外区含 / 不含核势时 T½ 随 r_m ----------
def half_life_at_q(num, v=V0):
    """调 V0 使 solve 的自洽能量等于 Q_exp（Newton，导数 -cᵀFc），返回 T½。"""
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
    return solve(s, Potential(v, g.R0, g.a), RANK, num).half_life()


RMS = (10, 11, 12, 13, 14, 15, 16, 18, 20, 22)
dens = lambda rm: Numerics(rm=float(rm), N=int(round(120 / 14 * rm)))  # 默认网格密度
full_ext = ccrm.exterior
T_full = np.array([half_life_at_q(dens(rm)) for rm in RMS])
ccrm.exterior = lambda s_, p_, num, E: full_ext(s_, Potential(0.0, p_.R0, p_.a), num, E)   # 外区去掉核势，其余不变
try:
    T_cut = np.array([half_life_at_q(dens(rm)) for rm in RMS])
finally:
    ccrm.exterior = full_ext
T_ref = np.median(T_full[[rm in range(11, 21) for rm in RMS]])     # 平台 r_m = 11–20 fm 的中位数
rows = [dict(r_m=rm, VN_tip=float(tip(rm)), VN_00=float(vn00(rm)[0]), dT_full=float(tf / T_ref - 1),
             dT_cut=float(tc / T_ref - 1), dT_drop=float(tc / tf - 1)) for rm, tf, tc in zip(RMS, T_full, T_cut)]
for r in rows:
    print(f"r_m = {r['r_m']:2d} fm：V0·f_tip = {r['VN_tip']:.1e} MeV，|V_N,00| = {r['VN_00']:.1e} MeV；T½ 相对平台  "
          f"外区含核势 {r['dT_full']:+.1e}，外区去掉核势 {r['dT_cut']:+.1e}；去掉核势带来的变化 {r['dT_drop']:+.1e}")
RADII = (14, 20, 25, 30, ccrm.Numerics.Rout)
for r0 in RADII:
    print(f"r = {r0:5.1f} fm：核势（尖端取向）{tip(r0):.1e} MeV，库仑单极 {coul0(r0):.2f} MeV，"
          f"库仑四极耦合 {abs(quad(r0)):.1e} MeV")
json.dump(dict(note=__doc__.split("\n")[0], V0=V0, R0=g.R0, diffuseness=g.a, delta=s.shift.tolist(), T_ref=T_ref, rows=rows,
               radii={str(r0): dict(VN_tip=float(tip(r0)), VN_waist=float(nuc(r0, s.shift.min())), VN_00=float(vn00(r0)[0]),
                                    coulomb0=float(coul0(r0)), quad02=float(abs(quad(r0))), hexa04=float(abs(hexa(r0))))
                      for r0 in RADII}),
          open(HERE / "potentials.json", "w"), indent=1, ensure_ascii=False)

# ---------- 作图 ----------
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # 分类色，按固定顺序使用
shade = lambda t: tuple((1 - t) * np.array(to_rgb(C[1])) + t * np.array(to_rgb(BG)))   # 核势色调由浅到深
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": BG, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlecolor": INK})
fig = plt.figure(figsize=(10.5, 12.6), facecolor=BG, layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.8])
ax, bx, cx, dx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])


def vline(axx, x, label, y, **kw):
    axx.axvline(x, color=MUTED, lw=0.7)
    axx.text(x, y, " " + label, color=MUTED, fontsize=8, **kw)


# (a) 核势：四个取向分量与 ℓ=0 对角矩阵元（线性坐标）
r = np.linspace(0.02, 20, 2000)
for j, (sh, t) in enumerate(zip(s.shift, (0.62, 0.42, 0.22, 0.0))):
    ax.plot(r, -nuc(r, sh), color=shade(t), lw=1.6,
            label=rf"$\delta_s$ = {sh:+.2f} fm" + (" (waist)" if j == 0 else " (tip)" if j == len(s.shift) - 1 else ""))
ax.plot(r, -vn00(r), color=INK, lw=1.0, ls="--", label=r"$\ell$=0 diagonal element $(V_N)_{00}$")
vline(ax, g.R0, r"$R_0$", 2)
vline(ax, 14, r"$r_m$ = 14", 2)
ax.text(14.4, -55, f"at $r_m$ = 14 fm:\n$V_0 f_s$ = {nuc(14, s.shift.min()):.4f}–{tip(14):.3f} MeV\n"
        f"= {nuc(14, s.shift.min()) / V0:.0e}–{tip(14) / V0:.0e} of $V_0$", color=MUTED, fontsize=7.5, va="top")
ax.set_xlim(0, 20); ax.set_ylim(-165, 12)
ax.set_xlabel(r"$r$ (fm)"); ax.set_ylabel("MeV")
ax.set_title(r"(a) nuclear potential $-V_0 f_s(r)$, four orientations", loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

# (b) 各道有效势与 E = Q_α（线性坐标）
r = np.linspace(0.05, 50, 3000)
V = potential(s, p, r)
for i, l in enumerate(s.l):
    bx.plot(r, V[:, i, i] + s.c * l * (l + 1) / r ** 2 + s.ex[i], color=C[i], lw=1.3, label=rf"$\ell={l}$")
bx.axhline(Q, color=INK, lw=0.8)
bx.text(16, Q - 3, rf"$E=Q_\alpha$ = {Q:.3f} MeV", color=INK, fontsize=8, va="top")
vline(bx, 14, r"$r_m$", 28)
vline(bx, r_t[0], r"$r_{t,0}$ (outer turning point)", 28, ha="right")
bx.set_xlim(0, 50); bx.set_ylim(-125, 35)
bx.set_xlabel(r"$r$ (fm)")
bx.set_title(r"(b) effective potentials $V_{\ell\ell}+c\ell(\ell+1)/r^2+\varepsilon_\ell$", loc="left")
bx.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)

# (c) 各项大小（对数坐标）
r = np.linspace(5, 60, 2000)
cx.semilogy(r, tip(r), color=C[1], lw=1.8, label="nuclear, tip orientation (bounds every nuclear matrix element)")
cx.semilogy(r, nuc(r, s.shift.min()), color=C[1], lw=1.0, ls="--", label="nuclear, waist orientation")
cx.semilogy(r, coul0(r), color=C[0], lw=1.3, label="Coulomb monopole")
cx.semilogy(r, abs(quad(r)), color=C[2], lw=1.3, label=r"Coulomb quadrupole coupling $\ell$=0↔2")
cx.semilogy(r, abs(hexa(r)), color=C[3], lw=1.3, label=r"Coulomb hexadecapole coupling $\ell$=0↔4")
cx.semilogy(r, s.c * 42 / r ** 2, color=C[4], lw=1.3, label=r"centrifugal $\ell$=6")
for x in (14, 20, 25, 30):
    cx.plot(x, tip(x), "o", ms=5, color=C[1], mec=BG, mew=1.0, zorder=3)
    cx.annotate(f"{tip(x):.1e} MeV", (x, tip(x)), xytext=(7, 2), textcoords="offset points", color=INK, fontsize=7.5)
vline(cx, 14, r"$r_m$ = 14 (default)", 2e2)
vline(cx, r_t[0], r"$r_{t,0}$", 2e2)
cx.set_xlim(5, 60); cx.set_ylim(1e-16, 1e3)
cx.set_xlabel(r"$r$ (fm)"); cx.set_ylabel("|potential| (MeV)")
cx.set_title("(c) size of each term: the nuclear potential decays like exp(−r/0.57 fm), "
             r"the Coulomb couplings only like $r^{-\lambda-1}$", loc="left")
cx.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.05), frameon=True, facecolor=BG, edgecolor=BG, framealpha=1)

# (d) T½ 随 r_m：外区含 / 不含核势
cx_kw = dict(lw=1.3, marker="o", ms=4.5, mec=BG, mew=1.0)
dx.semilogy(RMS, abs(T_full / T_ref - 1), "-", color=C[0], label="exterior propagated with the full potential (this code)", **cx_kw)
dx.semilogy(RMS, abs(T_cut / T_ref - 1), "--", color=C[1],
            label=r"nuclear potential dropped beyond $r_m$ (standard R-matrix assumption)", **cx_kw)
vline(dx, 14, r"default $r_m$", 0.3)
dx.annotate(rf"$r_m$ = 22 fm: dropping $|V_N|\leq${tip(22):.0e} MeV changes $T_{{1/2}}$ by"
            f" {abs(T_cut[-1] / T_full[-1] - 1):.0e},\nimpossible physically: round-off in the tiny $u(r_m)$ (see a_convergence)",
            (22, abs(T_cut[-1] / T_ref - 1)), xytext=(15.6, 2e-2), color=MUTED, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
dx.set_xlim(9.5, 22.5); dx.set_ylim(1e-7, 1)
dx.set_xlabel(r"matching radius $r_m$ (fm)"); dx.set_ylabel(r"$|T_{1/2}/T_{1/2}^{\rm plateau}-1|$")
dx.set_title(r"(d) $T_{1/2}$ at $E=Q_\alpha$ vs matching radius ($N$ = 120·$r_m$/14)", loc="left")
dx.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.suptitle(rf"$^{{252}}$Cf → $^{{248}}$Cm + α: potentials (Ni–Ren geometry, $V_0$ = {V0} MeV, rank {RANK})",
             color=INK, fontsize=10.5)
fig.savefig(HERE / "potentials.png", dpi=150, facecolor=BG)
print("写出 docs/potentials.png、docs/potentials.json")
