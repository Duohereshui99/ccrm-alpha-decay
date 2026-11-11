"""耦合道 R 矩阵实能量本征值法：形变偶偶核 α 衰变的准束缚（窄共振）态。

输入  System（体系）+ Potential（待推断的 V0、R0、a）+ target（目标态：位次、参考本征矢或上次的 Result）[+ Numerics]
输出  Result：该态能量 Q（模型预测的 Q 值）、各道宽度 Γ_ℓ、内外区波函数、出射振幅、位次和诊断。

模型（Ni & Ren, PRC 81, 064318 (2010)）：母核 0⁺ → 子核 K=0 基带 I = 0,2,4,6，ℓ = I，
    [-c d²/dr² + cℓ(ℓ+1)/r² + ε_ℓ - E] u_ℓ + Σ_ℓ' V_ℓℓ'(r) u_ℓ' = 0,   c = ħ²/2μ,
    V = V_C - V0·f：f 为形变 Woods–Saxon（有限位移算符，对 β 全阶），V_C 为均匀带电球 + 线性 β 多极。
数值方法（solve 的执行顺序）：
    1. 内区 [0,a]：正则化 Lagrange–Legendre 网格 + Bloch 算符（Descouvemont, CPC 200 (2016) 199），边界
       u'(a) = B u(a)，B = sym Re[H'(a) H(a)⁻¹]（实能量窄共振近似，不是 Gamow 极点）。先取 B = 0 对角化选态
       （只求目标位次附近的本征对）。
    2. 外区：nch 组出射解 H(r) 在该态能量下从 Rout 向内传播到 a（含完整耦合势）。方程线性，固定步长的
       8 阶 RK 步（DOP853 的 Butcher 表）即乘一个 8×8 矩阵，所有步的矩阵向量化地一次算出。
    3. 以 B 修正内区：对 H(B) 在该能量处做反幂迭代，直到本征值与外区能量自洽（通常一遍）。
    4. 宽度：u(a) = H(a) A，Γ_ℓ = 2 c k_ℓ |A_ℓ|² / (1 + N_ext)；T½ = ħ ln2 / (P_α ΣΓ)。
共振态须逐体系判定（candidates 给出判定所需的量，见 README；它把 E_k(V0) = Q 化为广义本征问题一次求出各位次的 V0）。
来源：势、角耦合与网格公式改写自早期的 solver.py（原项目 archive/solver_v1/，其动能矩阵参考科研库 code/rmatrix-cc-python）；
      大 ρ 出射库仑波用 DLMF 33.11.1 渐近级数。单位：fm、MeV、s。
"""
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.integrate import DOP853
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.special import expit, loggamma

HBARC, HBAR = 197.3269804, 6.582119569e-22   # ħc (MeV fm)，ħ (MeV s)
E2, AMU = 1.4399764, 931.49410242            # e²/4πε0 (MeV fm)，原子质量单位 (MeV)
RK_A, RK_B, RK_C = DOP853.A, DOP853.B, DOP853.C   # 8 阶 Dormand–Prince 的 Butcher 表（12 级）


def angular_matrix(l, lam):
    """K=0、J=0 通道基中 Y_λ0 的矩阵 A_λ[ℓ,ℓ'] = sqrt((2ℓ+1)(2ℓ'+1)(2λ+1)/4π)/2 ∫P_ℓ P_λ P_ℓ' dx。"""
    x, w = leggauss(40)                       # 被积多项式阶数 ≤ 2·ℓmax + λ，40 点 Gauss 积分精确
    p = np.array([legval(x, [0] * li + [1]) for li in l])
    s = np.sqrt(2 * np.asarray(l) + 1)
    return (p * w * legval(x, [0] * lam + [1])) @ p.T * np.outer(s, s) / 2 * np.sqrt((2 * lam + 1) / (4 * np.pi))


@dataclass
class System:
    """体系信息（推断中固定）。通道为子核 K=0 基带 I = 0,2,4,…；母核 0⁺，故 ℓ = I。"""
    Z: int                 # 母核电荷数
    A: int                 # 母核质量数
    beta: tuple            # 子核形变 (β2, β4[, β6, …])，对应多极 λ = 2, 4, 6, …
    ex: tuple              # 子核 0⁺, 2⁺, 4⁺, 6⁺, … 激发能 (MeV)，首项为 0；长度即通道数
    G: int = 22            # Wildermuth 量子数 G = 2n + ℓ（N > 126 的子核取 22）
    Rd: float = None       # 形变位移所乘半径 (fm)；缺省为 Ni–Ren 的 (1 + 0.39 I) A_d^(1/3)
    Rc: float = None       # 库仑均匀球半径 (fm)；缺省为 1.20 (4^(1/3) + A_d^(1/3))

    def __post_init__(self):
        zd, ad = self.Z - 2, self.A - 4
        self.I = (ad - 2 * zd) / ad                                  # 子核同位旋不对称度
        self.Rd = (1 + 0.39 * self.I) * ad ** (1 / 3) if self.Rd is None else self.Rd
        self.Rc = 1.20 * (4 ** (1 / 3) + ad ** (1 / 3)) if self.Rc is None else self.Rc
        self.ex = np.asarray(self.ex, float)
        self.l = 2 * np.arange(len(self.ex))                         # 通道角动量 ℓ
        self.lams = 2 * np.arange(1, len(self.beta) + 1)             # 形变多极 λ
        self.c = HBARC ** 2 / (2 * AMU * 4 * ad / (4 + ad))          # ħ²/2μ (MeV fm²)，整数质量数
        self.zz = 2 * zd * E2                                        # Z_α Z_d e² (MeV fm)
        self.ang = {lam: angular_matrix(self.l, lam) for lam in self.lams}
        # 形变位移算符 O = R_d Σ β_λ A_λ = U diag(λ_s) Uᵀ（Ni–Ren 式 (7)），λ_s 为各取向的半径位移
        self.shift, self.U = eigh(self.Rd * sum(b * self.ang[lam] for lam, b in zip(self.lams, self.beta)))


@dataclass
class Potential:
    """待推断的核势参数：V_N = -V0 / {1 + exp[(r - R0 - R_d Σ β_λ Y_λ0)/a]}。"""
    V0: float              # 势深 (MeV，取正值)
    R0: float              # 半径 (fm)
    a: float               # 弥散 (fm)

    @classmethod
    def ni_ren(cls, s, V0):
        """Ni–Ren 系统学几何参数：R0 = R_d + 1.17 fm，a = 0.50 + 0.33 I fm。"""
        return cls(V0, s.Rd + 1.17, 0.50 + 0.33 * s.I)


@dataclass
class Numerics:
    """数值参数；物理结果不应随它们改变（收敛检验见 check_ccrm.py）。"""
    rm: float = 14.0       # 匹配半径 a (fm)：须在 Rc 之外、势垒之下；外区传播含核势，不必在核势力程之外
    N: int = 120           # 每道 Lagrange 网格点数；Q = Q_exp 处 T½ 与 N = 200 相差约 2e-5
    Rout: float = 360.0    # 外区出射波起点 (fm)；其外的库仑耦合被忽略，弱分支（6⁺）误差约 2e-3，减小时增大
    dr: float = 0.5        # 外区传播步长 (fm)；8 阶方法，传播误差 ∝ dr⁸，默认值下对 Γ 约 1e-7
    kmax: int = 60         # 按参考向量选态时求升序最低 kmax+1 个本征对
    window: int = 8        # 以上次的 Result 为 target 时，只求其位次 ± window 的本征对（不足时自动退回全范围）
    etol: float = 1e-7     # 外区能量自洽容差 (MeV)；对应宽度相对误差约 10·etol/MeV


@dataclass
class Result:
    """solve 的输出；宽度不含预形成因子 P_α。"""
    Q: float               # 所选态能量，即模型预测的 Q 值 (MeV，相对 α + 子核基态)
    gamma: np.ndarray      # 各道宽度 Γ_ℓ (MeV)，顺序 ℓ = 0, 2, 4, 6
    rank: int              # 所选本征值在升序谱中的位次（从 0 起）
    r: np.ndarray          # 内区网格 r_n (fm)
    u: np.ndarray          # 内区波函数 u_ℓ(r_n) (fm^-1/2)，形状 (nch, N)；∫₀^a Σ u² dr = 1
    amp: np.ndarray        # 出射振幅 A_ℓ：r → ∞ 时 u_ℓ(r) → A_ℓ H⁺_ℓ(k_ℓ r)
    r_ext: np.ndarray = field(repr=False)   # 外区网格 (fm)，a → Rout 升序，等步长 dr
    u_ext: np.ndarray = field(repr=False)   # 外区波函数 u_ℓ(r_ext) = H(r_ext) A，复数，形状 (nch, len(r_ext))
    c: np.ndarray = field(repr=False)       # 本征矢系数（长 nch·N），可作同一网格下次调用的 target
    info: dict = None      # 选态与数值诊断，见 solve

    def half_life(self, P=0.39):
        """α 衰变分半衰期 T½ = ħ ln2 / (P_α ΣΓ) (s)。"""
        return HBAR * np.log(2) / (P * self.gamma.sum())


def coulomb_h(l, eta, rho):
    """出射库仑波 H⁺_ℓ = G + iF 及 dH⁺/dρ（DLMF 33.11.1 渐近级数，可对各道向量化）。

    要求 ρ 在外转折点 η + sqrt(η² + ℓ(ℓ+1)) 之外较远，此时级数在双精度内收敛。
    """
    a, b = 1 + l + 1j * eta, -l + 1j * eta
    term, s, ds = np.ones_like(a), np.ones_like(a), np.zeros_like(a)
    for k in range(300):
        term = term * (a + k) * (b + k) / ((k + 1) * 2j * rho)       # (a)_k (b)_k / (k! (2iρ)^k)
        s, ds = s + term, ds - (k + 1) * term / rho
        if np.all(abs(term) < 1e-17 * abs(s)):
            break
    e = np.exp(1j * (rho - eta * np.log(2 * rho) - l * np.pi / 2 + loggamma(1 + l + 1j * eta).imag))
    return e * s, e * (1j * (1 - eta / rho) * s + ds)


def potential(s, p, r, deformed=True):
    """局域势矩阵 V(r) = V_C(r) - V0·f(r) (MeV)，形状 (len(r), nch, nch)。

    写成 Σ_j g_j(r) M_j，一次矩阵乘法求出：核势在位移本征基中为 f_s = 1/[1 + exp((r - R0 - λ_s)/a)]，
    M_s = U_s U_sᵀ（即 U diag(f) Uᵀ）；库仑势为均匀带电球单极项（M = 1）加线性 β_λ 多极项（M = A_λ）。
    deformed=False 给出 β=0 的球形极限。
    """
    r, nch = np.atleast_1d(r)[:, None], len(s.l)
    shift, U, beta = (s.shift, s.U, s.beta) if deformed else (0 * s.shift, np.eye(nch), 0 * np.asarray(s.beta))
    inside = r < s.Rc
    g = [-p.V0 * expit(-(r - p.R0 - shift) / p.a),                                 # 核势各取向；expit 防溢出
         np.where(inside, s.zz / (2 * s.Rc) * (3 - (r / s.Rc) ** 2), s.zz / r)]     # 库仑单极
    M = [np.einsum("is,js->sij", U, U), np.eye(nch)[None]]
    for lam, b in zip(s.lams, beta):                                                # 库仑 β_λ 多极
        g.append(3 * s.zz / (2 * lam + 1) * b * np.where(inside, r ** lam / s.Rc ** (lam + 1), s.Rc ** lam / r ** (lam + 1)))
        M.append(s.ang[lam][None])
    return (np.hstack(g) @ np.concatenate(M).reshape(-1, nch * nch)).reshape(-1, nch, nch)


@lru_cache(maxsize=8)
def mesh(N, rm):
    """[0, a] 上的正则化 Lagrange–Legendre 网格（a = rm）。

    返回 r_n、[0,1] 上的权重 w_n、基函数边界值 φ_n(a)，以及 -d²/dr² + Bloch(B=0) 的矩阵 T (fm⁻²)。
    """
    xi, wi = leggauss(N)
    x, w = (xi + 1) / 2, wi / 2
    phia = (-1.0) ** (N + np.arange(1, N + 1)) / np.sqrt(rm * x * (1 - x))
    X, Y = x[:, None], x[None, :]
    D = X - Y
    np.fill_diagonal(D, 1.0)                                         # 对角元另算，这里只防除零
    T = np.outer(phia, phia) / rm * (N * N + N + 1 + (X + Y - 2 * X * Y) / D ** 2 - 1 / (1 - X) - 1 / (1 - Y))
    np.fill_diagonal(T, phia ** 2 / rm * ((4 * N * N + 4 * N + 3) * x * (1 - x) - 6 * x + 1) / (3 * x * (1 - x)))
    return rm * x, w, phia, T


def hamiltonian(s, p, num, deformed=True):
    """B=0 的内区矩阵 c·T + V(r_n) + cℓ(ℓ+1)/r_n² + ε_ℓ，下标为 (通道, 节点)，维数 nch·N。"""
    r, _, _, T = mesh(num.N, num.rm)
    nch, N = len(s.l), num.N
    v = potential(s, p, r, deformed)
    v[:, range(nch), range(nch)] += s.ex + s.c * s.l * (s.l + 1) / r[:, None] ** 2
    h = np.kron(np.eye(nch), s.c * T)
    h.reshape(nch, N, nch, N)[:, range(N), :, range(N)] += v        # Gauss 近似：局域势只连接同一节点
    return h


def reference(s, p, num, nch=None):
    """球形极限（β=0）前 nch 道（缺省全部）第 n_ℓ = (G-ℓ)/2 个本征矢，即 G 壳成员 |ℓ, n_ℓ⟩（Sturm：恰有 n_ℓ 个节点），形状 (nch, N)。"""
    hs, N = hamiltonian(s, p, num, deformed=False), num.N               # 球形时各道互不耦合，只对角化对角块
    ref = [eigh(hs[i * N:(i + 1) * N, i * N:(i + 1) * N], subset_by_index=((s.G - l) // 2,) * 2)[1][:, 0]
           for i, l in enumerate(s.l[:nch])]
    return np.array([v * np.sign(v[-1]) for v in ref])                # 符号约定：近 a 处尾部为正


def nodes(u, thr=1e-4):
    """各道 u_ℓ(r_n) 的内部节点数；忽略幅度小于该道最大值 thr 倍的点（原点附近和尾部）。"""
    kept = [row[abs(row) > thr * abs(row).max()] for row in u]
    return [int(np.sum(v[:-1] * v[1:] < 0)) for v in kept]


def exterior(s, p, num, E):
    """在能量 E 下从 Rout 向内传播 nch 组独立出射解到 a（含完整耦合势），出射边界的外区解都是 u(r) = H(r) A。

    第 j 列在 Rout 处为 δ_ij H⁺_ℓj(η_j, k_j r)。Y = [H; H'] 满足线性方程 Y' = M Y，M = [[0, I], [K, 0]]，
    K = (V + ε + cℓ(ℓ+1)/r² - E)/c，所以每个 RK 步的各级状态 S_i Y_n、导数 G_i Y_n 和整步 T_n Y_n 都是矩阵乘法；
    各级状态同时给出 ∫|u|² dr 的同阶求积 Σ_n Σ_i |h| b_i |u(r_n + c_i h)|²。返回量按传播方向 Rout → a 排列。
    """
    q = E - s.ex                                                     # 各道相对运动能 (MeV)
    if np.any(q <= 0):
        raise ValueError(f"E = {E:.4f} MeV 低于某道阈值；本程序要求各道开放")
    k = np.sqrt(q / s.c)
    eta = s.zz / (2 * s.c * k)
    if np.any(k * num.Rout < 2 * (eta + np.sqrt(eta ** 2 + s.l * (s.l + 1)))):
        raise ValueError("Rout 离外转折点太近，渐近库仑级数不可靠")
    if num.rm < s.Rc:
        raise ValueError("匹配半径须大于库仑半径 Rc")
    m, n = len(s.l), int(np.ceil((num.Rout - num.rm) / num.dr))
    r = np.linspace(num.Rout, num.rm, n + 1)
    h = r[1] - r[0]                                                  # 向内传播，h < 0
    x = (r[:-1, None] + RK_C * h).ravel()                            # 各步各级的求值点
    K = potential(s, p, x)
    K[:, range(m), range(m)] += s.ex - E + s.c * s.l * (s.l + 1) / x[:, None] ** 2
    K = (K / s.c).reshape(n, RK_C.size, m, m)
    I, G, S = np.eye(2 * m), np.empty((RK_C.size, n, 2 * m, 2 * m)), np.empty((RK_C.size, n, m, 2 * m))
    for i in range(RK_C.size):
        Si = I + h * np.tensordot(RK_A[i, :i], G[:i], 1)            # 第 i 级的状态 = Si Y_n
        G[i] = np.concatenate((Si[:, m:], K[:, i] @ Si[:, :m]), axis=1)   # 第 i 级的导数 = M Si Y_n
        S[i] = Si[:, :m]
    T = I + h * np.tensordot(RK_B, G, 1)                             # 每步的传播矩阵
    hp, dhp = coulomb_h(s.l, eta, k * num.Rout)
    Y = np.empty((n + 1, 2 * m, m), complex)
    Y[0] = np.vstack((np.diag(hp), np.diag(k * dhp)))                # dH/dr = k dH/dρ
    for j in range(n):
        Y[j + 1] = T[j] @ Y[j]
    L = np.linalg.solve(Y[-1, :m].T, Y[-1, m:].T).T.real             # Re[H' H⁻¹]（在 a 处）
    return SimpleNamespace(r=r, Y=Y, S=S, wq=abs(h) * RK_B, k=k, eta=eta, Ha=Y[-1, :m], B=(L + L.T) / 2)


def solve(s, p, target, num=Numerics(), diag=True):
    """正向计算：体系 s、势参数 p 下目标态的观测量（Result）。

    target 为整数时取升序第 target 个本征态；为参考系数向量（须同一网格）时，在最低 kmax+1 个本征态中取与之
    重叠最大的态；为上次的 Result 时同样按重叠选态，但只求其位次 ± window 的本征对。窗外各态的重叠平方和
    不超过 1 - Σ_窗内 重叠²，若它可能超过窗内次大重叠的平方、或所选态落在窗口边缘，就退回全范围，所以选态
    结果与全范围相同。先用 B=0 对角化选态，在该态能量处传播外区得 B、H(a)，再对 H(B) 做反幂迭代，直到其
    本征值与外区所用能量之差 < etol；最后由边界值 u(a) 求出射振幅和各道宽度。diag=False 时不算 G 壳投影
    （info 中 w_l0、w_G 为 None），推断中可省时。
    """
    nch, N = len(s.l), num.N
    r, w, phia, _ = mesh(N, num.rm)
    h0 = hamiltonian(s, p, num)
    ref = target.c if isinstance(target, Result) else target
    track = np.ndim(ref) > 0
    k0 = target.rank if isinstance(target, Result) else None if track else int(target)   # 已知或上次的位次
    d = num.window if track else 1                                   # 整数选态只需 k0 及其上下相邻（间距诊断）
    lo = 0 if k0 is None else max(k0 - d, 0)
    e, vec = eigh(h0, subset_by_index=(lo, num.kmax if k0 is None else k0 + d), driver="evr")  # 第 1 遍：B = 0，选态
    if track:
        ov = abs(vec.T @ ref)
        j = int(np.argmax(ov))
        if k0 is not None and (np.sort(ov)[-2] ** 2 < 1 - ov @ ov or j == len(e) - 1 or (j == 0 and lo > 0)):
            lo, (e, vec) = 0, eigh(h0, subset_by_index=(0, num.kmax), driver="evr")   # 窗口不足：退回全范围
            ov = abs(vec.T @ ref)
            j = int(np.argmax(ov))
    else:
        j = k0 - lo
    k, E, x = lo + j, e[j], vec[:, j]
    for passes in range(2, 6):                                       # 之后各遍：外区 → B → 反幂迭代
        ext = exterior(s, p, num, E)
        hB = h0 - s.c * np.kron(ext.B, np.outer(phia, phia))         # Robin 边界项 -c φ(a)ᵀ B φ(a)
        lu = lu_factor(hB - E * np.eye(nch * N), check_finite=False)
        for _ in range(2):                                           # E 与目标本征值之差 ~1e-8 MeV，两次即到机器精度
            x = lu_solve(lu, x, check_finite=False)
            x /= np.linalg.norm(x)
        E, E_ext = x @ hB @ x, E
        if abs(E - E_ext) < num.etol:
            break
    else:
        raise RuntimeError("外区能量自洽迭代未收敛")
    want = nch if diag else 0 if track else 1                        # 不算诊断时，整数选态的符号约定只需 ℓ=0 道
    proj = np.einsum("in,in->i", reference(s, p, num, want), x.reshape(nch, N)[:want]) if want else None  # ⟨ℓ, n_ℓ|ψ⟩
    sign = np.sign(x @ ref) if track else np.sign(proj[0])
    c = x * (sign or 1.0)                                            # 符号约定：与参考态（或 |ℓ=0, n=G/2⟩）重叠为正
    u, ua = c.reshape(nch, N) / np.sqrt(num.rm * w), c.reshape(nch, N) @ phia   # u_ℓ(r_n)，u_ℓ(a)
    amp = np.linalg.solve(ext.Ha, ua)                                # u(a) = H(a) A
    y = ext.Y @ amp                                                  # 外区网格上的 [u; u']（Rout → a）
    uq = (ext.S @ y[:-1, :, None])[..., 0]                           # 各求积点上的 u_ℓ
    p_ext = np.einsum("i,inl->l", ext.wq, abs(uq) ** 2)              # 各道外区模方积分 ∫_a^Rout |u_ℓ|² dr
    gamma = 2 * s.c * ext.k * abs(amp) ** 2 / (1 + p_ext.sum())      # 1 + N_ext：全空间归一
    pad = np.r_[-np.inf, e, np.inf]
    info = dict(
        overlap=abs(c @ ref) if track else None,   # |⟨参考态|ψ⟩|；明显小于 1 时跟踪到的态已与邻近态混合
        overlap_next=np.sort(ov)[-2] if track else None,   # 次大重叠（B = 0 的谱）
        nodes=nodes(u),                            # 各道内部节点数；Wildermuth 条件要求 n_ℓ = (G-ℓ)/2
        w_l0=proj[0] ** 2 if diag else None,       # |⟨ℓ=0, n=G/2|ψ⟩|²
        w_G=(proj ** 2).sum() if diag else None,   # G 壳总权重 Σ_ℓ |⟨ℓ, n_ℓ|ψ⟩|²
        gaps=(e[j] - pad[j], pad[j + 2] - e[j]),   # 与上下相邻本征值的间距 (MeV，B = 0 的谱)
        tail=abs(ua).max() / abs(u).max(),         # |u(a)|/max|u|；过小时尾部被本征求解舍入误差主导
        E_ext=E_ext, passes=passes, k=ext.k, eta=ext.eta,
        n_ext=p_ext.sum(), p_ext=p_ext)            # 外区范数 N_ext = Σ_ℓ ∫_a^Rout |u_ℓ|² dr 及各道分量
    return Result(E, gamma, k, r, u, amp, ext.r[::-1], y[::-1, :nch].T, c, info)


def candidates(s, R0, a, Q, ranks=range(30, 50), num=Numerics(), lo=80.0, hi=240.0):
    """判定共振态用：对每个位次 k 求 V0 使 E_k = Q（只保留 lo ≤ V0 ≤ hi），返回 V0、各道节点数和 Wildermuth 权重。

    H(V0) = h0 - V0·F，F 为核势形状因子矩阵（各节点 U diag(f_s) Uᵀ，正定），所以 E_k(V0) = Q 就是广义本征问题
    (h0 - Q) x = V0 F x：E_k 随 V0 严格递减，第 k 小的广义本征值正是位次 k 的根，一次对角化得到所有位次。
    数值上解等价的 F x = μ W x（μ = 1/V0，W = h0 - Q，降序第 k+1 个 μ 对应位次 k）：V0 = 0 时没有能级低于 Q，
    W 正定且最小本征值不小（252Cf、rm = 14 fm 时 15.7 MeV），只对 W 做 Cholesky 分解，1/μ 已准确到 ~1e-8 MeV。
    反过来对 F 分解则病态：F 在网格外端小到 1e-5 且随匹配半径 rm 指数变小，rm ≳ 20 fm 时结果出错。
    F 直接由形状因子构造（H(0) - H(1) 相减会带来 ~1e-8 的误差）。最后在原始矩阵上取 Rayleigh 商
    V0 = xᵀWx / xᵀFx（保险），与逐位次 Newton 求根相差 ≲ 2e-8 MeV。要求 rm 在势垒内（W 正定），已核对到 32 fm。
    只用 B=0 本征值，这要求 u(rm) 很小：默认 rm = 14 fm 时与 B≠0 的能量差约 2e-8 MeV，rm ≲ 12 fm 时不再可忽略
    （rm = 10 fm 时 V0 偏 0.04 MeV）。
    """
    h0, N, nch = hamiltonian(s, Potential(0.0, R0, a), num), num.N, len(s.l)
    f = expit(-(mesh(N, num.rm)[0][:, None] - R0 - s.shift) / a)     # 各节点、各取向的形状因子 f_s(r_n)
    F = np.zeros_like(h0)                                            # F = -∂H/∂V0，只连接同一节点
    F.reshape(nch, N, nch, N)[:, range(N), :, range(N)] = np.einsum("is,ns,js->nij", s.U, f, s.U)
    W, n = h0 - Q * np.eye(len(h0)), len(h0)                        # W 正定：V0 = 0 时没有能级低于 Q
    ks = range(min(ranks), max(ranks) + 1)
    X = eigh(F, W, subset_by_index=(n - 1 - ks[-1], n - 1 - ks[0]), driver="gvx")[1]  # F x = (1/V0) W x，μ 升序
    X = np.ascontiguousarray(X[:, ::-1])                             # 改为位次升序（连续存储，后面的矩阵乘法才走 BLAS）
    X /= np.linalg.norm(X, axis=0)
    V = np.einsum("ij,ij->j", X, W @ X) / np.einsum("ij,ij->j", X, F @ X)   # Rayleigh 商修正的 V0
    w, rows = mesh(num.N, num.rm)[1], []
    for k, v, x in zip(ks, V, X.T):
        if k not in ranks or not lo <= v <= hi:
            continue
        x = x.reshape(len(s.l), num.N)
        proj = np.einsum("in,in->i", reference(s, Potential(v, R0, a), num), x)
        rows.append(dict(rank=k, V0=v, nodes=nodes(x / np.sqrt(num.rm * w)), w_l0=proj[0] ** 2, w_G=(proj ** 2).sum()))
    return rows
