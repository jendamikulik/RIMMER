import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("singularity_numerics_out")
OUT.mkdir(exist_ok=True)

def delta(u, a=1.3, M=0.6):
    b = M * np.sin(2.7*u) / (1 + 0.2*u*u)
    return a*u + b

def P4_numeric(tau, L, a=1.3, M=0.6, n=20000):
    u = np.linspace(tau, tau + L, n)
    y = np.abs(delta(u, a=a, M=M))**4
    return np.trapezoid(y, u)

def P4_bound(tau, L, a=1.3):
    return (abs(a)**4 / 16.0) * (
        tau**4 * L + 2*tau**3 * L**2 + 2*tau**2 * L**3 + tau * L**4 + (1/5) * L**5
    )

def explosive_lower_bound(tau, L, a=1.3, c1=0.8, c2=2.0):
    return c1 * P4_bound(tau, L, a=a) / L - c2

def winding_number(m=1, eps=0.25, n=4000):
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    z = eps * np.exp(1j * theta)
    f = z**m
    ang = np.unwrap(np.angle(f))
    return (ang[-1] - ang[0]) / (2*np.pi)

def run_demo():
    a = 1.3
    M = 0.6
    tau = 2*M/abs(a) + 0.8

    Ls = np.linspace(1, 12, 120)
    p4_vals = np.array([P4_numeric(tau, L, a=a, M=M) for L in Ls])
    bound_vals = np.array([P4_bound(tau, L, a=a) for L in Ls])

    plt.figure(figsize=(8,5))
    plt.plot(Ls, p4_vals, label=r"$P_4(\tau,L)$ numeric")
    plt.plot(Ls, bound_vals, label=r"quartic lower bound")
    plt.xlabel("L")
    plt.ylabel("value")
    plt.title("Quartic window: numeric value vs proven lower bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "quartic_bound_vs_numeric.png", dpi=180)
    plt.close()

    Ls2 = np.linspace(1, 20, 200)
    ex_vals = np.array([explosive_lower_bound(tau, L, a=a) for L in Ls2])

    plt.figure(figsize=(8,5))
    plt.plot(Ls2, ex_vals)
    plt.xlabel("L")
    plt.ylabel(r"lower bound for $(Q_{\rm bank}-E_0L)/L$")
    plt.title("Explosive excess lower bound grows like $L^4$")
    plt.tight_layout()
    plt.savefig(OUT / "explosive_excess_lower_bound.png", dpi=180)
    plt.close()

    report = []
    report.append("Singularity numerics demo")
    report.append("========================")
    report.append(f"a = {a}, M = {M}")
    report.append(f"tau = {tau:.6f} > 2M/|a| = {2*M/abs(a):.6f}")
    report.append("")
    report.append("Quartic window checks")
    report.append("---------------------")
    for L in [1, 2, 5, 10]:
        p4 = P4_numeric(tau, L, a=a, M=M)
        bd = P4_bound(tau, L, a=a)
        report.append(f"L={L:>4}: P4_numeric = {p4:.6f}, bound = {bd:.6f}, ratio P4/bound = {p4/bd:.4f}")
    report.append("")
    report.append("Explosive excess lower bound")
    report.append("----------------------------")
    for L in [2, 5, 10, 15, 20]:
        ex = explosive_lower_bound(tau, L, a=a)
        report.append(f"L={L:>4}: lower bound = {ex:.6f}")
    report.append("")
    report.append("Winding numbers around z=0 for f(z)=z^m")
    report.append("----------------------------------------")
    for m in [1, 2, 4]:
        w = winding_number(m=m)
        report.append(f"m={m}: numeric winding = {w:.6f}, expected = {m}")
    report.append("")
    report.append("Quotient extension examples")
    report.append("---------------------------")
    report.append("z^2 / z extends holomorphically through 0 as z.")
    report.append("z / z^2 does not extend holomorphically through 0 (pole 1/z).")
    (OUT / "report.txt").write_text("\\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved plots and report in: {OUT.resolve()}")

if __name__ == "__main__":
    run_demo()
