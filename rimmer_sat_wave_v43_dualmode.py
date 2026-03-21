
"""
rimmer_sat_wave_v43_dualmode.py

V4.3 dual mode:
- exact mode for n <= 12 (full 2^n basis)
- projected mode for larger DIMACS CNF (sampled assignment subspace)
- DIMACS loading
- SAT vs UNSAT frustration graph
- collapse / ABS / monodromy
- Trotter evolution in both modes

Controls:
  space : pause/resume
  c     : standard collapse
  a     : ABS closure
  m     : monodromy flip
  i     : cycle built-in instance
  l     : load DIMACS CNF
  r     : reset current instance
  s     : resample projected subspace (projected mode only)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


# ----------------------------
# Parsing / utilities
# ----------------------------
def int_to_bits(x: int, n: int):
    return tuple((x >> i) & 1 for i in range(n))


def bits_to_str(bits):
    return "".join(str(int(b)) for b in bits)


def eval_lit(bits, lit):
    var = abs(lit) - 1
    val = bits[var]
    return val == 1 if lit > 0 else val == 0


def clause_satisfied(bits, clause):
    return any(eval_lit(bits, lit) for lit in clause)


def violations(bits, clauses):
    return sum(0 if clause_satisfied(bits, c) else 1 for c in clauses)


def parse_dimacs_cnf(path):
    n_vars = None
    clauses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        current = []
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    n_vars = int(parts[2])
                continue
            nums = [int(x) for x in line.split()]
            for lit in nums:
                if lit == 0:
                    if current:
                        clauses.append(tuple(current))
                        current = []
                else:
                    current.append(lit)
        if current:
            clauses.append(tuple(current))
    if n_vars is None:
        max_var = 0
        for c in clauses:
            for lit in c:
                max_var = max(max_var, abs(lit))
        n_vars = max_var
    if n_vars <= 0:
        raise ValueError("DIMACS parse failed: no variables found.")
    return n_vars, clauses


INSTANCES = [
    {"name": "SAT: (x1 v x2) ^ (!x1 v x2)", "n": 2, "clauses": [(1, 2), (-1, 2)]},
    {"name": "UNSAT: (x1) ^ (!x1)", "n": 1, "clauses": [(1,), (-1,)]},
    {"name": "SAT: 3-var mixed", "n": 3, "clauses": [(1, 2, -3), (-1, 2), (1, -2, 3)]},
    {"name": "UNSAT: 2-var contradiction family", "n": 2, "clauses": [(1, 2), (1, -2), (-1, 2), (-1, -2)]},
    {"name": "SAT: 4-var small", "n": 4, "clauses": [(1, -2, 3), (2, 4), (-1, -3, 4), (1, 2, -4)]},
]


# ----------------------------
# Exact engine
# ----------------------------
class ExactSATWaveEngine:
    mode_name = "exact"

    def __init__(self, n_vars, clauses, dt=0.05, lam=2.2, mix=1.0):
        self.n = n_vars
        self.N = 2 ** n_vars
        self.clauses = [tuple(c) for c in clauses]
        self.dt = dt
        self.lam = lam
        self.mix = mix

        self.basis = [int_to_bits(i, self.n) for i in range(self.N)]
        self.labels = [bits_to_str(b) for b in self.basis]
        self.H_clause_diag = np.array([violations(bits, self.clauses) for bits in self.basis], dtype=float)
        self.H_mix = self._build_mix_matrix()
        self._build_trotter_operators()

        self.psi = None
        self.last_event = "none"
        self.last_index = None
        self.monodromy_count = 0
        self.t = 0.0
        self.reset()

    def _build_mix_matrix(self):
        Hm = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for b in range(self.n):
                Hm[i, i ^ (1 << b)] = 1.0
        return Hm

    def _build_trotter_operators(self):
        self.U_clause_half = np.exp(-1j * self.dt * self.lam * self.H_clause_diag / 2.0)
        vals, vecs = np.linalg.eigh(self.H_mix)
        self.U_mix = vecs @ np.diag(np.exp(+1j * self.dt * self.mix * vals)) @ vecs.conj().T

    def reset(self):
        self.psi = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        self.last_event = "reset"
        self.last_index = None
        self.monodromy_count = 0
        self.t = 0.0

    def normalize(self):
        self.psi /= np.linalg.norm(self.psi)

    def step(self, steps=1):
        for _ in range(steps):
            self.psi *= self.U_clause_half
            self.psi = self.U_mix @ self.psi
            self.psi *= self.U_clause_half
            self.normalize()
            self.t += self.dt

    def probs(self):
        return np.abs(self.psi) ** 2

    def sat_mask(self):
        return self.H_clause_diag == 0

    def unsat_mask(self):
        return ~self.sat_mask()

    def sat_mass(self):
        p = self.probs()
        return float(np.sum(p[self.sat_mask()]))

    def frustration_mass(self):
        p = self.probs()
        return float(np.sum(p[self.unsat_mask()]))

    def expected_violations(self):
        return float(np.sum(self.probs() * self.H_clause_diag))

    def overlap_ground(self):
        H = np.diag(self.lam * self.H_clause_diag) - self.mix * self.H_mix
        vals, vecs = np.linalg.eigh(H)
        g = vecs[:, np.argmin(vals)]
        return float(np.abs(np.vdot(g, self.psi)) ** 2)

    def collapse(self, idx=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        p = self.probs()
        if idx is None:
            idx = int(rng.choice(self.N, p=p / np.sum(p)))
        self.psi[:] = 0.0
        self.psi[idx] = 1.0
        self.last_event = "collapse"
        self.last_index = idx

    def abs_closure(self, idx=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        p = self.probs()
        if idx is None:
            idx = int(rng.choice(self.N, p=p / np.sum(p)))
        target_v = int(self.H_clause_diag[idx])
        env = np.sqrt(p)
        sector = np.array([1.0 if int(v) == target_v else 1.0 / (1.0 + abs(v - target_v)) for v in self.H_clause_diag])
        self.psi = (env * sector).astype(complex)
        self.normalize()
        self.last_event = "ABS"
        self.last_index = idx

    def monodromy_flip(self):
        phase = np.ones(self.N, dtype=complex)
        for i, bits in enumerate(self.basis):
            weight = sum(bits)
            branch = -1.0 if bits[0] == 1 else 1.0
            gray = i ^ (i >> 1)
            theta = np.pi * (gray / max(1, self.N - 1))
            phase[i] = branch * np.exp(1j * (theta + np.pi * (weight % 2) / 2.0))
        self.psi *= phase
        self.normalize()
        self.last_event = "monodromy"
        self.monodromy_count += 1

    def winding_proxy(self):
        ph = np.unwrap(np.angle(self.psi + 1e-15))
        return float((ph[-1] - ph[0]) / (2.0 * np.pi))

    def top_k_view(self, k=32):
        p = self.probs()
        idx = np.argsort(-p)[:min(k, self.N)]
        return idx, p[idx], np.angle(self.psi[idx])


# ----------------------------
# Projected engine
# ----------------------------
class ProjectedSATWaveEngine:
    mode_name = "projected"

    def __init__(self, n_vars, clauses, k_samples=256, dt=0.05, lam=2.2, mix=1.0, seed=0):
        self.n = n_vars
        self.N = 2 ** n_vars
        self.clauses = [tuple(c) for c in clauses]
        self.k_samples = int(k_samples)
        self.dt = dt
        self.lam = lam
        self.mix = mix
        self.rng = np.random.default_rng(seed)

        self.sample_bits = None
        self.labels = None
        self.H_clause_diag = None
        self.H_mix = None
        self.U_clause_half = None
        self.U_mix = None

        self.psi = None
        self.last_event = "none"
        self.last_index = None
        self.monodromy_count = 0
        self.t = 0.0
        self.resample()

    def random_bits(self):
        return tuple(self.rng.integers(0, 2, size=self.n).tolist())

    def hamming(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    def _rebuild_subspace(self):
        self.labels = [bits_to_str(b) for b in self.sample_bits]
        self.H_clause_diag = np.array([violations(bits, self.clauses) for bits in self.sample_bits], dtype=float)

        K = len(self.sample_bits)
        Hm = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(i + 1, K):
                d = self.hamming(self.sample_bits[i], self.sample_bits[j])
                if d == 1:
                    Hm[i, j] = Hm[j, i] = 1.0
                elif d == 2:
                    Hm[i, j] = Hm[j, i] = 0.25
        self.H_mix = Hm

        self.U_clause_half = np.exp(-1j * self.dt * self.lam * self.H_clause_diag / 2.0)
        vals, vecs = np.linalg.eigh(self.H_mix)
        self.U_mix = vecs @ np.diag(np.exp(+1j * self.dt * self.mix * vals)) @ vecs.conj().T

    def resample(self):
        # keep unique sample set, bias toward low-violation states by local greedy flips
        pool = set()
        while len(pool) < self.k_samples:
            pool.add(self.random_bits())
        self.sample_bits = list(pool)

        # quick local improvement pass
        improved = []
        for bits in self.sample_bits:
            best = bits
            best_v = violations(bits, self.clauses)
            for b in range(self.n):
                cand = list(bits)
                cand[b] ^= 1
                cand = tuple(cand)
                v = violations(cand, self.clauses)
                if v < best_v:
                    best, best_v = cand, v
            improved.append(best)

        self.sample_bits = list(dict.fromkeys(improved))
        while len(self.sample_bits) < self.k_samples:
            self.sample_bits.append(self.random_bits())
        self.sample_bits = self.sample_bits[:self.k_samples]

        self._rebuild_subspace()
        K = len(self.sample_bits)
        self.psi = np.ones(K, dtype=complex) / np.sqrt(K)
        self.last_event = "resample"
        self.last_index = None
        self.monodromy_count = 0
        self.t = 0.0

    def reset(self):
        K = len(self.sample_bits)
        self.psi = np.ones(K, dtype=complex) / np.sqrt(K)
        self.last_event = "reset"
        self.last_index = None
        self.monodromy_count = 0
        self.t = 0.0

    def normalize(self):
        self.psi /= np.linalg.norm(self.psi)

    def step(self, steps=1):
        for _ in range(steps):
            self.psi *= self.U_clause_half
            self.psi = self.U_mix @ self.psi
            self.psi *= self.U_clause_half
            self.normalize()
            self.t += self.dt

    def probs(self):
        return np.abs(self.psi) ** 2

    def sat_mask(self):
        return self.H_clause_diag == 0

    def unsat_mask(self):
        return ~self.sat_mask()

    def sat_mass(self):
        p = self.probs()
        return float(np.sum(p[self.sat_mask()]))

    def frustration_mass(self):
        p = self.probs()
        return float(np.sum(p[self.unsat_mask()]))

    def expected_violations(self):
        return float(np.sum(self.probs() * self.H_clause_diag))

    def overlap_ground(self):
        H = np.diag(self.lam * self.H_clause_diag) - self.mix * self.H_mix
        vals, vecs = np.linalg.eigh(H)
        g = vecs[:, np.argmin(vals)]
        return float(np.abs(np.vdot(g, self.psi)) ** 2)

    def collapse(self, idx=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        p = self.probs()
        if idx is None:
            idx = int(rng.choice(len(p), p=p / np.sum(p)))
        self.psi[:] = 0.0
        self.psi[idx] = 1.0
        self.last_event = "collapse"
        self.last_index = idx

    def abs_closure(self, idx=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        p = self.probs()
        if idx is None:
            idx = int(rng.choice(len(p), p=p / np.sum(p)))
        target_v = int(self.H_clause_diag[idx])
        env = np.sqrt(p)
        sector = np.array([1.0 if int(v) == target_v else 1.0 / (1.0 + abs(v - target_v)) for v in self.H_clause_diag])
        self.psi = (env * sector).astype(complex)
        self.normalize()
        self.last_event = "ABS"
        self.last_index = idx

    def monodromy_flip(self):
        K = len(self.sample_bits)
        phase = np.ones(K, dtype=complex)
        for i, bits in enumerate(self.sample_bits):
            weight = sum(bits)
            branch = -1.0 if bits[0] == 1 else 1.0
            theta = np.pi * (i / max(1, K - 1))
            phase[i] = branch * np.exp(1j * (theta + np.pi * (weight % 2) / 2.0))
        self.psi *= phase
        self.normalize()
        self.last_event = "monodromy"
        self.monodromy_count += 1

    def winding_proxy(self):
        ph = np.unwrap(np.angle(self.psi + 1e-15))
        return float((ph[-1] - ph[0]) / (2.0 * np.pi))

    def top_k_view(self, k=32):
        p = self.probs()
        idx = np.argsort(-p)[:min(k, len(p))]
        return idx, p[idx], np.angle(self.psi[idx])


# ----------------------------
# App
# ----------------------------
class RimmerSATWaveApp:
    def __init__(self):
        self.inst_idx = 0
        self.custom_path = None
        self.engine = self.make_engine_from_builtin(self.inst_idx)
        self.paused = False
        self.t_hist = []
        self.sat_hist = []
        self.frus_hist = []
        self.energy_hist = []

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2.5, 1.2], width_ratios=[1.2, 1.2, 1.1])

        self.ax_amp = self.fig.add_subplot(gs[0, 0])
        self.ax_phase = self.fig.add_subplot(gs[0, 1])
        self.ax_compare = self.fig.add_subplot(gs[0, 2])

        self.ax_prob = self.fig.add_subplot(gs[1, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_info.axis("off")

        self.fig.canvas.manager.set_window_title("Rimmer SAT Wave V4.3 — dual mode + DIMACS")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.timer = self.fig.canvas.new_timer(interval=35)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.refresh()

    def _choose_engine(self, n, clauses):
        if n <= 12:
            return ExactSATWaveEngine(n, clauses)
        return ProjectedSATWaveEngine(n, clauses, k_samples=256)

    def make_engine_from_builtin(self, idx):
        inst = INSTANCES[idx]
        return self._choose_engine(inst["n"], inst["clauses"])

    def make_engine_from_dimacs(self, path):
        n, clauses = parse_dimacs_cnf(path)
        return self._choose_engine(n, clauses)

    def reset_history(self):
        self.t_hist = []
        self.sat_hist = []
        self.frus_hist = []
        self.energy_hist = []

    def load_builtin(self, idx):
        self.custom_path = None
        self.inst_idx = idx % len(INSTANCES)
        self.engine = self.make_engine_from_builtin(self.inst_idx)
        self.reset_history()

    def load_dimacs_dialog(self):
        if not TK_AVAILABLE:
            raise RuntimeError("tkinter is not available on this Python installation.")
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select DIMACS CNF file",
            filetypes=[("DIMACS CNF", "*.cnf *.dimacs *.txt"), ("All files", "*.*")]
        )
        root.destroy()
        if not path:
            return
        self.engine = self.make_engine_from_dimacs(path)
        self.custom_path = path
        self.reset_history()

    def current_instance_name(self):
        if self.custom_path:
            return f"DIMACS: {os.path.basename(self.custom_path)}"
        return INSTANCES[self.inst_idx]["name"]

    def preview_states(self):
        psi_backup = self.engine.psi.copy()
        rng = np.random.default_rng(123)
        p = self.engine.probs()
        idx = int(rng.choice(len(p), p=p / np.sum(p)))

        self.engine.psi = psi_backup.copy()
        self.engine.collapse(idx=idx)
        p_col = self.engine.probs()

        self.engine.psi = psi_backup.copy()
        self.engine.abs_closure(idx=idx)
        p_abs = self.engine.probs()

        self.engine.psi = psi_backup
        self.engine.normalize()
        return idx, p_col, p_abs

    def refresh(self):
        idx_order, p_top, phase_top = self.engine.top_k_view(k=32)
        labels_top = [self.engine.labels[i] for i in idx_order]

        self.ax_amp.clear()
        self.ax_amp.bar(np.arange(len(idx_order)), p_top)
        for j, i in enumerate(idx_order):
            if self.engine.sat_mask()[i]:
                self.ax_amp.bar(j, p_top[j], color='tab:green')
        self.ax_amp.set_title(f"top assignments ({self.engine.mode_name} mode)")
        self.ax_amp.set_xticks(np.arange(len(idx_order)))
        self.ax_amp.set_xticklabels(labels_top, rotation=90, fontsize=8)

        self.ax_phase.clear()
        self.ax_phase.bar(np.arange(len(idx_order)), phase_top)
        self.ax_phase.set_title("top assignment phases")
        self.ax_phase.set_xticks(np.arange(len(idx_order)))
        self.ax_phase.set_xticklabels(labels_top, rotation=90, fontsize=8)

        meas_idx, p_col, p_abs = self.preview_states()
        # restrict previews to top indices
        map_pos = {i: j for j, i in enumerate(idx_order)}
        y_col = np.array([p_col[i] for i in idx_order])
        y_abs = np.array([p_abs[i] for i in idx_order])
        x = np.arange(len(idx_order))
        w = 0.38

        self.ax_compare.clear()
        self.ax_compare.bar(x - w / 2, y_col, width=w, label="collapse")
        self.ax_compare.bar(x + w / 2, y_abs, width=w, label="ABS")
        meas_label = self.engine.labels[meas_idx]
        self.ax_compare.set_title(f"preview at measured index {meas_label}")
        self.ax_compare.set_xticks(np.arange(len(idx_order)))
        self.ax_compare.set_xticklabels(labels_top, rotation=90, fontsize=8)
        self.ax_compare.legend(loc="upper right", fontsize=8)

        if len(self.t_hist) == 0 or (self.engine.t > self.t_hist[-1]):
            self.t_hist.append(self.engine.t)
            self.sat_hist.append(self.engine.sat_mass())
            self.frus_hist.append(self.engine.frustration_mass())
            self.energy_hist.append(self.engine.expected_violations())

        self.ax_prob.clear()
        self.ax_prob.plot(self.t_hist, self.sat_hist, lw=2, label="SAT mass")
        self.ax_prob.plot(self.t_hist, self.frus_hist, lw=2, label="UNSAT frustration")
        self.ax_prob.set_title("SAT vs frustration mass")
        self.ax_prob.legend(loc="best")

        self.ax_energy.clear()
        self.ax_energy.plot(self.t_hist, self.energy_hist, lw=2, label="expected violations")
        self.ax_energy.set_title("violation energy")
        self.ax_energy.legend(loc="best")

        current_idx = "none" if self.engine.last_index is None else self.engine.labels[self.engine.last_index]
        info = (
            f"instance={self.current_instance_name()}\n"
            f"mode={self.engine.mode_name}\n"
            f"n={self.engine.n}, visible_dim={len(self.engine.labels)}\n"
            f"clauses={len(self.engine.clauses)}\n"
            f"t={self.engine.t:.3f}\n"
            f"sat_mass={self.engine.sat_mass():.5f}\n"
            f"frustration={self.engine.frustration_mass():.5f}\n"
            f"E[viol]={self.engine.expected_violations():.5f}\n"
            f"ground_overlap={self.engine.overlap_ground():.5f}\n"
            f"winding~={self.engine.winding_proxy():.3f}\n"
            f"last_event={self.engine.last_event}\n"
            f"last_index={current_idx}\n"
            f"monodromy_count={self.engine.monodromy_count}\n\n"
            f"[space]=pause  [c]=collapse  [a]=ABS\n"
            f"[m]=monodromy   [i]=builtin instance\n"
            f"[l]=load DIMACS [r]=reset [s]=resample"
        )
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(0.0, 1.0, info, va="top", family="monospace")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key.lower() if event.key else ""
        try:
            if key == " ":
                self.paused = not self.paused
            elif key == "c":
                self.engine.collapse()
            elif key == "a":
                self.engine.abs_closure()
            elif key == "m":
                self.engine.monodromy_flip()
            elif key == "i":
                self.load_builtin(self.inst_idx + 1)
            elif key == "l":
                self.load_dimacs_dialog()
            elif key == "r":
                self.engine.reset()
                self.reset_history()
            elif key == "s" and hasattr(self.engine, "resample"):
                self.engine.resample()
                self.reset_history()
        except Exception as e:
            self.ax_info.clear()
            self.ax_info.axis("off")
            self.ax_info.text(0.0, 1.0, f"ERROR:\n{e}", va="top", family="monospace")
            self.fig.canvas.draw_idle()
            return
        self.refresh()

    def tick(self):
        if not self.paused:
            self.engine.step(steps=1)
            self.refresh()


if __name__ == "__main__":
    app = RimmerSATWaveApp()
    plt.show()
