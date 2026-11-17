"""e2_instrument.py --- look at the optimiser instead of its output.

Both solvers are reproduced here verbatim from the implementations used for
inference, with logging added and no change to the arithmetic. The point of the
paper is undermined if the instrumented solver is not the solver, so the inner
loops below must stay byte-for-byte equivalent to
`common.notears_legacy` and `structure_learners.notears_linear`.

What gets logged per outer iteration:
  * ||W||_F, so collapse to the origin is visible directly;
  * h(W), rho, alpha, the augmented-Lagrangian state;
  * for the legacy solver, the gradient split into its data term and its
    acyclicity term, the norm before clipping, and the clip factor applied.

The acyclicity gradient is E^T (*) 2W, which is proportional to W entrywise and
therefore points away from the origin; subtracting it moves W toward zero. Once
the multipliers make that term dominate, and the clip removes the scale
information that would otherwise let the data term compete, each outer
iteration returns W to its initialisation.

Output: results/e2_instrument.json
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

import jdssv_common as J
import structure_learners as SL


def trace_iterates_legacy(X, lambda1=J.LAMBDA1, max_iter=50, h_tol=1e-8,
                          rho_max=1e16, clip=10.0, lr0=0.01):
    """`common.notears_legacy` with logging. Arithmetic unchanged."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    W = np.zeros((d, d))
    rho, alpha = 1.0, 0.0
    log = []

    def _h(W):
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        return np.trace(E) - d

    def _grad_parts(W):
        R = X - X @ W
        grad_loss = -1.0 / n * X.T @ R
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        grad_h = E.T * W * 2
        mult = rho * _h(W) + alpha
        g_raw = grad_loss + mult * grad_h + lambda1 * np.sign(W)
        gnorm = np.linalg.norm(g_raw)
        factor = (clip / gnorm) if gnorm > clip else 1.0
        return grad_loss, mult * grad_h, g_raw, gnorm, factor

    for outer in range(max_iter):
        W_old = W.copy()
        gl, gh, g_raw, gnorm, factor = _grad_parts(W)
        for step in range(50):
            lr = lr0 / (1 + step * 0.01)
            _, _, g, gn, f = _grad_parts(W)
            g = g * f
            W_new = W - lr * g
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * lambda1, 0)
            np.fill_diagonal(W_new, 0)
            if not np.isfinite(W_new).all():
                break
            W = W_new
        h_val = _h(W)
        log.append({
            "outer": outer,
            "W_norm": float(np.linalg.norm(W)),
            "W_norm_before_inner": float(np.linalg.norm(W_old)),
            "max_abs_w": float(np.abs(W).max()),
            "h": float(h_val) if np.isfinite(h_val) else None,
            "rho": float(rho), "alpha": float(alpha),
            "grad_data_norm": float(np.linalg.norm(gl)),
            "grad_acyclicity_norm": float(np.linalg.norm(gh)),
            "grad_raw_norm": float(gnorm),
            "clip_factor": float(factor),
            "clipped": bool(factor < 1.0),
        })
        if not np.isfinite(h_val):
            W = W_old
            break
        if h_val > 0.25 * _h(W_old) and _h(W_old) > h_tol:
            rho *= 10
        alpha += rho * h_val
        if h_val < h_tol or rho > rho_max:
            break
    return W, log


def trace_iterates_corrected(X, lambda1=J.LAMBDA1, max_iter=100, h_tol=1e-8,
                             rho_max=1e16):
    """`structure_learners.notears_linear` with logging. Arithmetic unchanged."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    log = []

    def _loss(W):
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):
        E = slin.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G_smooth + lambda1, -G_smooth + lambda1),
                                   axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]
    for outer in range(max_iter):
        w_new, h_new = w_est, h
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True,
                                bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        W_cur = _adj(w_est)
        G_loss = _loss(W_cur)[1]
        _, G_h = _h(W_cur)
        log.append({
            "outer": outer,
            "W_norm": float(np.linalg.norm(W_cur)),
            "max_abs_w": float(np.abs(W_cur).max()),
            "h": float(h), "rho": float(rho), "alpha": float(alpha),
            "grad_data_norm": float(np.linalg.norm(G_loss)),
            "grad_acyclicity_norm": float(np.linalg.norm((rho * h + alpha) * G_h)),
            "clipped": False, "clip_factor": 1.0,
        })
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    return _adj(w_est), log


def main():
    X, names = J.load_nepal()
    J.print_banner("e2 --- instrumenting the optimiser")

    W_l, log_l = trace_iterates_legacy(X)
    W_c, log_c = trace_iterates_corrected(X)

    # Confirm the instrumented copies reproduce the solvers used for inference.
    W_l_ref = J.compute_legacy_W(X)
    W_c_ref = J.compute_corrected_W(X)
    dev_l = float(np.abs(W_l - W_l_ref).max())
    dev_c = float(np.abs(W_c - W_c_ref).max())
    print(f"  instrumented vs reference solver, max |deviation|:")
    print(f"    legacy    {dev_l:.3e}")
    print(f"    corrected {dev_c:.3e}")
    assert dev_l < 1e-10, "instrumented legacy solver diverged from the reference"
    assert dev_c < 1e-8, "instrumented corrected solver diverged from the reference"

    print(f"\n  legacy: {len(log_l)} outer iterations")
    print(f"    ||W|| at outer 0 / 1 / last = "
          f"{log_l[0]['W_norm']:.4f} / "
          f"{log_l[min(1, len(log_l) - 1)]['W_norm']:.4f} / "
          f"{log_l[-1]['W_norm']:.4f}")
    n_clipped = sum(r["clipped"] for r in log_l)
    print(f"    outer iterations where the gradient was clipped: "
          f"{n_clipped}/{len(log_l)}")
    ratios = [r["grad_acyclicity_norm"] / max(r["grad_data_norm"], 1e-30)
              for r in log_l]
    print(f"    acyclicity/data gradient-norm ratio: "
          f"first {ratios[0]:.3e}, max {max(ratios):.3e}")

    print(f"\n  corrected: {len(log_c)} outer iterations")
    print(f"    ||W|| at outer 0 / last = "
          f"{log_c[0]['W_norm']:.4f} / {log_c[-1]['W_norm']:.4f}")
    print(f"    h at last outer = {log_c[-1]['h']:.3e}")

    out = {
        "operating_point": {"lambda1": J.LAMBDA1, "tau": J.TAU},
        "verification": {"legacy_max_dev": dev_l, "corrected_max_dev": dev_c},
        "legacy": {"n_outer": len(log_l), "log": log_l,
                   "n_outer_clipped": n_clipped,
                   "final_W_norm": log_l[-1]["W_norm"],
                   "final_max_abs_w": log_l[-1]["max_abs_w"]},
        "corrected": {"n_outer": len(log_c), "log": log_c,
                      "final_W_norm": log_c[-1]["W_norm"],
                      "final_max_abs_w": log_c[-1]["max_abs_w"]},
    }
    J.save_json("e2_instrument", out)
    return out


if __name__ == "__main__":
    main()
