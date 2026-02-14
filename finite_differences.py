import numpy as np

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
], dtype=float)

w = np.array([0.5, -1.0], dtype=float)

b = np.array([1.0, 0.0, -1.0], dtype=float)

def lsq_grad(w, A, b):
    return A.T @ (A @ w - b)

def get_u(i, total):
    eyes = np.eye(total)
    return eyes[i]

# print(lsq_grad(w, A, b))
def f(w, A, b):
    # AI NOTICE: used CoPilot with GPT 5.2 to write this line below
    return 0.5 * np.linalg.norm(A @ w - b) ** 2

def lsq_finite_diff_grad(w, A, b, epsilon=1e-5):
    w_len = w.shape[0]
    # algorithm
    # for each w_i in w:
    #compute f(w + epsilon * get_u(i, w_len))
    # compute above but with -
    # compute diff over 2 epsilon
    def get_grad_i(i):
        w_plus = w + epsilon * get_u(i, w_len)
        w_minus = w - epsilon * get_u(i, w_len)
        grad_i = (f(w_plus, A, b) - f(w_minus, A, b)) / (2 * epsilon)
        return grad_i
    
    return np.array([get_grad_i(i) for i in range(w_len)])

# print(get_u(3, 5))
# print(f(w, A, b))
print(lsq_finite_diff_grad(w, A, b))

# --- AI NOTICE ---
# The verification / gradient-check code below was generated with AI assistance (GitHub Copilot, GPT-5.2).
# It is intended only to sanity-check that the finite-difference implementation matches the analytic gradient.

def _run_random_gradient_checks(num_trials=8, epsilon=1e-6, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)

    # A few different (n, d) shapes to ensure we aren't accidentally relying on one fixed size.
    shapes = [(3, 2), (5, 3), (8, 4)]
    trial = 0
    for n, d in shapes:
        for _ in range(num_trials):
            trial += 1
            A_rand = rng.normal(size=(n, d))
            w_rand = rng.normal(size=(d,))
            b_rand = rng.normal(size=(n,))

            g_analytic = lsq_grad(w_rand, A_rand, b_rand)
            g_fd = lsq_finite_diff_grad(w_rand, A_rand, b_rand, epsilon=epsilon)

            max_abs_err = np.max(np.abs(g_analytic - g_fd))
            ok = max_abs_err < tol
            print(
                f"trial {trial:02d} shape (n={n}, d={d}) max|analytic-fd|={max_abs_err:.3e} "
                f"{'OK' if ok else 'FAIL'}"
            )
            if not ok:
                print("analytic:", g_analytic)
                print("finite-diff:", g_fd)
                return False
    return True


print("\nRandom gradient checks:")
_run_random_gradient_checks()