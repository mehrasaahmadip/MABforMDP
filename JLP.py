# complete_grid_bandit_vi.py
import numpy as np
import random
from collections import defaultdict

############################################
# 4×4 slippery grid environment
############################################
class SlipperyGrid:
    def __init__(self, n=4, slip_p=0.8, holes=None, rng=None):
        self.n = n
        self.S = n * n
        self.A = 4  # actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.slip_p = slip_p
        self.holes = set(holes or [])
        self.goal = self.S - 1
        self.rng = rng or random.Random()
        self.slip_map = {
            0: [2, 3],  # UP  slips to LEFT/RIGHT
            1: [2, 3],  # DOWN slips to LEFT/RIGHT
            2: [0, 1],  # LEFT slips to UP/DOWN
            3: [0, 1],  # RIGHT slips to UP/DOWN
        }
        self.reset()

    def to_state(self, i, j):
        return i * self.n + j

    def to_pos(self, s):
        return divmod(s, self.n)

    def reset(self):
        self.state = 0
        return self.state

    def set_state(self, s):
        self.state = s
        return s

    def _move(self, i, j, a):
        if a == 0:
            i = max(i - 1, 0)
        elif a == 1:
            i = min(i + 1, self.n - 1)
        elif a == 2:
            j = max(j - 1, 0)
        elif a == 3:
            j = min(j + 1, self.n - 1)
        return i, j

    def step(self, a):
        acts = [a] + self.slip_map[a]
        probs = [1 - self.slip_p, 0.5 * self.slip_p, 0.5 * self.slip_p]
        a_eff = self.rng.choices(acts, probs)[0]
        i, j = self.to_pos(self.state)
        ni, nj = self._move(i, j, a_eff)
        s_next = self.to_state(ni, nj)
        reward = 1.0 if s_next == self.goal else 0.0
        done = reward == 1.0 or s_next in self.holes
        self.state = s_next
        return s_next, reward, done

############################################
# KL-LUCB internals (based on S. Filippi et al.)
############################################
def _maxKL(p, V, eps, it=60):
    p, V = np.asarray(p, float), np.asarray(V, float)
    pos = p > 0
    if pos.sum() <= 1:
        return p.copy()
    def f(nu):
        d = nu - V[pos]
        if (d <= 0).any():
            return np.inf
        return (p[pos] * np.log(d)).sum() + np.log((p[pos] / d).sum()) - eps
    lo = V[pos].max() + 1e-9
    if f(lo) <= 0:
        return p.copy()
    hi = lo + 1.0
    while f(hi) > 0:
        hi *= 2
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    nu = 0.5 * (lo + hi)
    diff = nu - V[pos]
    q = np.zeros_like(p)
    q[pos] = p[pos] / diff
    q[pos] /= q[pos].sum()
    return q


def _minKL(p, V, eps):
    return _maxKL(p, -V, eps)


def _kl_rad(n, t, K, delta, d):
    if n < 1:
        return np.inf
    return np.log((K * d * (t + 1)) / delta) / n

############################################
# KL-LUCB bandit for one state
############################################
def KL_state_unknown(env, s, V, counts, trans, max_pulls=200, delta=0.05):
    K, d = env.A, env.S
    # ensure counts/trans entries exist
    for a in range(K):
        counts[(s, a)]
        trans[(s, a)]
    t = 0
    while t < max_pulls:
        t += 1
        # empirical distribution per action
        p_hat = np.array([trans[(s, a)] / max(1, counts[(s, a)]) for a in range(K)])
        # compute KL bounds
        U = np.zeros(K)
        L = np.zeros(K)
        for a in range(K):
            eps = _kl_rad(counts[(s, a)], t, K, delta, d)
            U[a] = _maxKL(p_hat[a], V, eps).dot(V)
            L[a] = _minKL(p_hat[a], V, eps).dot(V)
        leader = int(np.argmax(U))
        challenger = int(np.argmax(np.where(np.arange(K) == leader, -np.inf, U)))
        # check stopping condition
        if L[leader] > U[challenger]:
            break
        # sample both leader and challenger once
        for a in (leader, challenger):
            env.set_state(s)
            s2, r, _ = env.step(a)
            counts[(s, a)] += 1
            trans[(s, a)][s2] += 1
    return leader, t

############################################
# Online model-based control with KL-LUCB
############################################
def bandit_grid_control(env, gamma=0.95, delta=0.05, max_passes=10, max_pulls_state=200):
    S, A = env.S, env.A
    counts = defaultdict(int)
    trans = defaultdict(lambda: np.zeros(S, int))
    V = np.zeros(S)
    V[env.goal] = 1.0
    policy = np.zeros(S, int)

    # initial sweep: one sample per (s,a)
    for s in range(S):
        if s == env.goal or s in env.holes:
            continue
        for a in range(A):
            env.set_state(s)
            s2, r, _ = env.step(a)
            counts[(s, a)] += 1
            trans[(s, a)][s2] += 1

    # main loop
    for _ in range(max_passes):
        # policy improvement via KL-LUCB
        for s in range(S):
            if s == env.goal or s in env.holes:
                continue
            best, pulls = KL_state_unknown(env, s, V, counts, trans, max_pulls_state, delta)
            policy[s] = best
        # build empirical model
        P_hat = np.zeros((S, A, S))
        for (s, a), n in counts.items():
            if n > 0:
                P_hat[s, a] = trans[(s, a)] / n
            else:
                P_hat[s, a, s] = 1.0
        # policy evaluation via value iteration
        for _ in range(200):
            V_new = V.copy()
            for s in range(S):
                if s == env.goal or s in env.holes:
                    continue
                V_new[s] = np.max(P_hat[s] @ V)
            if np.max(np.abs(V_new - V)) < 1e-6:
                break
            V = V_new
    total_samples = sum(counts.values())
    return policy, V, total_samples

############################################
# Model-free Q-learning baseline
############################################
def qlearn_grid(env, episodes=30000, alpha=0.1, gamma=0.95, eps0=1.0, eps_end=0.05, eps_decay=0.9995):
    S, A = env.S, env.A
    Q = np.zeros((S, A))
    eps = eps0
    steps = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(A)
            else:
                a = int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s = s2
            steps += 1
        eps = max(eps_end, eps * eps_decay)
    policy = np.argmax(Q, axis=1)
    return policy, steps

############################################
# Evaluation
############################################
def evaluate(env, policy, episodes=2000):
    total_steps, successes = 0, 0
    for _ in range(episodes):
        s = env.reset(); done = False; steps = 0
        while not done and steps < 100:
            a = policy[s]
            s, r, done = env.step(a)
            steps += 1
        total_steps += steps
        if s == env.goal:
            successes += 1
    return total_steps / episodes, successes / episodes

############################################
# Main
############################################
if __name__ == '__main__':
    random.seed(0); np.random.seed(0)
    env = SlipperyGrid()

    pi_kl, V_kl, samps = bandit_grid_control(env)
    steps_kl, succ_kl = evaluate(SlipperyGrid(), pi_kl)
    print("KL-LUCB policy:\n", pi_kl.reshape(env.n, env.n))
    print(f"Samples used: {samps}")
    print(f"Test avg steps: {steps_kl:.1f}, success: {succ_kl*100:.1f}%\n")

    random.seed(1); np.random.seed(1)
    env2 = SlipperyGrid()
    pi_q, q_steps = qlearn_grid(env2)
    steps_q, succ_q = evaluate(SlipperyGrid(), pi_q)
    print("Q-learning policy:\n", pi_q.reshape(env.n, env.n))
    print(f"Train steps: {q_steps}, Test avg steps: {steps_q:.1f}, success: {succ_q*100:.1f}%")