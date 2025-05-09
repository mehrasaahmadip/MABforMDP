import numpy as np
import random
from collections import defaultdict

###################################
###################################
# Slippery Frozen-Lake
#
class FrozenLakeEnv:
    def __init__(self, slip_p=0.8):
        self.n_rows = self.n_cols = 4
        self.S, self.A = 16, 4
        self.slip_p = slip_p
        self.holes = {(1, 1), (1, 3), (2, 3), (3, 0)}
        self.goal  = (3, 3)
        self.slip_map = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self.reset()

    def to_state(self, pos): return pos[0]*4 + pos[1]
    def to_pos(self, s):     return divmod(s, 4)

    def reset(self):                 # normal Gym start (0,0)
        self.state = 0
        return self.state

    def set_state(self, s):          # extra – lets us “probe” any state
        self.state = s
        return s

    def step(self, a):
        acts  = [a] + self.slip_map[a]
        probs = [self.slip_p, 0.5*(1-self.slip_p), 0.5*(1-self.slip_p)]
        act   = random.choices(acts, probs)[0]

        i, j  = self.to_pos(self.state)
        if act == 0: i = max(i-1, 0)
        elif act == 1: i = min(i+1, 3)
        elif act == 2: j = max(j-1, 0)
        elif act == 3: j = min(j+1, 3)

        s_next = self.to_state((i, j))
        reward = int((i, j) == self.goal)
        done   = (i, j) in self.holes or (i, j) == self.goal
        self.state = s_next
        return s_next, reward, done

###################################
###################################
###################################
###################################
# ────────────────────────────────────────────────────────────────
#   Bisection KL , based on S.Fillipi 2010
#
def _maxKL(p, V, eps, it=60):
    p, V = np.asarray(p, float), np.asarray(V, float)
    pos  = p > 0
    if pos.sum() <= 1:
        return p.copy()

    def f(nu):                               # KL(q||p)−eps
        d = nu - V[pos]
        if (d <= 0).any(): return np.inf
        return (p[pos]*np.log(d)).sum() + np.log((p[pos]/d).sum()) - eps

    lo = V[pos].max() + 1e-9
    if f(lo) <= 0:                           # already feasible
        return p.copy()

    hi = lo + 100
    while f(hi) > 0: hi *= 2

    for _ in range(it):                      # bisection
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    nu   = 0.5 * (lo + hi)
    diff = nu - V[pos]
    q    = np.zeros_like(p)
    q[pos] = p[pos] / diff
    q[pos] /= q[pos].sum()
    return q

def _minKL(p, V, eps):   return _maxKL(p, -V, eps)
def _kl_rad(n, t, K, delta, d):  # KL-LUCB radius
    return 1e5 if n < 1 else np.log((K*d*(t+1))/delta) / n


############################################
#  LUCB for one state, sampling the real env
#
def KL_state_unknown(env, s, V,
                     counts, trans,
                     max_pulls=750, delta=0.01): # A: ??change the delta, dependes of gamma,

    """
    Identify best action in state s, update shared (counts, trans),
    return (best_action, pulls_used).
    """
    K, d = env.A, env.S
    for a in range(K):   # make sure dict keys exist
        counts[(s,a)]; trans[(s,a)]  #Counts: at state 's', How # times we have executed action
                                     #trans[(s,a)][s′]:  How # of the action ended up in successor state s′.


    t = 0
    while t < max_pulls:
        t += 1
        p_hat = np.array([trans[(s,a)] / max(counts[(s,a)],1)
                          for a in range(K)])
        U = np.zeros(K); L = np.zeros(K)
        for a in range(K):
            eps = _kl_rad(counts[(s,a)], t, K, delta, d)
            U[a] = _maxKL(p_hat[a], V, eps).dot(V)
            L[a] = _minKL(p_hat[a], V, eps).dot(V)

        leader     = int(np.argmax(U))
        challenger = int(np.argmax(np.where(np.arange(K)==leader, -np.inf, U)))
        if L[leader] > U[challenger]:
            break

        arm = leader if random.random() < 0.5 else challenger
        env.set_state(s)
        s_next, _, _ = env.step(arm)
        counts[(s,arm)] += 1
        trans[(s,arm)][s_next] += 1

    return leader, t


#######################
#  KL-LUCB (EL-LUCB)
#
def learn_KL_unknown(env,
                     gamma            = 0.99,
                     delta            = 0.01,
                     max_passes       = 6,
                     max_pulls_state  = 750):

    S, A = env.S, env.A
    counts = defaultdict(int)
    trans  = defaultdict(lambda: np.zeros(S, int)) ## default [0,0,…]
    V      = np.zeros(S)
    policy = np.full(S, -1, int)

    for _ in range(max_passes):
        # ------------ improvement (bandit per state) -------------
        for s in range(S):
            pos = env.to_pos(s)
            if pos == env.goal or pos in env.holes: continue
            best, _ = KL_state_unknown(env, s, V,
                                       counts, trans,
                                       max_pulls=max_pulls_state,
                                       delta=delta)
            policy[s] = best

        # ------------ empirical model + value-iteration ----------
        P_hat = np.zeros((S, A, S))
        for (s, a), n_sa in counts.items():
            if n_sa > 0:
                P_hat[s, a] = trans[(s, a)] / n_sa
            else:
                P_hat[s, a] = np.ones(S) / S   # unseen: uniform

        R = np.zeros(S); R[env.to_state(env.goal)] = 1
        for _ in range(300): # 1/(1-\gamma) check ?
            # ???? have a backup from hat{P}, do not need to start from zero.

            EV = np.tensordot(P_hat, V, axes=[2, 0])      # (S,A)
            V_new = R + gamma * EV.max(axis=1)
            if np.max(np.abs(V_new - V)) < 1e-6:
                break
            V = V_new

    samples_per = np.zeros(S, int)
    for (s,a), n in counts.items():
        samples_per[s] += n # sum of all bandit experiments
    return policy, samples_per, samples_per.sum()


#################################
#   Model-free Q-learning baseline
#
def q_learning(env,
               episodes    = 20_000,
               alpha       = 0.1,
               gamma       = 0.99,
               eps_start   = 1.0,
               eps_final   = 0.05,
               eps_decay   = 0.9995):
    Q = np.zeros((env.S, env.A))
    eps = eps_start
    total_steps = 0
    for _ in range(episodes):
        s = env.reset(); done = False
        while not done:
            a = random.randrange(env.A) if random.random() < eps else int(np.argmax(Q[s]))
            s_next, r, done = env.step(a)
            Q[s,a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s,a])
            s = s_next
            total_steps += 1
        eps = max(eps_final, eps * eps_decay)

    return np.argmax(Q, axis=1), total_steps


#######
#  Evaluation part: avg steps & success rate
#######
def eval_policy(env, policy, episodes=1000, max_steps=100):
    steps_total, successes = 0, 0
    for _ in range(episodes):
        s = env.reset(); done=False; steps=0
        while not done and steps < max_steps:
            a = policy[s] if policy[s] != -1 else random.randrange(env.A)
            s, r, done = env.step(a)
            steps += 1
        steps_total += steps
        if r == 1: successes += 1
    return steps_total / episodes, successes / episodes


# ────────────────────────────────────────────────────────────────
#
if __name__ == "__main__":
    random.seed(42); np.random.seed(42)

    # ----- KL-LUCB learner -----
    env_KL = FrozenLakeEnv()
    policy_KL, samples_per, total_KL = learn_KL_unknown(
        env_KL, gamma=0.99, delta=0.001,
        max_passes=10, max_pulls_state=1000)

    avg_steps_KL, succ_KL = eval_policy(FrozenLakeEnv(), policy_KL)

    # ----- Q-learning baseline -----
    policy_Q, total_Q = q_learning(FrozenLakeEnv(),
                                   episodes=20_000,
                                   eps_decay=0.999)
    avg_steps_Q, succ_Q = eval_policy(FrozenLakeEnv(), policy_Q)

    # ----- Results -----
    print("===== KL-LUCB (model-based, unknown P) =====")
    print("Environment calls for learning :", total_KL)
    print(f"Avg steps to goal (test)       : {avg_steps_KL:.1f}")
    print(f"Success rate                   : {succ_KL*100:.1f}%\n")

    print("===== Q-learning (model-free)  =====")
    print("Environment steps for training :", total_Q)
    print(f"Avg steps to goal (test)       : {avg_steps_Q:.1f}")
    print(f"Success rate                   : {succ_Q*100:.1f}%")
