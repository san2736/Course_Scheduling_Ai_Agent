"""Runs everything and writes results/ + plots.

    python run_experiments.py --episodes 1500 --load 90
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler import make_scenario, SchedulingEnv
from scheduler.train import train, evaluate
from scheduler.baselines import run_baseline
from scheduler.agents import ACTION_NAMES

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def smooth(x, k=50):
    if len(x) < k:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def baselines_for(n_sessions, seed, max_rounds):
    sc = make_scenario(n_sessions=n_sessions, seed=seed)
    out = {}
    for name in ("random", "greedy", "greedy_pref"):
        env = SchedulingEnv(sc, max_rounds=max_rounds, seed=seed)
        out[name] = run_baseline(env, name, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--load", type=int, default=90, help="number of sessions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    report = {"config": vars(args)}
    t0 = time.time()

    # ---------------- main run: Expected SARSA
    print(f"[1/4] Training Expected SARSA, {args.load} sessions, "
          f"{args.episodes} episodes")
    es = train(n_sessions=args.load, episodes=args.episodes, seed=args.seed,
               max_rounds=args.max_rounds, algo="expected_sarsa",
               log_every=max(1, args.episodes // 6))
    es_eval = evaluate(es["env"], es["depts"], es["rooms"], n=30)
    report["expected_sarsa"] = es_eval
    print("   eval:", {k: round(v, 2) for k, v in es_eval.items()})

    # ---------------- comparison arm: Independent Q-learning
    print("[2/4] Training Independent Q-Learning (comparison arm)")
    ql = train(n_sessions=args.load, episodes=args.episodes, seed=args.seed,
               max_rounds=args.max_rounds, algo="q_learning",
               log_every=max(1, args.episodes // 6))
    ql_eval = evaluate(ql["env"], ql["depts"], ql["rooms"], n=30)
    report["q_learning"] = ql_eval
    print("   eval:", {k: round(v, 2) for k, v in ql_eval.items()})

    # ---------------- baselines
    print("[3/4] Baselines")
    report["baselines"] = baselines_for(args.load, args.seed, args.max_rounds)
    for k, v in report["baselines"].items():
        print(f"   {k:12} placed {v['placed']}/{args.load} "
              f"pref {v['preference']:.2f} fair_min {v['fairness_min']:.2f}")

    # ---------------- action usage (emergent behaviour evidence)
    usage = np.sum([a.action_counts for a in es["depts"]], axis=0)
    usage = usage / max(1.0, usage.sum())
    report["action_mix"] = {n: float(round(u, 3)) for n, u in zip(ACTION_NAMES, usage)}
    report["q_table_sizes"] = [a.table_size() for a in es["depts"]]

    # ---------------- load sweep
    if args.sweep:
        print("[4/4] Load sweep")
        sweep = {}
        for n in (30, 60, 90, 105):
            r = train(n_sessions=n, episodes=max(400, args.episodes // 3),
                      seed=args.seed, max_rounds=args.max_rounds, verbose=False)
            ev = evaluate(r["env"], r["depts"], r["rooms"], n=20)
            base = baselines_for(n, args.seed, args.max_rounds)
            sweep[n] = {"agents": ev, "baselines": base}
            print(f"   {n:3} sessions | agents pref {ev['preference']:.2f} "
                  f"success {ev['success']:.2f} | greedy_pref pref "
                  f"{base['greedy_pref']['preference']:.2f} "
                  f"success {base['greedy_pref']['success']:.0f}")
        report["sweep"] = sweep

    report["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    np.save(os.path.join(RES, "curve_es.npy"), np.array(es["log"]["reward"]))
    np.save(os.path.join(RES, "curve_ql.npy"), np.array(ql["log"]["reward"]))
    np.save(os.path.join(RES, "curve_pref.npy"), np.array(es["log"]["preference"]))
    np.save(os.path.join(RES, "curve_success.npy"), np.array(es["log"]["success"]))

    # ---------------- one sample timetable, for the report
    with open(os.path.join(RES, "sample_timetable.txt"), "w") as f:
        f.write(es["env"].grid.render())

    print(f"\nDone in {report['runtime_sec']}s -> results/report.json")


if __name__ == "__main__":
    main()
