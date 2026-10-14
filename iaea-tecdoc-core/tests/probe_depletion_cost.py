"""
tests/probe_depletion_cost.py
-----------------------------
REDUCED-ZONING DEPLETION COST PROBE — a Task 2 deliverable, not a smoke test.

THE QUESTION IT ANSWERS. At the live 2 x 10 per-plate zoning (12,280 depletable
materials) a depletion run's step 0 transport finished in ~5 minutes and the
process then sat at ~94.4% CPU — ONE thread of 128 — for 2+ hours with RSS
climbing 9.3 -> 25.2 GB. That is serial Python doing reaction-rate extraction
and burnup-matrix assembly. Particle count is irrelevant to it.

    Is that serial phase PER-STEP, or ONE-TIME SETUP?

Everything downstream depends on the answer. Per-step at 17 steps means ~34 h;
one-time means ~2 h plus transport. The discriminator is not the total, it is
the SHAPE of the per-step timing curve — hence this script instruments every
step rather than reporting a single wall time.

METHOD. Two constants change and nothing else: N_X_ZONES and N_AXIAL_ZONES,
giving 614 plates x 1 x 2 = 1,228 depletable materials, exactly 1/10 of the
live scheme. Full step count, so the per-step curve is resolved.

    per-step time flat and >> setup   -> PER-STEP. cost ~ materials x steps.
    one large step 0, rest small      -> ONE-TIME. cost ~ materials.

Either way the run also yields the scaling law — depletion cost vs depletable
material count — which is the "computational resources needed for ADDER-OpenMC"
question in the project scope, and a stronger Task 2 result than a runtime
ratio.

INSTRUMENTATION. depletion_results.h5 is written incrementally, once per step,
so its mtime IS the per-step boundary. A sampler thread watches it and records
(step, wall_s, rss_gb). No change to the production code path.

RESTORE AFTERWARD. Zoning returns to 2 x 10 per-plate — that scheme is
[MCNP — Kyle confirmed 2026-08-12] and 1 x 2 is a measurement configuration
only. This script never writes to materials.py.

    python tests/probe_depletion_cost.py --steps 17 --step-days 30
"""
import argparse, os, sys, threading, time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))

PROBE_N_X, PROBE_N_Z = 1, 2          # 614 x 1 x 2 = 1,228 materials


def _rss_gb():
    try:
        with open('/proc/self/status') as f:
            for ln in f:
                if ln.startswith('VmRSS:'):
                    return int(ln.split()[1]) / 1048576.0
    except OSError:
        pass
    return float('nan')


def _sampler(results_path, log, stop, t0):
    """Record a row each time depletion_results.h5 gains a step."""
    last_size, step = -1, 0
    while not stop.is_set():
        try:
            sz = os.path.getsize(results_path)
            if sz != last_size:
                if last_size >= 0:
                    step += 1
                    log.append((step, time.time() - t0, _rss_gb(), sz))
                    print(f"  [probe] step {step:2d}  t={log[-1][1]:8.1f}s  "
                          f"rss={log[-1][2]:5.2f} GB", flush=True)
                last_size = sz
        except OSError:
            pass
        stop.wait(5.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    p.add_argument('--steps', type=int, default=17)
    p.add_argument('--step-days', type=float, default=30.0)
    p.add_argument('--particles', type=int, default=50000)
    p.add_argument('--batches', type=int, default=100)
    p.add_argument('--inactive', type=int, default=30)
    p.add_argument('--output-dir', default=os.path.join(REPO_ROOT, 'run_results',
                                                        'probe_depletion_cost'))
    p.add_argument('--csv', default=None)
    args = p.parse_args(argv)

    # Patch the zone counts BEFORE geometry imports them. Same mechanism
    # tests/check_depletion_zoning.py:_run_degenerate uses.
    import materials
    materials.N_X_ZONES, materials.N_AXIAL_ZONES = PROBE_N_X, PROBE_N_Z
    import importlib, geometry
    importlib.reload(geometry)

    import core
    from settings import format_provenance

    n_plates = 23 * geometry.N_PLATES_STD + 5 * geometry.N_CTRL_FUEL_PLATES
    n_mats = n_plates * PROBE_N_X * PROBE_N_Z
    print("Run provenance:"); print(format_provenance()); print()
    print(f"PROBE zoning {PROBE_N_X} x {PROBE_N_Z} -> {n_mats} depletable "
          f"materials ({n_plates} plates), {args.steps} steps")
    print(f"  live scheme is 2 x 10 -> {n_plates * 20} materials; this is "
          f"1/{(n_plates * 20) // n_mats} of it\n")

    cfg = core.CoreConfig()
    cfg.depletion_zoning = True
    cfg.particles, cfg.batches, cfg.inactive = (args.particles, args.batches,
                                                args.inactive)
    cfg.depletion_timesteps = [(args.step_days, 'd')] * args.steps
    cfg.output_dir = args.output_dir

    results = os.path.join(os.path.abspath(args.output_dir),
                           'depletion_results.h5')
    log, stop, t0 = [], threading.Event(), time.time()
    th = threading.Thread(target=_sampler, args=(results, log, stop, t0),
                          daemon=True)
    th.start()
    try:
        core.run_depletion(cfg)
    finally:
        stop.set(); th.join(timeout=10)

    total = time.time() - t0
    csv = args.csv or os.path.join(os.path.abspath(args.output_dir),
                                   'probe_depletion_cost.csv')
    with open(csv, 'w') as f:
        f.write('step,wall_s,delta_s,rss_gb,results_bytes\n')
        prev = 0.0
        for st, w, r, sz in log:
            f.write(f'{st},{w:.1f},{w - prev:.1f},{r:.2f},{sz}\n'); prev = w
    print(f"\nTotal {total:.1f} s over {len(log)} recorded steps -> {csv}")
    if len(log) >= 3:
        d = [log[i][1] - log[i - 1][1] for i in range(1, len(log))]
        first, rest = d[0], sum(d[1:]) / max(len(d) - 1, 1)
        print(f"  step 1 delta {first:.1f} s;  mean of the rest {rest:.1f} s;  "
              f"ratio {first / rest if rest else float('nan'):.2f}")
        print("  ratio near 1  -> PER-STEP serial cost")
        print("  ratio >> 1    -> ONE-TIME setup, amortised over the run")
    print("\nRESTORE zoning to 2 x 10 per-plate before any production run.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
