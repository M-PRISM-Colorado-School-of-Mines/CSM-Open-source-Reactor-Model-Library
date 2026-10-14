"""
core.py
-------
Central driver for the IAEA TECDOC-643 Appendix A-2 Generic 10 MW LEU
Research Reactor OpenMC model.

Single control surface: edit CoreConfig (or use the CLI) — everything else
(materials, geometry, settings, tallies) is assembled by build_model() so
every run goes through one code path with tallies attached.

CLI examples (from the repo root, conda env `openmc-env`):
    python model/core.py                               # defaults (all-in)
    python model/core.py --insertion 0 --particles 50000 --batches 150
    python model/core.py --insertion 68.5 --output-dir run_results/crit_search

BLADE DIRECTION CONVENTION (read this before touching insertion logic):
    CoreConfig.blade_insertion_percent is the INTUITIVE control-room sense:
        0   = blades fully WITHDRAWN  (core most reactive)
        100 = blades fully INSERTED   (absorber spans the active fuel)
    geometry.build_core_geometry() takes a WITHDRAWAL fraction f:
        f = 0.0 fully INSERTED, f = 1.0 fully WITHDRAWN
    build_model() converts: f_withdrawal = 1.0 - blade_insertion_percent/100.
"""

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass, field, asdict

# Flat imports (materials.py, geometry.py, ... live in this directory)
_MODEL_DIR = pathlib.Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

import openmc

# Local fallback only — the OPENMC_CROSS_SECTIONS env var takes precedence
# (set it in the Slurm submission script on the cluster).
_LOCAL_CROSS_SECTIONS = '/home/tmccoy/nuclear-data/endfb-viii.0-hdf5/cross_sections.xml'


# =============================================================================
# CONFIG
# =============================================================================

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class CoreConfig:
    """All run knobs in one place. CLI overrides via main()."""
    # Blade position — intuitive sense: 0 = fully withdrawn, 100 = fully
    # inserted. See module docstring for the conversion to withdrawal fraction.
    blade_insertion_percent: float = 100.0

    # Eigenvalue run statistics
    particles: int = 50000
    batches:   int = 200
    inactive:  int = 50
    seed:      int | None = None          # None → OpenMC default (1)
    
    # Fission Q-recoverable source for depletion power normalization.
    # 'chain' = values in the depletion chain (OpenMC default, used by jobs
    # 15117/15196/15348). 'adder' = ADDER/MCNP empirical fit, see
    # adder_fission_q(). Set explicitly with --fission-q.
    fission_q_mode: str = 'chain'

    # Depletion material zoning — segment the fuel meat on an (x, z) grid into
    # PER-PLATE, per-zone depletable materials: 614 plates x N_X_ZONES x
    # N_AXIAL_ZONES = 12,280 materials at the 2 x 10 default, filling 12,280
    # meat cells. One material per cell, 1:1 — no sharing between plates.
    #
    # SUPERSEDED, kept because the number 560 appears in older notes and runs:
    # this was once 28 elements x N_X x N_Z = 560 materials shared across the
    # plates of an element. That scheme was replaced on 2026-08-12 when Kyle
    # confirmed the reference differentiates per plate [MCNP]. A 560 in any
    # log or document predates that and describes a different model.
    # OFF by default — the Phase One fresh-core cross-validation against the
    # reference MCNP model must keep seeing the unchanged model. Turning it on
    # is not free at transport time; see the COST note in materials.py.
    depletion_zoning: bool = False

    # Paths
    #
    # ABSOLUTE, resolved from the REPOSITORY ROOT — not the cwd. This was a
    # relative 'run_results/core_run' until 2026-08-03, which meant a bare run
    # wrote into whatever directory it happened to start in. On 2026-07-31 that
    # silently overwrote the model.xml and summary.h5 of the archived production
    # run; see model/run_results/core_run/README.md. On a cluster, where the
    # submit script's cwd is rarely the repo, the relative form is worse still.
    output_dir: str = str(_REPO_ROOT / 'run_results' / 'core_run')
    cross_sections: str | None = None     # None → env var, then local fallback

    # Writing into a directory that already holds run output is refused
    # unless this is set. See _prepare_output_dir().
    overwrite_output: bool = False

    # Temperature treatment (materials without an explicit temperature
    # evaluate at 'default'; flux-trap water sets its own 316.8 K).
    temperature: dict = field(
        default_factory=lambda: {'method': 'interpolation', 'default': 294.0})

    # ── Depletion ────────────────────────────────────────────────────────────
    # These fields are LIVE: run_depletion() reads chain_file, power_w,
    # depletion_timesteps, depletion_integrator and solver. They carry the
    # ADDER-side configuration Kyle confirmed, next to the code that consumes
    # it rather than in a chat log. `substeps` remains a record only — see
    # below.
    #
    # CHAIN FILE — the intent is an ENDF/B-VIII.0 depletion chain MATCHED to
    # the ENDF/B-VIII.0 continuous-energy cross sections this model already
    # uses (see _LOCAL_CROSS_SECTIONS). None exists on this workstation; the
    # cluster path is _CHAIN_DIR below. Leave None to take the default there,
    # or set an explicit path. resolve_chain_file() decides and refuses the
    # unsuitable ones by name.
    # [MCNP/ADDER — Kyle 2026-08-12]
    chain_file: str | None = None
    power_w: float = 10.0e6                # 10 MW nominal core power
    depletion_timesteps: list = field(default_factory=list)   # e.g. [(30, 'd'), ...]

    # INTEGRATOR — ADDER uses a CE/CM predictor-corrector scheme. The OpenMC
    # equivalent is openmc.deplete.CECMIntegrator (present in the installed
    # 0.15.3). Was 'predictor' as a placeholder; 'cecm' is the confirmed match.
    # [MCNP/ADDER — Kyle 2026-08-12]
    depletion_integrator: str = 'cecm'

    # MATRIX EXPONENTIAL SOLVER — ADDER uses 48th-order CRAM. This is ALSO
    # OpenMC's default (openmc.deplete.abc.Integrator has solver='cram48',
    # 48th-order IPF CRAM), so this is a MATCH, NOT AN OVERRIDE. Set explicitly
    # anyway so the agreement is on the record rather than incidental.
    # [MCNP/ADDER — Kyle 2026-08-12]
    solver: str = 'cram48'

    # DEPLETION SUBSTEPS — ADDER uses 4. TWO SEPARATE PROBLEMS, both open:
    #
    # 1. CANNOT BE APPLIED ON THIS BUILD. OpenMC's `substeps` parameter does not
    #    exist in the installed 0.15.3 — verified: the string appears in no file
    #    in the installed package, and Integrator.__init__ takes only
    #    (operator, timesteps, power, power_density, source_rates,
    #    timestep_units, solver, continue_timesteps). Upstream, substeps was
    #    added in 0.16.0 FOR CECMIntegrator, which is the integrator we use.
    #    (It landed earlier, in 0.15.4, for LEQIIntegrator/SILEQIIntegrator —
    #    noted so the bare version number is not miscopied onto CECM.) Either
    #    way 0.15.3 predates both. Matching ADDER here requires an upgrade.
    #
    # 2. IT IS NOT ESTABLISHED THAT THE TWO "SUBSTEPS" ARE THE SAME OPERATION.
    #    OpenMC's subdivides the Bateman/CRAM solve interval into identical
    #    sub-intervals (reusing LU factorizations) with NO ADDITIONAL TRANSPORT.
    #    Whether ADDER's substep does the same, or re-solves transport, is
    #    UNCONFIRMED. If they differ, setting 4 on both sides is a false match
    #    and would look like agreement while comparing different schemes.
    #    [ASSUMED-EQUIVALENT — needs Kyle]
    substeps: int = 4


def resolve_cross_sections(cfg: CoreConfig) -> str:
    """Prefer OPENMC_CROSS_SECTIONS, then cfg override, then local fallback."""
    env = os.environ.get('OPENMC_CROSS_SECTIONS')
    if env:
        source, path = 'OPENMC_CROSS_SECTIONS env var', env
    elif cfg.cross_sections:
        source, path = 'CoreConfig.cross_sections', cfg.cross_sections
    else:
        source, path = 'local fallback path', _LOCAL_CROSS_SECTIONS
    print(f"[core] cross_sections from {source}: {path}")
    openmc.config['cross_sections'] = path
    return path


# =============================================================================
# MODEL ASSEMBLY
# =============================================================================

def build_model(cfg: CoreConfig) -> openmc.Model:
    """Assemble materials + geometry + settings + tallies into one Model."""
    resolve_cross_sections(cfg)

    # Run provenance is PRIMED HERE, AT RUN START — deliberately, and this must
    # not drift to the point where results are written. run_provenance() caches
    # on first call, so whichever call comes first fixes the recorded state. A
    # long cluster run that captured its tree state at result-write would record
    # the working tree hours after the run began, and a commit landing mid-run
    # would stamp the results with a SHA that never produced them. Capturing at
    # build time means the recorded SHA is the code that built the model.
    from settings import run_provenance, format_provenance
    run_provenance()          # prime the cache before anything else runs
    print("Run provenance (captured at model build):")
    print(format_provenance())

    # CRITICAL direction conversion — geometry uses WITHDRAWAL fraction
    # (f=0 fully inserted, f=1 fully withdrawn); cfg uses INSERTION percent
    # (0 withdrawn, 100 inserted). These are opposite senses:
    if not 0.0 <= cfg.blade_insertion_percent <= 100.0:
        raise ValueError(
            f"blade_insertion_percent must be in [0, 100], "
            f"got {cfg.blade_insertion_percent}")
    f_withdrawal = 1.0 - cfg.blade_insertion_percent / 100.0

    from materials import materials
    from geometry import build_core_geometry
    from settings import settings
    from tallies import build_tallies

    geometry = build_core_geometry(withdrawn_fraction=f_withdrawal,
                                   depletion_zoning=cfg.depletion_zoning)

    # Zoned fuel materials are created during the geometry build, so they are
    # collected after it. With zoning off this is the untouched `materials`
    # object itself — the exported model is unchanged.
    if cfg.depletion_zoning:
        from materials import fuel as base_fuel, get_zoned_fuels
        zoned_fuels = get_zoned_fuels()

        # When zoned, every meat cell is filled by a zoned clone and the base
        # fuel fills nothing. Verify that rather than assume it: a base-fuel
        # cell surviving here would mean the axial split missed a meat cell.
        stray = [c.name for c in geometry.get_all_cells().values()
                 if c.fill is base_fuel]
        if stray:
            raise RuntimeError(
                f"depletion zoning: {len(stray)} cell(s) still filled with the "
                f"base fuel material — the axial split missed them: "
                f"{stray[:5]}{' ...' if len(stray) > 5 else ''}")

        # Drop the base fuel from the exported set. It survives in materials.py
        # as the clone() source for make_zoned_fuel, but it fills no cell here,
        # and OpenMC auto-flags any actinide-bearing material depletable — so
        # exporting it would hand a future depletion solve an unused depletable
        # material with volume=None, which misnormalizes quietly rather than
        # failing. Zoning-off path keeps the collection exactly as it was.
        model_materials = openmc.Materials(
            [m for m in materials if m is not base_fuel] + zoned_fuels)
        print(f"[core] depletion zoning ON: {len(zoned_fuels)} zoned fuel "
              f"materials (unused base fuel excluded from the export)")
    else:
        model_materials = materials

    settings.particles   = cfg.particles
    settings.batches     = cfg.batches
    settings.inactive    = cfg.inactive
    settings.temperature = cfg.temperature
    if cfg.seed is not None:
        settings.seed = cfg.seed

    return openmc.Model(
        geometry=geometry,
        materials=model_materials,
        settings=settings,
        tallies=build_tallies(),
    )


# =============================================================================
# EIGENVALUE RUN
# =============================================================================

# Files a run writes that would silently misdescribe a previous run if
# overwritten. model.xml and summary.h5 are the geometry/material description;
# statepoint*.h5 and tallies.out are the results.
_RUN_ARTIFACTS = ('model.xml', 'summary.h5', 'tallies.out')


def _prepare_output_dir(cfg: CoreConfig) -> pathlib.Path:
    """Create the output directory, refusing to clobber an existing run.

    Overwriting only SOME files of a previous run is the dangerous case: it
    leaves results beside a geometry description that does not match them, and
    nothing errors when the two are later read together. That is exactly what
    happened to model/run_results/core_run/ on 2026-07-31 — a build_model()
    smoke test replaced model.xml and summary.h5 next to a statepoint from a
    geometry nine commits older.
    """
    out = pathlib.Path(cfg.output_dir).resolve()

    existing = [f.name for f in out.glob('*')
                if f.name in _RUN_ARTIFACTS or f.name.startswith('statepoint')]
    if existing and not cfg.overwrite_output:
        raise FileExistsError(
            f"refusing to write into a directory that already holds run output:\n"
            f"    {out}\n"
            f"    found: {', '.join(sorted(existing))}\n"
            f"Overwriting only some of these leaves results beside a geometry "
            f"description that no longer matches them, and the mismatch is "
            f"silent. Choose a different --output-dir, or pass --overwrite-output "
            f"if you genuinely mean to replace this run.")

    out.mkdir(parents=True, exist_ok=True)
    return out


def run_eigenvalue(cfg: CoreConfig):
    """Build and run one eigenvalue calculation; return keff (ufloat)."""
    model = build_model(cfg)

    out = _prepare_output_dir(cfg)

    print(f"[core] insertion={cfg.blade_insertion_percent:.1f}% "
          f"(withdrawal f={1.0 - cfg.blade_insertion_percent / 100.0:.3f})  "
          f"{cfg.particles} p/batch, {cfg.batches} batches "
          f"({cfg.inactive} inactive)  ->  {out}")

    sp_path = model.run(cwd=str(out))
    with openmc.StatePoint(sp_path) as sp:
        keff = sp.keff
    print(f"[core] keff = {keff.nominal_value:.5f} +/- {keff.std_dev:.5f}")
    return keff


# =============================================================================
# DEPLETION
# =============================================================================

# Cluster depletion-data directory. Nothing here exists on the workstation;
# resolve_chain_file() raises with this path in the message when it is missing,
# which is the expected local outcome.
_CHAIN_DIR = '/beegfs2/data/EP/openmc/data/depletion'

# ONLY THIS ONE IS CONFIRMED BY NAME. The 71/80 variants below are INFERRED
# from the same naming pattern and have not been listed on the cluster — treat
# a miss on them as "the guess was wrong", not "the file was deleted".
_CHAIN_DEFAULT = 'endfb81_chain.pwr.xml'      # [CONFIRMED by name]
_CHAIN_INFERRED = ('endfb71_chain.pwr.xml',   # [INFERRED from the pattern]
                   'endfb80_chain.pwr.xml')   # [INFERRED from the pattern]


# NAMED DEPLETION SCHEDULES. Selected explicitly with --schedule; there is
# still no default, and run_depletion() still raises when nothing is set.
# Naming a schedule at submit time is a choice on the record, not a value
# inherited from a dataclass default.

_SCHEDULES = {
    # kyle-400d: full rundown with blades fully WITHDRAWN (pair with
    # --insertion 0), per Kyle Anderson 2026-08-31 — deplete until the reactor
    # can no longer run, NO critical blade search, run until k-eff < 1.
    # Fibonacci ramp maxing at 21 d: 1,1,2,3,5,8,13,21 = 54 d in 8 steps, then
    # 17 x 21 d = 357 d, ending at 411 days over 25 steps.
    # [ADDER/MCNP — Kyle 2026-08-31]
    'kyle-400d': [(d, 'd') for d in
                  [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0] + [21.0] * 17],
                  
    # fine-318end: kyle-400d step lengths truncated to end at 318 d, matching
    # valerio-318d / coarse-318end for final-state and isotopic comparison.
    # A 12 d step is inserted before the 13 d step so the 21 d cap lands on
    # 318: 1,1,2,3,5,8,12,13,21 = 66 d in 9 steps, then 12 x 21 d = 252 d,
    # 318 days over 21 steps. Grid matches kyle-400d only through t = 20 d.
    # [1x step-size schedule — paired with coarse-318end for ADDER comparison]
    'fine-318end': [(d, 'd') for d in
                  [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 13.0, 21.0] + [21.0] * 12],
    # valerio-318d: doubled step lengths against kyle-400d, to measure CE/CM
    # sensitivity to step size. Ramp 2,2,4,6,10,16,26,42 = 108 d in 8 steps,
    # then 5 x 42 d = 210 d, ending at 318 days over 13 steps.
    # Shares only t = 0, 2 and 4 d with kyle-400d, so the two are compared on
    # the interpolated k-eff = 1 crossing, not point by point.
    # [SUGGESTED — Valerio 2026-09-01, not an ADDER-matched schedule]
    'valerio-318d': [(d, 'd') for d in
                     [2.0, 2.0, 4.0, 6.0, 10.0, 16.0, 26.0, 42.0] + [42.0] * 5],
                     
    # coarse-318end: identical to valerio-318d; separate name for the
    # ADDER-MCNP coarse-schedule comparison (pairs with fine-318end).
    'coarse-318end': [(d, 'd') for d in
                     [2.0, 2.0, 4.0, 6.0, 10.0, 16.0, 26.0, 42.0] + [42.0] * 5],
    # valerio-300d: quadrupled step lengths against kyle-400d (doubled against
    # valerio-318d), extending the CE/CM step-size sensitivity study to a third
    # point. Ramp 4,4,8,12,20,32,52,84 = 216 d in 8 steps, then one more 84 d
    # step, ending at 300 days over 9 steps (job 15348).
    # Shares only t = 0 and 4 d with kyle-400d, and t = 0, 4 and 8 d with
    # valerio-318d, so all three are compared on the interpolated crossing.
    # [SUGGESTED — Valerio 2026-09-14, not an ADDER-matched schedule]
    'valerio-300d': [(d, 'd') for d in
                     [4.0, 4.0, 8.0, 12.0, 20.0, 32.0, 52.0, 84.0,
                      84.0]],
}

def adder_fission_q(chain_path: str) -> dict:
    """ADDER/MCNP fission Q-recoverable in eV for every chain nuclide with a
    fission reaction: Q = 1.29927e-3 * Z^2 * sqrt(A) + 33.12 MeV.
    [ADDER convention — matches ADDER-MCNP power normalization]"""
    import xml.etree.ElementTree as ET
    q = {}
    for nuc in ET.parse(chain_path).getroot():
        if any(r.get('type') == 'fission' for r in nuc.findall('reaction')):
            name = nuc.get('name')
            Z, A, _ = openmc.data.zam(name)
            q[name] = (1.29927e-3 * Z * Z * A ** 0.5 + 33.12) * 1.0e6
    return q


def resolve_chain_file(cfg: CoreConfig) -> str:
    """Pick the depletion chain file, refusing the ones that would mislead.

    TWO CLASSES ARE REFUSED OUTRIGHT rather than warned about, because both
    produce a run that completes and reports plausible numbers:

      simplified_*  — a truncated nuclide set. It runs, it is fast, and its
                      answers are wrong in a direction that looks like physics.
                      Nothing downstream can tell that the chain was reduced.
      *.fast        — a fast-spectrum chain. This is a thermal, light-water
                      research reactor; a fast chain's branching ratios and
                      fission yields do not apply, and again nothing errors.

    An explicit cfg.chain_file is honoured, subject to the same two refusals —
    naming the file by hand is not a reason to accept a wrong one.
    """
    if cfg.chain_file is not None:
        path = cfg.chain_file
        source = 'CoreConfig.chain_file'
    else:
        path = os.path.join(_CHAIN_DIR, _CHAIN_DEFAULT)
        source = f'default ({_CHAIN_DEFAULT})'

    name = os.path.basename(path)
    if name.startswith('simplified_'):
        raise ValueError(
            f"refusing the simplified chain {name!r}.\n"
            f"    It carries a truncated nuclide set: the run will COMPLETE "
            f"and report plausible-looking numbers that are wrong, and nothing "
            f"downstream can detect the truncation. Use the full chain.")
    if name.endswith('.fast') or name.endswith('.fast.xml'):
        raise ValueError(
            f"refusing the fast-spectrum chain {name!r}.\n"
            f"    This is a thermal light-water research reactor. A fast "
            f"chain's branching ratios and fission yields do not apply here, "
            f"and the run will complete anyway. Use the thermal (pwr) chain.")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"depletion chain file not found:\n"
            f"    {path}    (from {source})\n"
            f"Depletion cannot run without one. On the cluster these live in "
            f"{_CHAIN_DIR}; only {_CHAIN_DEFAULT!r} is confirmed to exist by "
            f"name — {', '.join(repr(c) for c in _CHAIN_INFERRED)} are inferred "
            f"from the naming pattern and may not be there. On this "
            f"workstation no chain file exists at all; set "
            f"CoreConfig.chain_file to an explicit path.")

    print(f"[core] chain file from {source}: {path}")
    return path


def run_depletion(cfg: CoreConfig):
    """Run a depletion sequence from the CoreConfig settings.

    NO DEFAULT STEP SCHEDULE. cfg.depletion_timesteps must be set explicitly;
    an empty list raises. A plausible-looking default here would be the worst
    kind of guess — the run would succeed and produce a burnup history that
    nobody chose, and the schedule is exactly what has to match ADDER-MCNP for
    the comparison to mean anything. That structure is still unconfirmed.

    Output goes to cfg.output_dir, prepared by _prepare_output_dir() the same
    way the eigenvalue path prepares it, and handed to the operator as
    op.output_dir. That is where depletion_results.h5 lands — the artifact that
    matters, since it is written incrementally and is the only record of
    completed steps. Still NO os.chdir(): the directory is passed, never
    entered, so a long run cannot be relocated out from under itself.
    """
    if not cfg.depletion_timesteps:
        raise ValueError(
            "cfg.depletion_timesteps is empty and there is no default.\n"
            "    Set it explicitly, e.g. [(30, 'd')] * 17. The step structure "
            "is UNCONFIRMED against the ADDER-MCNP sequence this run is meant "
            "to be compared with — step count, lengths and units all have to "
            "match on both sides, and a default chosen here would silently "
            "become that answer.")

    import openmc.deplete

    chain = resolve_chain_file(cfg)
    out = _prepare_output_dir(cfg)
    model = build_model(cfg)

    # diff_burnable_mats is NOT set: the meat is already split into distinct
    # per-plate, per-zone materials by the zoning layer (see depletion_zoning),
    # so asking OpenMC to differentiate again would re-split materials that are
    # already unique and multiply the count for nothing.
    print(f'fission Q source: {cfg.fission_q_mode}')
    if cfg.fission_q_mode == 'adder':
        op = openmc.deplete.CoupledOperator(
            model, chain_file=chain, fission_q=adder_fission_q(chain))
    else:
        op = openmc.deplete.CoupledOperator(model, chain_file=chain)
    op.output_dir = out
    integrator_cls = {
        'predictor': openmc.deplete.PredictorIntegrator,
        'cecm':      openmc.deplete.CECMIntegrator,
    }[cfg.depletion_integrator]

    # substeps is deliberately NOT passed. It does not exist on the installed
    # OpenMC 0.15.3 (Integrator.__init__ takes no such parameter — it arrived
    # in 0.16.0 for CECM), so passing it would be a TypeError, and its
    # equivalence to ADDER's substep is unconfirmed regardless. cfg.substeps
    # stays a record of the ADDER-side setting, not an input.
    integrator = integrator_cls(op, cfg.depletion_timesteps,
                                power=cfg.power_w, solver=cfg.solver)

    print(f"[core] depletion: {cfg.depletion_integrator} / {cfg.solver}, "
          f"{len(cfg.depletion_timesteps)} steps, power {cfg.power_w:.4g} W")
    print(f"[core] writing depletion_results.h5 into {out}")
    integrator.integrate()


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description='TECDOC-643 10 MW LEU core — central OpenMC driver')
    p.add_argument('--insertion', type=float, default=None, metavar='PCT',
                   help='blade insertion %% (0 = fully withdrawn, '
                        '100 = fully inserted)')
    p.add_argument('--particles', type=int, default=None)
    p.add_argument('--batches', type=int, default=None)
    p.add_argument('--inactive', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--overwrite-output', action='store_true',
                   help='allow writing into a directory that already holds '
                        'run output (refused by default)')
    p.add_argument('--depletion-zoning', action='store_true',
                   help='split the fuel meat into per-plate, per-zone '
                        'depletable materials (12,280 at the 2 x 10 default). '
                        'Structural only — configures no depletion. Default '
                        'off.')
    p.add_argument('--deplete', action='store_true',
                   help='run a depletion sequence instead of an eigenvalue '
                        'calculation. Requires --timesteps and a chain file.')
    p.add_argument('--chain-file', type=str, default=None,
                   help=f'depletion chain file (default: {_CHAIN_DEFAULT} in '
                        f'{_CHAIN_DIR}). simplified_* and .fast are refused.')
    p.add_argument('--power', type=float, default=None, metavar='W',
                   help='core power in watts for depletion (default 10 MW)')
    p.add_argument('--timesteps', type=int, default=None, metavar='N',
                   help='number of depletion steps; use with --step-days. '
                        'There is no default schedule.')
    p.add_argument('--step-days', type=float, default=None, metavar='D',
                   help='length of each depletion step in days')
    p.add_argument('--fission-q', type=str, default=None,
                   choices=['chain', 'adder'],
                   help="fission Q-recoverable: 'chain' (default) or 'adder' "
                        "(ADDER/MCNP 1.29927e-3*Z^2*sqrt(A)+33.12 MeV)")
    p.add_argument('--schedule', type=str, default=None,
                   choices=sorted(_SCHEDULES),
                   help='named depletion schedule (see _SCHEDULES). Mutually '
                        'exclusive with --timesteps/--step-days.')
    args = p.parse_args(argv)

    cfg = CoreConfig()
    if args.insertion is not None:
        cfg.blade_insertion_percent = args.insertion
    if args.particles is not None:
        cfg.particles = args.particles
    if args.batches is not None:
        cfg.batches = args.batches
    if args.inactive is not None:
        cfg.inactive = args.inactive
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.depletion_zoning:
        cfg.depletion_zoning = True
    if args.overwrite_output:
        cfg.overwrite_output = True
    if args.chain_file is not None:
        cfg.chain_file = args.chain_file
    if args.power is not None:
        cfg.power_w = args.power
    if args.fission_q is not None:
        cfg.fission_q_mode = args.fission_q
    if args.schedule is not None:
        if args.timesteps is not None or args.step_days is not None:
            p.error('--schedule cannot be combined with --timesteps/--step-days')
        cfg.depletion_timesteps = list(_SCHEDULES[args.schedule])
    if args.timesteps is not None or args.step_days is not None:
        if args.timesteps is None or args.step_days is None:
            p.error('--timesteps and --step-days must be given together')
        cfg.depletion_timesteps = [(args.step_days, 'd')] * args.timesteps

    # What prints depends on what the run actually reads.
    #
    # On an EIGENVALUE run the depletion fields are inert, and printing them
    # would imply this run used them. On a DEPLETION run all of them are live
    # except substeps, so hiding them would misreport the run's own settings —
    # they were hidden on the grounds that "nothing reads them", which stopped
    # being true when run_depletion() was wired up.
    #
    # substeps stays hidden in BOTH cases: it is not passed to the integrator
    # (absent on 0.15.3, equivalence to ADDER's substep unconfirmed) and it
    # would be the one printed value the run did not honour.
    _depletion_fields = {'chain_file', 'power_w', 'depletion_timesteps',
                         'depletion_integrator', 'solver'}
    _hidden = {'substeps'} | (set() if args.deplete else _depletion_fields)
    print(f"[core] config: { {k: v for k, v in asdict(cfg).items() if k not in _hidden} }")

    if args.deplete:
        run_depletion(cfg)
    else:
        run_eigenvalue(cfg)


if __name__ == '__main__':
    main()
