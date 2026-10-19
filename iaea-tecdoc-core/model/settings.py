"""
settings.py
-----------
Simulation settings for the IAEA TECDOC-643 Appendix A-2
Generic 10 MW LEU Research Reactor Core (Argonne design).

Reference:
    IAEA-TECDOC-643, "Research Reactor Core Conversion Guidebook,
    Volume 2: Analysis (Appendices A-F)," IAEA, Vienna, 1992.
    Appendix A-2: Generic 10 MW Reactor — Argonne National Laboratory.

About this file:
    This file controls HOW OpenMC runs the simulation — not what
    the reactor looks like (that's geometry.py) or what it's made
    of (that's materials.py). Think of it as the "run instructions".

Key concepts:
    - Criticality (k-eigenvalue) calculation: finds keff
    - Particles: the neutrons OpenMC simulates
    - Batches: groups of neutrons simulated together
    - Inactive batches: early batches discarded while the fission
      source distribution is still settling (not yet converged)
    - Active batches: batches used for actual statistics/results
"""

import datetime
import os
import subprocess

import openmc
import numpy as np

# =============================================================================
# RUN PROVENANCE
#
# The OpenMC version is an EXPERIMENTAL VARIABLE in a cross-validation study,
# not an implementation detail. Phase 1 geometry was verified on 0.15.3; the
# production statepoint in model/run_results/ records 0.15.0. Nothing has
# confirmed the two agree, and the difference was only discoverable after the
# fact by reading version attributes out of a statepoint.
#
# This block RECORDS; it does not enforce. A hard version assert would turn a
# cluster upgrade into a build failure on a machine nobody here administers,
# and the failure mode that matters — two versions producing different answers —
# is a comparison defect, not a build defect. The one thing worth failing on is
# the cross-section library (see assert_cross_section_library below).
#
# Computed lazily and cached: at import time this would shell out to git twice
# on every `import settings`, and settings is imported by geometry, tallies and
# every driver.
# =============================================================================

_PROVENANCE_CACHE = None


def _git_state():
    """'<short-sha>' or '<short-sha>-dirty', or 'unknown' if git is unavailable.

    A bare SHA taken from a modified working tree does not describe the code
    that ran, and is worse than recording nothing because it looks
    authoritative. The dirty flag is the point of this function.

    Never raises. Provenance capture must not be able to kill a production run:
    git may be absent, the cwd may sit outside the work tree, or the subprocess
    may time out on a cluster filesystem. It degrades to 'unknown'.
    """
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip())
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def run_provenance():
    """Everything needed to reproduce a run, as a dict. Cached after first call."""
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None:
        _PROVENANCE_CACHE = {
            'openmc_version': openmc.__version__,
            'cross_sections': os.environ.get('OPENMC_CROSS_SECTIONS', 'unset'),
            'git': _git_state(),
            # datetime.utcnow() is deprecated in Python 3.12+ and would emit a
            # DeprecationWarning into every run log.
            'utc': datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec='seconds'),
        }
    return dict(_PROVENANCE_CACHE)


def format_provenance(prefix='  '):
    """Provenance as aligned lines, for printing at the head of a run log."""
    p = run_provenance()
    width = max(len(k) for k in p)
    return '\n'.join(f'{prefix}{k:<{width}}  {v}' for k, v in p.items())


def assert_cross_section_library(expect):
    """Fail loudly if OPENMC_CROSS_SECTIONS does not contain `expect`.

    NARROW BY DESIGN — for matched-library runs only, and not called anywhere
    yet. Running the ENDF/B-VII.0 baseline against VIII.0 by accident is the one
    error that silently invalidates an entire cross-validation comparison: it
    produces a plausible k-eff and no warning. run_vii_mat.py already keys its
    USE_NATURAL_CARBON switch off this same variable.

        assert_cross_section_library('endfb70')   # in the VII.0 driver
    """
    actual = os.environ.get('OPENMC_CROSS_SECTIONS', '')
    if expect not in actual:
        raise RuntimeError(
            f"cross-section library mismatch: expected a path containing "
            f"'{expect}', got OPENMC_CROSS_SECTIONS={actual!r}. Refusing to "
            f"run — a matched-library comparison against the wrong library is "
            f"silently wrong, not loudly wrong.")

# =============================================================================
# SIMULATION MODE
# We are running a k-eigenvalue (criticality) calculation.
# This finds keff — the effective neutron multiplication factor.
# keff > 1.0 = supercritical (reaction growing)
# keff = 1.0 = critical (steady state, what we want for 10 MW operation)
# keff < 1.0 = subcritical (reaction dying out)
# =============================================================================

settings = openmc.Settings()
settings.run_mode = 'eigenvalue'   # criticality calculation

# =============================================================================
# PARTICLE STATISTICS
#
# particles: number of neutrons simulated per batch
# batches:   total number of batches to run
# inactive:  number of batches to discard at the start
#
# Rule of thumb:
#   - inactive batches should be ~40-50% of total batches
#   - more particles = more accurate but slower
#   - start small for testing, scale up for final results
#
# For a first test run (fast, less accurate):
#   particles=1000, batches=50, inactive=20
# For a production run (slow, more accurate):
#   particles=10000, batches=200, inactive=50
# =============================================================================

settings.particles  = 50000    # neutrons per batch (increase for production)
settings.batches    = 150      # total batches
settings.inactive   = 50      # discard first 50 batches (source convergence)

# =============================================================================
# INITIAL FISSION SOURCE
#
# OpenMC needs a starting guess for where fission neutrons come from.
# We use a uniform spatial distribution across the active core volume.
#
# The LATTICE is 6 (x) x 7 (y) = 42 positions: the 5 x 6 = 30 core positions
# plus 12 graphite reflector positions. The FUELLED sub-block — the 28
# standard and control elements, excluding the graphite reflector rows at the
# top and bottom — is 6 columns wide and 5 rows tall, spanning +/-23.1 cm in x
# and +/-20.25 cm in y. The box below covers it with margin in y:
#   x: -3*PITCH_X to +3*PITCH_X  = +/-23.10 cm  (the full 6 fuel columns)
#   y: -3*PITCH_Y to +3*PITCH_Y  = +/-24.30 cm  (covers the 5 fuel rows)
#   z: -HALF_Z to +HALF_Z        = +/-30.00 cm  (active fuel meat height)
#
# Points landing in graphite, flux traps or water gaps are discarded by the
# fissionable constraint below, so over-coverage is harmless.
#
# OpenMC will refine this distribution over the inactive batches
# until it converges to the true fission source shape.
# =============================================================================

# Bounds are IMPORTED from geometry.py, never restated. They were local
# PITCH_X/PITCH_Y = 7.7/8.1 and +/-30.0 literals until 2026-07-31; the audit
# flagged them as the last surviving duplication of geometry constants.
from geometry import PITCH_X, PITCH_Y, HALF_Z

source_box = openmc.stats.Box(
    lower_left  = (-3 * PITCH_X, -3 * PITCH_Y, -HALF_Z),
    upper_right = ( 3 * PITCH_X,  3 * PITCH_Y,  HALF_Z),
)

# 'constraints' replaces the deprecated Box(only_fissionable=True) in 0.15.0:
# source points are rejected unless they land in fissionable material.
settings.source = openmc.IndependentSource(
    space=source_box,
    constraints={'fissionable': True},
)

# =============================================================================
# OUTPUT OPTIONS
#
# Controls what OpenMC writes to disk after the run.
# summary.h5    — geometry and material summary (always useful)
# tallies.out   — tally results in plain text
# =============================================================================

settings.output = {
    'tallies': True,    # write tallies.out
    'summary': True,    # write summary.h5
}

# =============================================================================
# TEMPERATURE SETTINGS
#
# Cross sections are temperature dependent. 'interpolation' interpolates
# between library temperatures; 'default': 294.0 makes any material WITHOUT an
# explicit .temperature evaluate at the reference MCNP model's 294 K basis
# instead of OpenMC's built-in 293.6 K. (Flux-trap water sets its own 316.8 K
# explicitly.)
# =============================================================================

settings.temperature = {'method': 'interpolation', 'default': 294.0}

# =============================================================================
# EXPORT SETTINGS TO XML
# =============================================================================

if __name__ == '__main__':
    settings.export_to_xml()
    print("settings.xml written successfully.")
    print(f"\nSimulation summary:")
    print(f"  Run mode       : {settings.run_mode}")
    print(f"  Particles/batch: {settings.particles}")
    print(f"  Total batches  : {settings.batches}")
    print(f"  Inactive       : {settings.inactive}")
    print(f"  Active         : {settings.batches - settings.inactive}")
    print(f"  Temperature    : {settings.temperature['method']} K")
    print(f"\nRun provenance:")
    print(format_provenance())