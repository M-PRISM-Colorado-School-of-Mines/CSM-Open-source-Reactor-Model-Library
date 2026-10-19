"""
test_reference_materials.py
---------------------------
Checks every base material in model/materials.py against
docs/reference_materials.csv (from the reference MCNP model).

For each material in the CSV:
  * every nuclide atom density matches to 1e-6 relative. "C (natural)" is
    compared against OpenMC's C12 + C13 (or C0 if USE_NATURAL_CARBON);
  * the material carries no nuclide the CSV does not list;
  * thermal scattering matches: none / Al-27 / H in H2O / graphite;
  * temperature matches (K);
  * every material in materials.materials appears in the CSV, and vice versa.

Runs under pytest, or directly: python tests/test_reference_materials.py
(exits non-zero on failure).
"""

import csv
import io
import os
import sys
from contextlib import redirect_stdout
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))

with redirect_stdout(io.StringIO()):
    import materials as M

CSV_PATH = os.path.join(REPO_ROOT, 'docs', 'reference_materials.csv')
REL_TOL = 1e-6

# CSV thermal_scattering label -> the OpenMC S(a,b) table it requires.
SAB_TABLE = {
    'none':     None,
    'Al-27':    'c_Al27',
    'H in H2O': 'c_H_in_H2O',
    'graphite': 'c_Graphite',
}

NATURAL_C = 'C (natural)'


def load_reference():
    """material name -> {'temperature': float, 'sab': str, 'nuclides': {nuc: N}}"""
    ref = defaultdict(lambda: {'nuclides': {}})
    with open(CSV_PATH, newline='') as fh:
        for row in csv.DictReader(fh):
            m = ref[row['material']]
            t = float(row['temperature_K'])
            assert m.setdefault('temperature', t) == t, \
                f"{row['material']}: inconsistent temperature rows in the CSV"
            assert m.setdefault('sab', row['thermal_scattering']) == row['thermal_scattering'], \
                f"{row['material']}: inconsistent thermal_scattering rows in the CSV"
            assert row['nuclide'] not in m['nuclides'], \
                f"{row['material']}: duplicate nuclide {row['nuclide']}"
            m['nuclides'][row['nuclide']] = float(row['atom_density_per_b_cm'])
    return dict(ref)


REFERENCE = load_reference()

# CSV material name -> OpenMC material object. The CSV names ARE the
# materials.py variable names.
MODEL = {name: getattr(M, name) for name in REFERENCE}


def _modelled_densities(mat):
    """OpenMC atom densities, with C12 + C13 (or C0) folded into 'C (natural)'."""
    ad = dict(mat.get_nuclide_atom_densities())
    carbon = sum(ad.pop(n) for n in ('C0', 'C12', 'C13') if n in ad)
    if carbon:
        ad[NATURAL_C] = carbon
    return ad


def test_every_model_material_is_in_the_reference():
    names = {m.name for m in M.materials}
    mapped = {m.name for m in MODEL.values()}
    assert names == mapped, (
        f"materials.materials and the CSV disagree: "
        f"unmapped model materials {sorted(names - mapped)}")


def test_nuclide_atom_densities():
    bad = []
    for name, ref in REFERENCE.items():
        got = _modelled_densities(MODEL[name])
        extra = set(got) - set(ref['nuclides'])
        if extra:
            bad.append(f"{name}: nuclides not in the reference {sorted(extra)}")
        for nuc, want in ref['nuclides'].items():
            have = got.get(nuc)
            if have is None:
                bad.append(f"{name}: {nuc} missing")
                continue
            rel = abs(have - want) / want
            if rel > REL_TOL:
                bad.append(f"{name}: {nuc} {have:.7e} vs {want:.7e} (rel {rel:.2e})")
    assert not bad, '\n'.join(bad)


def test_thermal_scattering():
    bad = []
    for name, ref in REFERENCE.items():
        have = sorted(s[0] for s in MODEL[name]._sab)
        want = [] if SAB_TABLE[ref['sab']] is None else [SAB_TABLE[ref['sab']]]
        if have != want:
            bad.append(f"{name}: S(a,b) {have}, reference {ref['sab']!r} -> {want}")
    assert not bad, '\n'.join(bad)


def test_temperature():
    bad = [f"{name}: model {MODEL[name].temperature} K, reference "
           f"{ref['temperature']} K"
           for name, ref in REFERENCE.items()
           if MODEL[name].temperature is None
           or abs(MODEL[name].temperature - ref['temperature']) > 1e-9]
    assert not bad, '\n'.join(bad)


if __name__ == '__main__':
    failures = 0
    for test in (test_every_model_material_is_in_the_reference,
                 test_nuclide_atom_densities,
                 test_thermal_scattering,
                 test_temperature):
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}\n{exc}")
    n_nuc = sum(len(r['nuclides']) for r in REFERENCE.values())
    print(f"{len(REFERENCE)} materials, {n_nuc} reference nuclide rows")
    sys.exit(1 if failures else 0)
