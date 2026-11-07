# emfin-real-estate

Self-contained real estate module extracted from the full emfin_capstone model, for drop-in use in DimaZhu/emfin_capstone.

## Contents
- `emfin_capstone/toolbox/real_estate.py`: unsmoothing (Geltner, Fisher-Geltner-Webb), Kalman state-space model, ACF / Ljung-Box / ARCH-LM diagnostics, block bootstrap, synthetic simulator.
- `tests/test_real_estate.py`: 14 tests on synthetic data with known truth.

Dependencies: numpy, scipy (already in the capstone pyproject). No imports from other capstone modules.

## Drop in
```bash
git clone https://github.com/TijsdeBoer/emfin-real-estate.git /tmp/re
cp /tmp/re/emfin_capstone/toolbox/real_estate.py <capstone>/emfin_capstone/toolbox/
mkdir -p <capstone>/tests && cp /tmp/re/tests/test_real_estate.py <capstone>/tests/
cd <capstone> && pip install pytest && python -m pytest tests/test_real_estate.py
```

Returns are quarterly log returns. `moment_fit` is a crude starting point, not EM.
