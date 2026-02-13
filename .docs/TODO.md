# TODO

## Non-Linear Elements
- [ ] Upgrade diode to SPICE Level 1 model (RS, BV/IBV, CJ0/VJ/M, TT)
- [ ] Add BJT non-linear element (Ebers-Moll Level 1, 3-terminal)
- [ ] Add MOSFET non-linear element (Shichman-Hodges Level 1, 3-terminal)
- [ ] Upgrade BJT to Gummel-Poon model (4th substrate terminal, base resistance, charge storage)
- [ ] Upgrade MOSFET to Level 2/3 (4th bulk terminal, body effect GAMMA/PHI, subthreshold)
- [ ] Add JFET non-linear element (NJF/PJF)
- [ ] **Investigate BJT Newton-Raphson non-convergence** (see below)
- [ ] Advanced convergence aids (source stepping, Gmin stepping, continuation methods)
- [ ] User-configurable convergence parameters via netlist options (`.OPTIONS`)

## GPU Backends
- [ ] CUDA backend implementing `SolverBackend` + `NonlinearBackend`
- [ ] ROCm backend implementing `SolverBackend` + `NonlinearBackend`

## Analysis
- [x] Transient analysis (BE+TRAP with adaptive timestep)
- [x] ~~BUG: UIC zeros all state variables instead of computing consistent initial conditions~~

## BJT Non-Convergence Investigation

The NPN common-emitter circuit fails DC operating point analysis:

```
DC analysis error: Newton solver did not converge after 50 iterations (max residual: 5.28e-2)
```

**Failing circuit** (`tests/regression/circuits/npn_common_emitter.spice`):
```spice
V1 vcc 0 DC 5
V2 vb 0 DC 1.0
R1 vcc vc 1k
R2 vb base 100k
.MODEL Q2N2222 NPN(IS=1e-14 BF=200 BR=2 NF=1.0 NR=1.0)
Q1 vc base 0 Q2N2222
```

The equivalent NMOS circuit converges without issue, so the problem is specific to the BJT path.

**Things to investigate:**
- [ ] Verify BJT Ebers-Moll Jacobian entries against a reference
- [ ] Check voltage limiting for BJT junctions -- Vbe may jump too far per iteration
- [ ] Try source stepping (ramp V2 from 0V to 1V) as workaround
- [ ] Try Gmin stepping (small conductances to ground, gradually removed)
- [ ] Check initial guess -- starting all voltages at 0V may put the BJT in a bad region
- [ ] Compare Newton iteration trajectory against ngspice debug output (`set ngdebug`)
- [ ] Determine whether MOSFET converges because its equations are better conditioned or because the circuit topology is simpler

## Input Format
- [ ] Replace SPICE syntax with a more powerful input format
