# TODO

## Non-Linear Elements
- [ ] Upgrade diode to SPICE Level 1 model (RS, BV/IBV, CJ0/VJ/M, TT)
- [ ] Add BJT non-linear element (Ebers-Moll Level 1, 3-terminal)
- [ ] Add MOSFET non-linear element (Shichman-Hodges Level 1, 3-terminal)
- [ ] Upgrade BJT to Gummel-Poon model (4th substrate terminal, base resistance, charge storage)
- [ ] Upgrade MOSFET to Level 2/3 (4th bulk terminal, body effect GAMMA/PHI, subthreshold)
- [ ] Add JFET non-linear element (NJF/PJF)
- [ ] Advanced convergence aids (source stepping, Gmin stepping, continuation methods)
- [ ] User-configurable convergence parameters via netlist options (`.OPTIONS`)

## GPU Backends
- [ ] CUDA backend implementing `SolverBackend` + `NonlinearBackend`
- [ ] ROCm backend implementing `SolverBackend` + `NonlinearBackend`

## Analysis
- [ ] Transient analysis for discrete-time / switching power supply simulation

## Input Format
- [ ] Replace SPICE syntax with a more powerful input format
