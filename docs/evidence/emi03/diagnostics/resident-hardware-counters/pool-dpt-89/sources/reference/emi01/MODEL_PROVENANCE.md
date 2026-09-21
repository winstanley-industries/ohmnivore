# EMI-01 public SiC reference model provenance

Audit date: 2026-09-16. The frozen contract is
[ADR-002](../../docs/adr/ADR-002-inverter-emi-design-study.md).
Model selection and initial parser experiments preceded that contract; their temporary
outputs are exploratory compatibility work, **not retained switching qualification or runtime
evidence**. The reference harness subsequently repeats qualification using the frozen inputs.

## Selected public device and exact input identity

The selected device is Microchip **MSC040SMA120B**, a documented 1200 V, nominal 40 milliohm
SiC MOSFET in TO-247. Microchip's [product page](https://www.microchip.com/en-us/product/msc040sma120b-mosfet)
and [public SPICE download catalogue](https://www.microchip.com/en-us/software-library/sic-products-spice-files)
identify the device and simulation source. This is a vendor behavioral model with dynamic
gate and output capacitances, not an ideal switch or memoryless Level-1 substitution.

| Input | Frozen identity |
|---|---|
| Original archive | [1200V-SMA-SiC-MOSFET-SPICE-Models.zip](https://ww1.microchip.com/downloads/aemDocuments/documents/sic/ProductDocuments/BoardDesignFiles/1200V-SMA-SiC-MOSFET-SPICE-Models.zip) |
| Archive length | 28,318 bytes |
| Archive SHA-256 | `de4a3acf222cbc7f6c1a4d1a2bc449dd31b5db2c6ed153920d797752c3831d51` |
| Selected archive member | `1200V-SMA-SiC-MOSFET-SPICE-Models/MSCSMA120.lib` |
| Original member length | 29,805 bytes |
| Original member SHA-256 | `6e888e103977f539e62b64952797ded391d2a49c59738bfd53f5fbe4b1fc8df7` |
| Library build / internal version | 2026-05-05 / 2026.5 |
| Selected top-level subcircuit | `MSC040SMA120B`, drain / gate / source |
| Local adapter | `emi01-microchip-ngspice-v1` in [adapter.py](adapter.py) |
| Local adapted UTF-8 length | 29,136 bytes |
| Local adapted SHA-256 | `17732ddd7ab5361073f8f23594e32158270af47d9b7188cf0ce773189cafcf96` |
| Reference simulator | Existing hermetic static ngspice 46; [build provenance](../../third_party/ngspice/PROVENANCE.md) |

The original archive is a public direct download without a login or submitted form. It is
fetched as a checksum-pinned Bazel input. The adapter checks archive size and SHA-256 **before**
opening the ZIP, then checks the unique selected member's size and SHA-256. It never extracts
other members. A replacement archive, changed member, duplicate member, or absent download
fails closed. It also verifies the complete adapted byte hash.

The device reference is [Microsemi datasheet 050-7734, revision C, October 2019](https://ww1.microchip.com/downloads/en/DeviceDoc/Microsemi_MSC040SMA120B_SiC_MOSFET_Datasheet_RevC.PDF),
download SHA-256 `accd7ec9156129c36e564401479ccf77b7347f3be75183a5f07e09eb2710ad78`.
That version is an explicit historical reference, not an assertion that it is the newest
datasheet. The model's later build date does not establish a datasheet fit or measured accuracy.

## Redistribution and use boundary

This is **proprietary vendor material**, not an open-source dependency. The catalogue and the
library header disclaim guaranteed accuracy and call the information proprietary; the header
does not grant redistribution rights. Microchip's [general software EULA](https://www.microchip.com/en-us/about/license-agreement-end-user)
permits source-form modification for the specified Microchip-product use and restricts
third-party distribution. That general page is recorded as context: the model catalogue does
not separately identify it as the model's complete license. This audit does not assert a broader
license than the public evaluation purpose and displayed terms establish.

Accordingly, neither the original archive/member nor the generated model is committed,
published, included in raw evidence, or embedded in retained decks. Reproduction obtains the
exact original from Microchip; adaptation occurs in local temporary storage, retaining notices.
Published artifacts contain the repository's own harness/fixtures, model hashes, source links,
and simulation results. Decks contain an include reference only. A missing or changed vendor
download is an explicit provenance failure, never permission to use an unpinned replacement.

## Syntax rejection and bounded translation

An exploratory attempt to load the unmodified library using the existing ngspice 46 with
PSpice compatibility failed. Two concrete incompatibilities were observed:

1. ngspice's compatibility reader maps the local identifier `TEMP` to ambient `temper`.
   The resulting nested subcircuit call is malformed and reports an undefined temperature
   parameter. The vendor instead uses a parameter passed to each instance.
2. Behavioral `Fgd` and `Fds` cards use `VALUE=`, which this reader does not accept as a
   behavioral current expression. It reports undefined `v`/`i` parameters during expansion.

The untouched model is therefore **rejected for direct execution** by this build. The study
accepts only a separately identified mechanical translation:

- Decode the original Windows-1252 text, normalize CRLF/CR to LF, and emit UTF-8.
- Alpha-rename all 109 complete `TEMP` tokens, case-insensitively, to `TJ_C`.
  No simulator `.TEMP` command is being rewritten; the pinned member contains none.
- Change exactly one `Fgd` source with output terminals 42/23 to `Bgd ... I=` and exactly one
  `Fds` source with terminals 42/44 to `Bds ... I=`. Keep the complete expressions, output
  nodes, sensing sources, baseline capacitors, and all numerical parameters unchanged.
- Reject unexpected counts, any remaining `F` source, or a different adapted hash.

No waveform, current, capacitance, threshold, temperature coefficient, or charge parameter is
fitted. There is no simulator patch, XSPICE model, encrypted code, OSDI model, plugin, or ambient
simulator. E/G behavioral sources, functions, arithmetic, powers, conditional expressions,
ordinary RLC elements, voltage-current sensing, and nested subcircuits remain in the external
oracle. Required Ohmnivore production semantics are a future contract, not implemented here.

### Direction and dynamic-current audit

The Cgd branch is drain 42 → capacitor → sensing source Vgd → gate 23. Positive `i(Vgd)`
therefore points drain to gate. The behavioral source has the **same** direction. Its total
terminal displacement current is the baseline capacitor current times **1 + multiplier**.
Replacing Fgd with Bgd does not reverse that direction or remove the baseline capacitor.

Similarly the Cds branch is drain 42 → capacitor → sensing source Vds → node 44; Fds/Bds
also flows 42 → 44. A finite p-well resistor connects node 44 to source 32. The output
capacitance multiplier depends on external drain-to-source voltage 42/32, while the sensing
capacitor voltage is 42/44. Those are not interchangeable during a transient. The adapter
preserves both the internal dynamic state and this voltage dependence. No other element
references the renamed Fgd/Fds branch identities.

[adapter_test.py](adapter_test.py) exercises the same syntax translation on independently
authored affine-capacitance fixtures and runs the hermetic ngspice binary. Positive and negative
linear voltage ramps test both branch signs, the baseline **1 + multiplier**, current derivative,
and integrated charge against closed-form expressions. For example, with baseline C0 and
multiplier `a+bV`, the expected current is `C0*(1+a+bV)*dV/dt` and charge change is
`C0*((1+a)*(V2-V1)+b*(V2^2-V1^2)/2)`. These tests establish the translation mechanism;
they do not independently validate the vendor's fitted capacitance curves. Complete model
switching/refinement qualification is retained separately by the study.

## Physical limits that matter to the study

The model includes package gate/source inductance and resistance, forward/reverse channel
conduction, body-diode conduction, fixed Cgs, and nonlinear Miller/output displacement current.
The fixture adds drain/commutation inductance and chassis capacitance because those are not
supplied by the selected package subcircuit. Junction temperature is an externally fixed
parameter, not an electrothermal solution.

The vendor explicitly omits bipolar body-diode reverse-recovery charge. This is a material
limitation: revision-C datasheet Table 5 reports 610 nC recovery charge at 800 V/40 A,
while Table 4 gives 560 microjoules turn-on energy with body-diode commutation versus
275 microjoules with the specified external SiC Schottky diode. These differing conditions
illustrate why a numerical result must not be presented as a full hardware-loss prediction.
They are **not** correction factors for the study's 400 V/about 20 A double-pulse fixture.

The selected fixture deliberately retains the vendor's stated omission. It qualifies numerical
switching behavior **within that model's scope**; it does not qualify recovery-dominated
commutation against silicon carbide hardware. No multiplication of simulated Eon, invented
recovery element, or empirical fit is introduced. The missing charge can change ringing,
overshoot, turn-on energy, and EMI spectra. The research-mask model reserve is an engineering
assumption and cannot prove that this omission is bounded. Avalanche, short-circuit behavior,
self-heating, statistical device spread and laboratory correlation remain unqualified.

## Public alternatives investigated

| Alternative | Investigation and bounded decision |
|---|---|
| Wolfspeed public SiC models | [Model portal](https://www.wolfspeed.com/tools-and-support/power/ltspice-and-plecs-models/) and [terms](https://www.wolfspeed.com/legal/terms-of-use/) were inspected. The portal is aimed at LTspice/PLECS, and public material does not grant repository redistribution. Its [module guide](https://assets.wolfspeed.com/uploads/2023/10/Wolfspeed_PRD-07913_Power_Modules_SPICE_Models_User_Guide.pdf) describes LTspice validation. No exact Wolfspeed model was accepted or qualified for this study. Vendor-targeted syntax is not assumed to work in ngspice. |
| Infineon AIMW120R045M1 | [Official model listing](https://www.infineon.com/gated/infineon-coolsic-mosfet-automotive-1200v-g1-2.1-enc-aimw120r045m1-simulationmodels-en_1c6233bb-96dc-45da-863d-d00690bb9404) requires login and describes an encrypted model. It was not selected for an unattended public-byte reproduction path; no ngspice execution was attempted. |
| ST SCTW70N120G2V | [Official product page](https://www.st.com/en/power-transistors/sctw70n120g2v.html) links [SPICE model 1.0](https://www.st.com/resource/en/spice_model/sctw70n120g2v_spice.zip), dated 2021-02-17. Direct retrieval failed/stalled in this environment. No bytes were accepted and no compatibility conclusion is claimed. |
| ROHM SCT3030AL | Official product/design-model information was inspected, but direct retrieval was access-blocked in this environment. No model bytes or ngspice compatibility were qualified. |
| Microchip external Schottky | [Public 1200 V diode archive](https://ww1.microchip.com/downloads/aemDocuments/documents/sic/ProductDocuments/BoardDesignFiles/1200V-SDA-SiC-DIODE-SPICE-Models.zip) downloaded: 3,124 bytes, SHA-256 `fd6252ad9a1bfec3c20ebb5be0e468a969750f53b91898a379d669bb1cafeda5`. It contains dynamic MSC015SDA120B_L1/other diode models and similar TEMP/F-VALUE syntax. Adding it would change the frozen device/topology domain; it was investigated but is not an input to EMI-01. |

The outcome is a bounded, explicitly adapted Microchip reference. Public availability alone
does not establish redistribution permission, ngspice compatibility, numerical accuracy, or
hardware correlation; each boundary above is recorded separately.
