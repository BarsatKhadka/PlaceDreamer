# PlaceDreamer — Design Set

All designs collected and **routed to a real chip** on the sky130 / OpenROAD-LibreLane
flow (LibreLane 3.0.4, sky130A PDK). Every design below reached detailed routing
with **0 DRC violations**. This is the diverse, multi-size, multi-macro dataset the
world model is built on.

**Count: 20 routed designs** — tiny (537 cells) → huge (microwatt/chameleon SoCs),
spanning pure-logic, SRAM-macro, SoC-with-macros, and a caravel wrapper.

*(`rtl/` and its run dirs are gitignored — this doc is the tracked record.)*

---

## Full roster

Sorted by routed wirelength (a rough size proxy). DRC = 0 for all.

| design | source | top module | clock (port / period ns) | routed WL (µm) | instances¹ | macro | notes |
|---|---|---|---|---|---|---|---|
| **microwatt** | ORFS | `microwatt` | ext_clk / 40 | 8,257,932 | 304,633 | ✅ SRAM×4 | POWER CPU; routed clean, **full GDS signoff incomplete** (see caveats) |
| **salsa20** | openlane | `salsa20` | clk / auto | 2,143,685 | 283,875 | — | stream cipher, wide datapath |
| **jpeg** | ORFS | `jpeg_encoder` | clk / 25 | 1,852,283 | 309,196 | — | JPEG encoder (DCT); **held-out** |
| **ethmac** | CTS-Bench | `eth_top` | wb_clk_i / 15 | 1,820,647 | 66,458 | — | Ethernet MAC; multi-clock (routed on wb_clk only), tri-state |
| **aes** | CTS-Bench | `aes` | clk / 10 | 1,272,748 | 99,737 | — | AES-128 (secworks) |
| **aes_core** | openlane | `aes_core` | clk / 15 | 1,211,432 | 28,669 | — | AES core (dup-file bug fixed) |
| **chameleon** | ORFS | `soc_core` | HCLK / 20 | 944,636 | 2,383,856 | ✅ ibex+3×DFFRAM+2 | full SoC, GDS signed off (KLayout DRC 0) |
| **y_huff** | openlane | `y_huff` | clk / 50 | 607,250 | 21,198 | — | JPEG Huffman; needed diode-insert off to legalize |
| **PPU** | openlane | `PPU` | clk / auto | 592,198 | 159,087 | — | pixel processing unit |
| **sha256** | CTS-Bench | `sha256` | clk / 10 | 571,167 | 57,550 | — | SHA-256 core |
| **picorv32** | YosysHQ | `picorv32` | clk / 24 | 553,403 | 55,966 | — | RV32 CPU; the original baseline design |
| **test_sram_macro** | openlane | `test_sram_macro` | clk / 25 | 65,035 | 137,664 | ✅ PDK SRAM×2 | macro-recipe reference (PDK SRAM) |
| **mem_1r1w** | openlane | `mem_1r1w` | clk / 10 | 60,158 | 6,421 | (self-macro) | behavioral RAM hardened as a macro block |
| **regfile_2r1w** | openlane | `regfile_2r1w` | clk / 15 | 30,211 | 82,054 | ✅ mem_1r1w×2 | hierarchical macro (uses hardened mem_1r1w) |
| **zipdiv** | CTS-Bench | `zipdiv` | i_clk / 10 | 29,026 | 4,671 | — | ZipCPU divider |
| **usb** | openlane | `usb` | clk_48 / auto | 27,788 | 4,322 | — | USB core |
| **aes_user_project_wrapper** | openlane | `aes_user_project_wrapper` | wb_clk_i / 25 | 13,416 | 1 | ✅ pre-hardened AES | caravel wrapper (1 macro, fixed die) |
| **gcd** | openlane | `gcd` | clk / 10 | 7,304 | 1,184 | — | GCD calculator (tiny) |
| **xtea** | openlane | `xtea` | clock / 10 | 4,452 | 537 | — | XTEA cipher (tiny) |
| **s44** | openlane | `lut_s44` | config_clk / 30 | 4,035 | 9,097 | — | LUT design (tiny logic) |

¹ Instance count is the routed count; for macro designs it includes flattened macro
internals (e.g. chameleon 2.38M ≈ mostly the 3× DFFRAM contents), so it is not a
clean logic-size measure across designs.

---

## Macro designs (the SRAM/hard-macro set)

Six designs exercise the macro flow — the hardest integration work:

| design | macros | style |
|---|---|---|
| test_sram_macro | 2× `sky130_sram_1kbyte_1rw1r_32x256_8` | PDK SRAM macro |
| mem_1r1w | — (hardens *into* a macro) | behavioral RAM block |
| regfile_2r1w | 2× hardened `mem_1r1w` | **hierarchical** (harden-then-instantiate) |
| chameleon | `ibex_wrapper`, `apb_sys_0`, `DMC_32x16HC`, 3× `DFFRAM_4K` | big SoC, committed ORFS macros |
| microwatt | `RAM512`, `RAM32_1RW1R`, `Microwatt_FP_DFFRFile`, `multiply_add_64x64` | committed ORFS macros |
| aes_user_project_wrapper | pre-hardened `aes_example` | caravel user-project wrapper |

**The macro recipe** (proven across all six):
1. **Minimal plain-Verilog blackboxes** — ports only, **no** `parameter` / attributes /
   `` `ifdef `` (OpenROAD's STA verilog reader rejects those), but **include `VPWR`/`VGND`**
   ports where the RTL connects them.
2. `EXTRA_LEFS` / `EXTRA_LIBS` / `EXTRA_GDS` → the macro physical + timing views.
3. **`VERILOG_POWER_DEFINE: "LIBRELANE_UNUSED_POWER_DEFINE"`** — critical when RTL hard-wires
   std-cell power to nets named `VPWR`/`VGND` under `USE_POWER_PINS` (else PDN fails).
4. `MACRO_PLACEMENT_CFG` → manual `<instance> <x> <y> <orient>`; **`FP_MACRO_*_HALO` ≈ 60µm**
   (small halos caused design-wide met1/met2 shorts).
5. `FP_PDN_MACRO_HOOKS` per instance — **wildcards** for generate-loop names
   (`RAM.genblk1.*0.*RAM`), since OpenDB stores them with literal backslashes.
6. `FP_SIZING: absolute` + a `DIE_AREA` sized to the macros; signoff-tolerance flags
   (`QUIT_ON_{MAGIC_DRC,LVS_ERROR,SETUP,HOLD}: false`, `RUN_KLAYOUT_XOR: false`).

For hierarchical macros: **harden the sub-macro first**, then point the parent's
`EXTRA_LEFS`/`EXTRA_GDS` at the sub-macro's hardened outputs (see regfile_2r1w ← mem_1r1w).

---

## Sources

- **ORFS** (OpenROAD-flow-scripts, sky130hd): microwatt, jpeg, chameleon
- **CTS-Bench** (BarsatKhadka/CTS-Bench): aes, sha256, ethmac, zipdiv
- **openlane-ci-designs** (local LibreLane examples): aes_core, salsa20, usb, PPU, s44,
  gcd, xtea, y_huff, mem_1r1w, regfile_2r1w, test_sram_macro, aes_user_project_wrapper
- **YosysHQ upstream**: picorv32 (canonical `picorv32.v`)

---

## Held-out set (for generalization testing)

Following GAN-CTS / FastTuner convention — hold out **entire designs by family and by
size extreme**, not configs (holding out configs of a trained design is only
interpolation):

- **jpeg** — unseen family, medium-large
- **microwatt** — unseen, largest (size extrapolation)

Everything else is training-eligible. Evaluate on the held-out designs zero-shot and
(optionally) few-shot.

---

## Caveats (honest)

- **microwatt**: detailed-routed clean (0 DRC, complete DEF, routed WL captured) but its
  **final GDS signoff did not complete** (Magic/KLayout/LVS on 150k+ instances kept dying
  on disk-full / process kills). The routing data we need exists; the GDS artifact does not.
- **Timing not signed off** on macro designs (chameleon, microwatt): the hard macros have
  no timing `.lib`, so paths through them aren't timed, and `QUIT_ON_SETUP_VIOLATIONS:false`
  let the flow finish. These are DRC-clean routed layouts, **not** timing-closed tapeouts.
  Fine for extracting congestion/wirelength data; not a signoff-quality chip.
- **ethmac**: multi-clock design (wb_clk_i / mtx_clk / mrx_clk) routed with CTS on the
  wishbone clock only; the RX/TX clock domains are unconstrained.
- **Clock period "auto"**: some openlane designs set `CLOCK_PERIOD` inside a PDK sub-block
  rather than at top level; effective value comes from the PDK default.

---

## Environment note

Runs were on macOS via `nix-shell ~/librelane/shell.nix`. A recurring failure mode —
nix garbage-collection deleting librelane's files mid-run — was fixed by GC-rooting the
closure (`~/.pd-gcroots/`). Full-flow runs produce ~1–4 GB each; the data-generation
sweep should extract features/metrics then delete run dirs per iteration.
