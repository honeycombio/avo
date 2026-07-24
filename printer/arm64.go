package printer

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/mmcloughlin/avo/buildtags"
	"github.com/mmcloughlin/avo/internal/prnt"
	"github.com/mmcloughlin/avo/ir"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

// arm64 is an EXPERIMENTAL printer that lowers an avo file whose instructions
// were written against the amd64 ISA into Go arm64 (AArch64) assembly.
//
// It is deliberately NOT a general arm64 backend. avo's frontend, IR, register
// allocator and ABI handling are all amd64-oriented; this printer runs *after*
// pass.Compile and mechanically lowers the already-allocated x86 instruction
// stream to arm64, translating a single x86 instruction into one or more arm64
// instructions. The goal is correct, regenerable arm64 asm that beats Go's
// pure-Go fallback — not optimal hand-tuned code.
//
// Design:
//
//   - Registers: avo allocated physical x86 GP registers. We remap by physical
//     index to a fixed arm64 register (RAX->R0, RCX->R1, ...), reserving R14/R15
//     as scratch for address computation, immediate-to-memory stores and
//     memory-destination read-modify-write. Vector (XMM) registers map by index
//     to V registers. Pseudo registers (FP/SB/SP) pass through unchanged.
//
//   - Memory operands: x86 base+index*scale+disp has no arm64 equivalent, so an
//     indexed operand is lowered to "ADD scratch, base, index<<log2(scale)"
//     followed by a simple disp(scratch) access.
//
//   - Flags: avo's IR does not model EFLAGS. x86 sets flags implicitly and a
//     later Jcc/CMOVcc consumes them. We do NOT fuse; instead:
//
//   - comparisons (CMP*/TEST*) always emit an arm64 flag-setter (CMP/CMN/TST);
//
//   - arithmetic that an immediately-following branch/CMOV consumes is emitted
//     as the flag-setting S-variant (SUBS/ADDS/...);
//
//   - conditional branches and CMOVcc consume the live flags (Bcc / CSEL).
//     This is safe because every other lowering uses non-flag-setting arm64 ops,
//     so NZCV survives from producer to consumer.
//
// Unsupported opcodes panic loudly, so growing the supported set surfaces
// exactly what is missing.
type arm64 struct {
	cfg Config
	prnt.Generator

	// pending buffers lowered instructions (opcode, operands) so a block can be
	// flushed with operands column-aligned, matching the goasm printer (and thus
	// asmfmt). clear tracks whether a blank line is already present.
	pending [][2]string
	clear   bool

	// One-instruction constant-tracking window: constReg holds constVal when
	// constOK and the immediately preceding instruction was "MOVQ/MOVL $imm,
	// constReg". Any other instruction or a label invalidates it (comments are
	// transparent). Used to fold BMI2 register-control ops (BEXTR/BZHI/SHLX/
	// SHRX) whose control register is loaded with a constant right before use
	// -- the x86 encodings have no immediate forms for these, but the arm64
	// equivalents do (UBFX/AND/LSL/LSR), collapsing multi-instruction mask
	// builds into one instruction.
	constReg string
	constVal int64
	constOK  bool
}

// NewARM64Asm constructs a printer for writing Go arm64 assembly files by
// lowering an amd64 avo instruction stream. EXPERIMENTAL.
func NewARM64Asm(cfg Config) Printer { return &arm64{cfg: cfg} }

// armReg maps an x86 physical GP register index to an arm64 register name.
// x86 indices: 0=AX 1=CX 2=DX 3=BX 4=SP 5=BP 6=SI 7=DI 8..15=R8..R15.
// RSP (4) is never allocated by avo. All targets are arm64 caller-saved
// registers (R0-R17, excluding R18), so the lowered leaf functions need no
// callee-save prologue.
var armReg = map[reg.Index]string{
	0: "R0", 1: "R1", 2: "R2", 3: "R3", 5: "R4",
	6: "R5", 7: "R6",
	8: "R7", 9: "R8", 10: "R9", 11: "R10", 12: "R11", 13: "R12", 14: "R13", 15: "R14",
}

const (
	// Scratch registers reserved for lowering; never produced by the x86 map.
	scratchAddr = "R15" // effective-address computation for indexed/RMW operands
	scratchVal  = "R16" // immediate materialization / RMW value
)

func (p *arm64) Print(f *ir.File) ([]byte, error) {
	p.header(f)
	twins := collectTwins(f)
	for _, s := range f.Sections {
		switch s := s.(type) {
		case *ir.Function:
			p.function(s, twins)
		case *ir.Global:
			p.global(s)
		default:
			panic("unknown section type")
		}
	}
	return p.Result()
}

// logicalName strips the trailing _amd64/_bmi2 arch+feature suffixes from a
// generated function name, yielding the architecture-independent base. The _safe
// marker and everything before it are preserved, so a generic variant and its
// BMI2 twin reduce to the same base (e.g. both foo_safe_amd64 and foo_safe_bmi2
// yield foo_safe).
func logicalName(name string) string {
	for {
		switch {
		case strings.HasSuffix(name, "_bmi2"):
			name = strings.TrimSuffix(name, "_bmi2")
		case strings.HasSuffix(name, "_amd64"):
			name = strings.TrimSuffix(name, "_amd64")
		default:
			return name
		}
	}
}

// arm64Name is the symbol emitted for a lowered function. A name carrying an
// arch/feature suffix is renamed to <base>_arm64; a name with no such suffix is
// architecture-independent and reused verbatim (the arm64 file selects it via a
// build tag, as with the hand-written differential tests).
func arm64Name(name string) string {
	if !strings.Contains(name, "_amd64") && !strings.Contains(name, "bmi2") {
		return name
	}
	return logicalName(name) + "_arm64"
}

// twinPair records, for a logical function name, which of the generic/BMI2
// variants are present.
type twinPair struct {
	generic, bmi2 bool
}

// collectTwins maps each logical function name to which variants exist, so
// function() can tell a real generic/BMI2 pair (a choice to make, per
// Config.ARM64PreferBMI2) apart from a function with only one variant (lower
// it regardless -- there is no alternative).
func collectTwins(f *ir.File) map[string]twinPair {
	twins := make(map[string]twinPair)
	for _, s := range f.Sections {
		fn, ok := s.(*ir.Function)
		if !ok {
			continue
		}
		name := logicalName(fn.Name)
		t := twins[name]
		if strings.Contains(fn.Name, "bmi2") {
			t.bmi2 = true
		} else {
			t.generic = true
		}
		twins[name] = t
	}
	return twins
}

func (p *arm64) header(f *ir.File) {
	p.Comment(p.cfg.GeneratedWarning())
	p.Comment("EXPERIMENTAL arm64 output lowered from an amd64 avo program.")
	// This printer only ever emits arm64; require GOARCH=arm64 so the file does
	// not clash with the amd64 implementation of the same symbols.
	if len(f.Constraints) > 0 {
		constraints, err := buildtags.Format(f.Constraints)
		if err != nil {
			p.AddError(err)
		}
		constraints = strings.Replace(constraints, "//go:build ", "//go:build arm64 && ", 1)
		p.NL()
		p.Printf(constraints)
	} else {
		p.NL()
		p.Printf("//go:build arm64\n")
	}
	if len(f.Includes) > 0 {
		p.NL()
		for _, path := range f.Includes {
			p.Printf("#include \"%s\"\n", path)
		}
	}
}

func (p *arm64) global(g *ir.Global) {
	p.NL()
	for _, d := range g.Data {
		a := operand.NewDataAddr(g.Symbol, d.Offset)
		p.Printf("DATA %s/%d, %s\n", a.Asm(), d.Value.Bytes(), d.Value.Asm())
	}
	p.Printf("GLOBL %s(SB), %s, $%d\n", g.Symbol, g.Attributes.Asm(), g.Size)
}

func (p *arm64) function(f *ir.Function, twins map[string]twinPair) {
	// On arm64 we emit one implementation per logical function. Where the
	// generator produced both a generic ("_amd64") and a BMI2 ("_bmi2") variant,
	// Config.ARM64PreferBMI2 picks which one; the other is skipped. arm64 has
	// native equivalents for the BMI2 idioms, but they are not reliably faster
	// than lowering the generic path -- BMI2 x86 code is tuned for x86 (e.g.
	// BEXTR packs into one instruction what arm64 needs two UBFX to unpack), so
	// this is a measure-and-choose knob, not a default preference. A function
	// with only one variant (no twin to choose between) is always lowered.
	isBMI2 := strings.Contains(f.Name, "bmi2")
	if t := twins[logicalName(f.Name)]; t.generic && t.bmi2 && isBMI2 != p.cfg.ARM64PreferBMI2 {
		p.NL()
		label := "generic"
		if p.cfg.ARM64PreferBMI2 {
			label = "BMI2"
		}
		p.Comment(fmt.Sprintf("skipped %s (%s twin preferred on arm64)", f.Name, label))
		return
	}
	name := arm64Name(f.Name)

	p.NL()
	p.Comment(f.Stub())
	// The BMI2 requirement does not survive lowering (the ops become native arm64
	// instructions), so do not carry a misleading "Requires: BMI2" over.
	if len(f.ISA) > 0 && !isBMI2 {
		p.Comment("Requires: " + strings.Join(f.ISA, ", "))
	}
	p.Printf("TEXT %s%s(SB)", dot, name)
	if f.Attributes != 0 {
		p.Printf(", %s", f.Attributes.Asm())
	}
	p.Printf(", %s\n", textsize(f))

	p.clear = true
	nodes := f.Nodes
	setflags := flagProducers(nodes)
	subwordSafe := subwordSafeEqNe(nodes)
	p.constOK = false
	for idx := 0; idx < len(nodes); idx++ {
		switch n := nodes[idx].(type) {
		case ir.Label:
			p.constOK = false
			p.flush()
			p.ensureclear()
			p.Printf("%s:\n", n)
		case *ir.Comment:
			p.flush()
			p.ensureclear()
			for _, line := range n.Lines {
				p.Printf("\t// %s\n", line)
			}
		case *ir.Instruction:
			switch {
			case n.Opcode == "JMP":
				p.emit("JMP %s", n.Operands[0].Asm())
			case strings.HasPrefix(n.Opcode, "CMOV"):
				p.lowerCMOV(n)
			case strings.HasPrefix(n.Opcode, "SET"):
				p.lowerSET(n)
			case isConditionalBranch(n):
				p.emit("%s %s", branchMnemonic(n.Opcode), n.Operands[0].Asm())
			default:
				p.lower(n, setflags[idx], subwordSafe[idx])
			}
			p.trackConst(n)
			if n.IsTerminal || n.IsUnconditionalBranch() {
				p.flush()
			}
		default:
			panic("unexpected node type")
		}
	}
	p.flush()
}

// emit buffers a single lowered arm64 instruction. The block is column-aligned
// and written by flush(), matching the goasm printer's layout (and asmfmt).
func (p *arm64) emit(format string, args ...interface{}) {
	line := fmt.Sprintf(format, args...)
	op, operands := line, ""
	if i := strings.IndexByte(line, ' '); i >= 0 {
		op, operands = line[:i], line[i+1:]
	}
	p.pending = append(p.pending, [2]string{op, operands})
	p.clear = false
}

// flush writes the buffered instructions with operands aligned to a common
// column (width of the widest opcode in the block), like the goasm printer.
func (p *arm64) flush() {
	if len(p.pending) == 0 {
		return
	}
	width := 0
	for _, in := range p.pending {
		if in[1] != "" && len(in[0]) > width {
			width = len(in[0])
		}
	}
	for _, in := range p.pending {
		if in[1] != "" {
			p.Printf("\t%-*s%s\n", width+1, in[0], in[1])
		} else {
			p.Printf("\t%s\n", in[0])
		}
	}
	p.pending = nil
}

// ensureclear emits a blank separator line unless one is already present.
func (p *arm64) ensureclear() {
	if !p.clear {
		p.NL()
		p.clear = true
	}
}

// rename returns the arm64 syntax for a register operand.
func rename(r reg.Register) string {
	switch r.Kind() {
	case reg.KindGP:
		if ph, ok := r.(reg.Physical); ok {
			if name, ok := armReg[ph.PhysicalIndex()]; ok {
				return name
			}
			panic(fmt.Sprintf("arm64: unmapped x86 GP register index %d", ph.PhysicalIndex()))
		}
		panic("arm64: non-physical register reached printer (run pass.Compile first)")
	case reg.KindVector:
		if ph, ok := r.(reg.Physical); ok {
			return fmt.Sprintf("V%d", ph.PhysicalIndex())
		}
		panic("arm64: non-physical vector register reached printer")
	}
	// Pseudo registers. FP/SB/PC share their tokens with arm64; the amd64
	// hardware stack pointer "SP" becomes the arm64 hardware stack pointer
	// "RSP" (bare "SP" is the pseudo frame-relative register in Go arm64 asm).
	if a := r.Asm(); a == "SP" {
		return "RSP"
	}
	return r.Asm()
}

// isHighByte reports whether op is an x86 high-byte register (AH/BH/CH/DH),
// which has no arm64 equivalent and must be read with a bitfield extract.
func isHighByte(op operand.Op) bool {
	r, ok := op.(reg.Register)
	if !ok {
		return false
	}
	switch r.Asm() {
	case "AH", "BH", "CH", "DH":
		return true
	}
	return false
}

func operandReg(op operand.Op) string {
	r, ok := op.(reg.Register)
	if !ok {
		panic(fmt.Sprintf("arm64: expected register operand, got %T", op))
	}
	return rename(r)
}

func immAsm(op operand.Op) (string, bool) {
	if c, ok := op.(operand.Constant); ok {
		return c.Asm(), true
	}
	return "", false
}

// trackConst maintains the one-instruction constant window: after a plain
// "MOVQ/MOVL $imm, reg" the register's value is known while lowering the next
// instruction. Every other instruction closes the window (labels do too, in
// the dispatch loop). MOVL immediates are <=32-bit and x86 zero-extends the
// destination, so the tracked 64-bit value is the same for both widths.
func (p *arm64) trackConst(n *ir.Instruction) {
	p.constOK = false
	if n.Opcode != "MOVQ" && n.Opcode != "MOVL" || len(n.Operands) != 2 {
		return
	}
	v, ok := immVal(n.Operands[0])
	if !ok {
		return
	}
	r, ok := n.Operands[1].(reg.Register)
	if !ok {
		return
	}
	p.constReg = rename(r)
	p.constVal = v
	p.constOK = true
}

// knownConst reports the constant held by register operand op, if the
// immediately preceding instruction loaded it with an immediate.
func (p *arm64) knownConst(op operand.Op) (int64, bool) {
	r, ok := op.(reg.Register)
	if !ok || !p.constOK || rename(r) != p.constReg {
		return 0, false
	}
	return p.constVal, true
}

func immVal(op operand.Op) (int64, bool) {
	c, ok := immAsm(op)
	if !ok {
		return 0, false
	}
	v, err := strconv.ParseInt(strings.TrimPrefix(c, "$"), 0, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func log2scale(s uint8) int {
	switch s {
	case 1:
		return 0
	case 2:
		return 1
	case 4:
		return 2
	case 8:
		return 3
	}
	panic(fmt.Sprintf("arm64: bad scale %d", s))
}

// memAsm lowers a memory operand to a simple base+disp arm64 operand string,
// emitting an ADD into scratchAddr first for indexed operands.
func (p *arm64) memAsm(m operand.Mem) string { return p.memAsmW(m, 0) }

// memAsmW renders a memory operand for a scalar access of the given width in
// bytes (0 when the caller cannot use arm64's folded base+index form).
func (p *arm64) memAsmW(m operand.Mem, width int) string {
	if m.Symbol.Name != "" {
		s := m.Symbol.String() + fmt.Sprintf("%+d", m.Disp)
		if m.Base != nil {
			s += fmt.Sprintf("(%s)", rename(m.Base))
		}
		if m.Index != nil {
			panic("arm64: indexed symbol operand not supported")
		}
		return s
	}
	if m.Index == nil || m.Scale == 0 {
		if m.Disp != 0 {
			return fmt.Sprintf("%d(%s)", m.Disp, rename(m.Base))
		}
		return fmt.Sprintf("(%s)", rename(m.Base))
	}
	sh := log2scale(m.Scale)
	// arm64 scalar loads/stores can fold base+index addressing into the
	// instruction, saving the separate effective-address ADD that x86's richer
	// addressing otherwise costs us on every access. The architecture allows it
	// only when the index is unscaled or scaled by exactly the access width, and
	// never alongside a displacement, so fall through to scratchAddr otherwise.
	// width == 0 means the caller cannot use the folded form (e.g. FMOVQ).
	if m.Disp == 0 && (m.Scale == 1 || (width > 0 && int(m.Scale) == width)) {
		if sh == 0 {
			return fmt.Sprintf("(%s)(%s)", rename(m.Base), rename(m.Index))
		}
		return fmt.Sprintf("(%s)(%s<<%d)", rename(m.Base), rename(m.Index), sh)
	}
	if sh == 0 {
		p.emit("ADD %s, %s, %s", rename(m.Index), rename(m.Base), scratchAddr)
	} else {
		p.emit("ADD %s<<%d, %s, %s", rename(m.Index), sh, rename(m.Base), scratchAddr)
	}
	if m.Disp != 0 {
		return fmt.Sprintf("%d(%s)", m.Disp, scratchAddr)
	}
	return fmt.Sprintf("(%s)", scratchAddr)
}

// memAddr materializes the effective address of m into a GP register and returns
// its name (for instructions like VLD1/VST1 that take only a base register).
func (p *arm64) memAddr(m operand.Mem) string {
	if m.Symbol.Name != "" {
		panic("arm64: address-of symbol operand not supported")
	}
	base := rename(m.Base)
	cur := base
	if m.Index != nil && m.Scale != 0 {
		sh := log2scale(m.Scale)
		if sh == 0 {
			p.emit("ADD %s, %s, %s", rename(m.Index), base, scratchAddr)
		} else {
			p.emit("ADD %s<<%d, %s, %s", rename(m.Index), sh, base, scratchAddr)
		}
		cur = scratchAddr
	}
	if m.Disp != 0 {
		p.emit("ADD $%d, %s, %s", m.Disp, cur, scratchAddr)
		cur = scratchAddr
	}
	return cur
}

func (p *arm64) lower(i *ir.Instruction, flags, subwordEqNeSafe bool) {
	ops := i.Operands
	switch i.Opcode {
	case "RET":
		p.emit("RET")

	// ---- moves and loads ----
	case "MOVQ":
		p.lowerMove("MOVD", ops[0], ops[1])
	case "MOVL":
		p.lowerMOVL(ops[0], ops[1])
	case "MOVW":
		p.lowerMOVW(ops[0], ops[1])
	case "MOVB":
		p.lowerMOVB(ops[0], ops[1])
	case "MOVWQSX": // load/extend int16, sign-extend (mem or reg source)
		p.emit("MOVH %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVWQZX": // load/extend uint16, zero-extend
		p.emit("MOVHU %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVBQZX": // load/extend uint8, zero-extend
		p.emit("MOVBU %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVUPS", "MOVOU", "MOVOA":
		// 128-bit SSE moves. arm64 NEON loads/stores have no alignment
		// requirement, so the aligned (MOVOA) and unaligned (MOVOU) forms lower
		// identically.
		p.lowerMOVUPS(ops[0], ops[1])
	case "PXOR":
		// 128-bit bitwise XOR of two vector registers.
		a, b := operandReg(ops[0]), operandReg(ops[1])
		p.emit("VEOR %s.B16, %s.B16, %s.B16", a, b, b)

	// ---- arithmetic / logic (dst is last operand) ----
	case "ADDQ":
		p.lowerArith("ADD", "ADDS", ops[0], ops[1], flags)
	case "SUBQ":
		p.lowerArith("SUB", "SUBS", ops[0], ops[1], flags)
	case "ANDQ":
		p.lowerArith("AND", "ANDS", ops[0], ops[1], flags)
	case "ORQ":
		p.lowerArith("ORR", "", ops[0], ops[1], flags)

	// 32-bit ALU. x86 zero-extends a 32-bit register destination to 64 bits,
	// which the arm64 W-forms do as well, so these map directly.
	case "ADDL":
		p.lowerArithW("ADDW", "ADDSW", ops[0], ops[1], flags)
	case "SUBL":
		p.lowerArithW("SUBW", "SUBSW", ops[0], ops[1], flags)
	case "ANDL":
		p.lowerArithW("ANDW", "ANDSW", ops[0], ops[1], flags)
	case "ORL":
		p.lowerArithW("ORRW", "", ops[0], ops[1], flags)
	case "XORQ":
		if ra, ok := ops[0].(reg.Register); ok {
			if rb, ok := ops[1].(reg.Register); ok && rename(ra) == rename(rb) {
				p.emit("MOVD $0, %s", rename(rb))
				return
			}
		}
		p.lowerArith("EOR", "", ops[0], ops[1], flags)
	case "XORL":
		if ra, ok := ops[0].(reg.Register); ok {
			if rb, ok := ops[1].(reg.Register); ok && rename(ra) == rename(rb) {
				p.emit("MOVD $0, %s", rename(rb))
				return
			}
		}
		p.lowerArith("EORW", "", ops[0], ops[1], flags)
	case "ADDB":
		// x86 ADDB replaces only the destination's addressed byte with the byte
		// sum (wrapping mod 256). Compute the sum in scratch — addition carries
		// travel upward only, so garbage above bit 7 of either input cannot
		// affect bits 7:0 — and insert exactly those 8 bits.
		if isHighByte(ops[1]) {
			v := p.byteVal(ops[0])
			d := operandReg(ops[1])
			p.emit("UBFX $8, %s, $8, %s", d, scratchAddr)
			p.emit("ADD %s, %s, %s", v, scratchAddr, scratchAddr)
			p.emit("BFI $8, %s, $8, %s", scratchAddr, d)
			return
		}
		v := p.byteVal(ops[0])
		d := operandReg(ops[1])
		p.emit("ADD %s, %s, %s", v, d, scratchAddr)
		p.emit("BFI $0, %s, $8, %s", scratchAddr, d)
	case "ADCB":
		// Narrow lowering of the carry-accumulate idiom "CMPQ x, y; ADCB $0, dst":
		// dst's low byte += x86 CF, where CF after a compare is the unsigned
		// borrow (x < y). arm64 inverts carry for subtraction, so borrow is the
		// LO condition and no-borrow is HS: CSINC yields dst on HS and dst+1
		// otherwise; only bits 7:0 of the result are inserted, as on x86.
		// Assumes NZCV comes from a compare (flagProducers verifies the producer).
		if v, ok := immVal(ops[0]); !ok || v != 0 {
			panic("arm64: ADCB only supported with a $0 immediate source")
		}
		if isHighByte(ops[1]) {
			panic("arm64: ADCB high-byte destination not supported")
		}
		d := operandReg(ops[1])
		p.emit("CSINC HS, %s, %s, %s", d, d, scratchVal)
		p.emit("BFI $0, %s, $8, %s", scratchVal, d)
	case "BSWAPL":
		// Byte-reverse the low 32 bits; the 32-bit result zero-extends, as on x86.
		r := operandReg(ops[0])
		p.emit("REVW %s, %s", r, r)
	case "INCQ":
		p.lowerIncDec("ADD", "ADDS", ops[0], flags)
	case "INCL":
		p.lowerIncDec("ADDW", "ADDSW", ops[0], flags)
	case "DECQ":
		p.lowerIncDec("SUB", "SUBS", ops[0], flags)
	case "DECL":
		p.lowerIncDec("SUBW", "SUBSW", ops[0], flags)
	case "NEGQ":
		p.emit("NEG %s, %s", operandReg(ops[0]), operandReg(ops[0]))
	case "NOTQ":
		p.emit("MVN %s, %s", operandReg(ops[0]), operandReg(ops[0]))
	case "SHRQ":
		p.lowerShift("LSR", ops[0], ops[1])
	case "SHLQ":
		p.lowerShift("LSL", ops[0], ops[1])
	case "SARQ":
		p.lowerShift("ASR", ops[0], ops[1])
	case "SHRL":
		p.lowerShift("LSRW", ops[0], ops[1])
	case "SHLL":
		p.lowerShift("LSLW", ops[0], ops[1])
	case "SARL":
		p.lowerShift("ASRW", ops[0], ops[1])
	case "SHLB":
		// x86 SHLB shifts only the destination's low byte and leaves the rest of
		// the register untouched. Shift in scratch and insert bits 7:0, matching
		// the partial-register semantics (see lowerMOVB).
		if isHighByte(ops[1]) {
			panic("arm64: SHLB high-byte destination not supported")
		}
		d := operandReg(ops[1])
		p.emit("LSLW %s, %s, %s", p.regOrImm(ops[0]), d, scratchVal)
		p.emit("BFI $0, %s, $8, %s", scratchVal, d)
	case "ROLQ":
		p.lowerROL(ops[0], ops[1])

	// ---- BMI2 flag-free shifts/rotate: SHIFTX count, src, dst ----
	// arm64 register shifts are already flag-free and take an arbitrary count
	// register, so these map one-to-one (the count is masked mod 64, matching x86).
	case "SHLXQ":
		p.lowerShiftX("LSL", ops[0], ops[1], ops[2])
	case "SHRXQ":
		p.lowerShiftX("LSR", ops[0], ops[1], ops[2])
	case "SARXQ":
		p.lowerShiftX("ASR", ops[0], ops[1], ops[2])
	case "RORXQ":
		p.lowerRORX(ops[0], ops[1], ops[2])

	case "LEAQ":
		p.lowerLEA(ops[0].(operand.Mem), operandReg(ops[1]))
	case "LEAL":
		// 32-bit LEA: the address arithmetic is computed modulo 2^32 and the
		// result zero-extends into the 64-bit destination.
		d := operandReg(ops[1])
		p.lowerLEA(ops[0].(operand.Mem), d)
		p.emit("MOVWU %s, %s", d, d)

	case "BTSQ":
		a := operandReg(ops[0])
		b := operandReg(ops[1])
		p.emit("MOVD $1, %s", scratchVal)
		p.emit("LSL %s, %s, %s", a, scratchVal, scratchVal)
		p.emit("ORR %s, %s, %s", scratchVal, b, b)

	case "BSFQ", "TZCNTQ":
		// Index of the lowest set bit == count of trailing zeros, which arm64
		// spells as reverse-then-count-leading-zeros. For a zero input this
		// yields 64, matching TZCNT exactly; x86 BSF leaves the destination
		// undefined there, so returning 64 is a valid refinement. The generators
		// emit both mnemonics guarded by #ifdef GOAMD64_v3, and both lower here
		// to the same sequence.
		src := operandReg(ops[0])
		dst := operandReg(ops[1])
		p.emit("RBIT %s, %s", src, dst)
		p.emit("CLZ %s, %s", dst, dst)

	case "BSRQ":
		src := operandReg(ops[0])
		dst := operandReg(ops[1])
		p.emit("CLZ %s, %s", src, scratchVal)
		p.emit("MOVD $63, %s", dst)
		p.emit("SUB %s, %s, %s", scratchVal, dst, dst)

	// ---- BMI2 bit-field ops: BZHI/BEXTR (counts assumed < 64, as in zstd) ----
	case "BZHIQ":
		p.lowerBZHI(ops[0], ops[1], ops[2])
	case "BEXTRQ":
		p.lowerBEXTR(ops[0], ops[1], ops[2])

	// ---- multiplication ----
	case "MULXQ":
		p.lowerMULX(ops[0], ops[1], ops[2])
	case "IMUL3Q":
		p.lowerIMUL3(ops[0], ops[1], ops[2])
	case "IMULQ":
		p.lowerIMUL(ops)
	case "MULQ":
		p.lowerWideMul("UMULH", ops[0])

	// ---- comparisons: always emit an arm64 flag-setter ----
	// The compare must run at the operand width: a 64-bit CMP of registers whose
	// upper bits are not provably zero sets flags from the wrong bits. arm64 has
	// a native 32-bit form (CMPW/TSTW); sub-32-bit widths would generally need the
	// operands extended for the consuming condition's signedness, which is not
	// modelled. The one exception is when every consumer is EQ/NE (checked by
	// subwordEqNeSafe, from subwordSafeEqNe): equality doesn't depend on sign, so
	// zero-extending both operands to the compared width before a full-width
	// compare is correct unconditionally, with no need to prove the operands were
	// already clean above that width. Anything else fails loudly rather than
	// silently comparing full 64-bit registers.
	case "CMPQ":
		p.lowerCompare("CMP", "CMN", ops[0], ops[1])
	case "CMPL":
		p.lowerCompare("CMPW", "CMNW", ops[0], ops[1])
	case "CMPW", "CMPB":
		if !subwordEqNeSafe {
			panic(fmt.Sprintf("arm64: %s not supported (sub-32-bit compare needs width- and sign-correct operand extension unless every consumer is EQ/NE)", i.Opcode))
		}
		bits := 16
		if i.Opcode == "CMPB" {
			bits = 8
		}
		p.lowerSubwordCompareEqNe(bits, "CMP", ops[0], ops[1])
	case "TESTQ":
		p.lowerTest("TST", ops[0], ops[1])
	case "TESTL":
		p.lowerTest("TSTW", ops[0], ops[1])
	case "TESTW", "TESTB":
		if !subwordEqNeSafe {
			panic(fmt.Sprintf("arm64: %s not supported (sub-32-bit test needs width-correct operands unless every consumer is EQ/NE)", i.Opcode))
		}
		bits := 16
		if i.Opcode == "TESTB" {
			bits = 8
		}
		p.lowerSubwordTestEqNe(bits, ops[0], ops[1])

	default:
		panic(fmt.Sprintf("arm64: unsupported opcode %q (operands: %s)", i.Opcode, joinOperands(ops)))
	}
}

func (p *arm64) regOrImm(op operand.Op) string {
	if imm, ok := immAsm(op); ok {
		return imm
	}
	return operandReg(op)
}

// srcAsm renders a source operand that may be a memory reference, immediate, or
// register.
func (p *arm64) srcAsm(op operand.Op) string {
	if m, ok := op.(operand.Mem); ok {
		return p.memAsm(m)
	}
	return p.regOrImm(op)
}

// valReg returns a register name holding op's value, loading a memory operand
// into scratchVal first. op must not be an immediate. x86 permits at most one
// memory operand per instruction, so callers never contend for scratchVal.
func (p *arm64) valReg(op operand.Op) string {
	if m, ok := op.(operand.Mem); ok {
		p.emit("MOVD %s, %s", p.memAsm(m), scratchVal)
		return scratchVal
	}
	return operandReg(op)
}

// lowerMove handles MOV* of reg/imm/mem to reg/mem.
func (p *arm64) lowerMove(op string, src, dst operand.Op) {
	w := accessWidth(op)
	dmem, dstIsMem := dst.(operand.Mem)
	smem, srcIsMem := src.(operand.Mem)
	switch {
	case dstIsMem:
		if imm, ok := immAsm(src); ok {
			p.emit("MOVD %s, %s", imm, scratchVal)
			p.emit("%s %s, %s", op, scratchVal, p.memAsmW(dmem, w))
			return
		}
		p.emit("%s %s, %s", op, operandReg(src), p.memAsmW(dmem, w))
	case srcIsMem:
		p.emit("%s %s, %s", op, p.memAsmW(smem, w), operandReg(dst))
	default:
		if imm, ok := immAsm(src); ok {
			p.emit("%s %s, %s", op, imm, operandReg(dst))
			return
		}
		p.emit("%s %s, %s", op, operandReg(src), operandReg(dst))
	}
}

// byteVal materializes the byte value of a MOVB/ADDB-style source operand in a
// register whose bits 7:0 hold the value (higher bits may be garbage unless the
// operand was a high-byte register, an immediate, or memory, which are cleanly
// extracted/loaded into scratch). Callers must consume only bits 7:0.
func (p *arm64) byteVal(op operand.Op) string {
	if isHighByte(op) {
		p.emit("UBFX $8, %s, $8, %s", rename(op.(reg.Register)), scratchVal)
		return scratchVal
	}
	if imm, ok := immAsm(op); ok {
		p.emit("MOVD %s, %s", imm, scratchVal)
		return scratchVal
	}
	if m, ok := op.(operand.Mem); ok {
		p.emit("MOVBU %s, %s", p.memAsmW(m, 1), scratchVal)
		return scratchVal
	}
	return operandReg(op)
}

// lowerMOVB lowers x86 MOVB with exact partial-register semantics: a byte store
// writes one byte of memory; a register destination has only its addressed byte
// (bits 7:0, or 15:8 for AH/BH/CH/DH) replaced, via a bit-field insert, with
// every other bit preserved.
func (p *arm64) lowerMOVB(src, dst operand.Op) {
	if dmem, ok := dst.(operand.Mem); ok {
		v := p.byteVal(src)
		p.emit("MOVB %s, %s", v, p.memAsmW(dmem, 1))
		return
	}
	v := p.byteVal(src)
	lsb := 0
	if isHighByte(dst) {
		lsb = 8
	}
	p.emit("BFI $%d, %s, $8, %s", lsb, v, operandReg(dst))
}

// lowerMOVW lowers a 16-bit move. Stores write 16 bits of memory. Loads
// zero-extend (Go's MOVH would sign-extend, inventing high bits x86 never
// writes); this is exact when the destination's upper 48 bits are zero or
// unread, the only patterns the generators use. Register-to-register moves
// and immediates insert into bits 15:0, preserving the rest, as on x86.
func (p *arm64) lowerMOVW(src, dst operand.Op) {
	if dmem, ok := dst.(operand.Mem); ok {
		if imm, ok := immAsm(src); ok {
			p.emit("MOVD %s, %s", imm, scratchVal)
			p.emit("MOVH %s, %s", scratchVal, p.memAsmW(dmem, 2))
			return
		}
		p.emit("MOVH %s, %s", operandReg(src), p.memAsmW(dmem, 2))
		return
	}
	if smem, ok := src.(operand.Mem); ok {
		p.emit("MOVHU %s, %s", p.memAsmW(smem, 2), operandReg(dst))
		return
	}
	v := operandReg(dst)
	if imm, ok := immAsm(src); ok {
		p.emit("MOVD %s, %s", imm, scratchVal)
		p.emit("BFI $0, %s, $16, %s", scratchVal, v)
		return
	}
	p.emit("BFI $0, %s, $16, %s", operandReg(src), v)
}

// accessWidth is the width in bytes a Go arm64 load/store mnemonic touches,
// which determines whether an index may be scaled in a folded address operand.
func accessWidth(op string) int {
	switch op {
	case "MOVD":
		return 8
	case "MOVW", "MOVWU":
		return 4
	case "MOVH", "MOVHU":
		return 2
	case "MOVB", "MOVBU":
		return 1
	}
	return 0
}

// lowerMOVL lowers a 32-bit move. x86 MOVL zero-extends a register destination
// to 64 bits, so loads and register-to-register moves use MOVWU (zero-extend);
// Go arm64 MOVW would sign-extend. Stores write the low 32 bits.
func (p *arm64) lowerMOVL(src, dst operand.Op) {
	if dmem, ok := dst.(operand.Mem); ok {
		if imm, ok := immAsm(src); ok {
			p.emit("MOVD %s, %s", imm, scratchVal)
			p.emit("MOVW %s, %s", scratchVal, p.memAsmW(dmem, 4))
			return
		}
		p.emit("MOVW %s, %s", operandReg(src), p.memAsmW(dmem, 4))
		return
	}
	if smem, ok := src.(operand.Mem); ok {
		p.emit("MOVWU %s, %s", p.memAsmW(smem, 4), operandReg(dst))
		return
	}
	if imm, ok := immAsm(src); ok {
		// Immediate is <=32-bit unsigned; MOVD leaves the upper 32 bits zero.
		p.emit("MOVD %s, %s", imm, operandReg(dst))
		return
	}
	p.emit("MOVWU %s, %s", operandReg(src), operandReg(dst))
}

// lowerMOVUPS lowers a 16-byte unaligned move between memory and a vector reg.
func (p *arm64) lowerMOVUPS(src, dst operand.Op) {
	// FMOVQ moves 128 bits and, unlike VLD1/VST1, takes a base+displacement
	// operand directly. That keeps the common "MOVOU disp(base), X" a single
	// instruction instead of materializing the address into a scratch register
	// first, which matters in the literal-copy paths where 16-byte moves are
	// dense. memAsm still folds an indexed operand into scratchAddr, and FMOVQ
	// accepts that plain "(reg)" form too. F<n> and V<n> name the same physical
	// register, so this interoperates with the VEOR/VMOV forms used elsewhere.
	if dmem, ok := dst.(operand.Mem); ok {
		p.emit("FMOVQ %s, %s", vecAsF(src), p.memAsm(dmem))
		return
	}
	if smem, ok := src.(operand.Mem); ok {
		p.emit("FMOVQ %s, %s", p.memAsm(smem), vecAsF(dst))
		return
	}
	p.emit("VMOV %s.B16, %s.B16", operandReg(src), operandReg(dst))
}

// vecAsF renders a vector register under its F name, which the scalar FMOVQ
// form expects (F<n> and V<n> are the same physical register).
func vecAsF(op operand.Op) string {
	name := operandReg(op)
	if len(name) > 1 && name[0] == 'V' {
		return "F" + name[1:]
	}
	panic(fmt.Sprintf("arm64: expected a vector register, got %q", name))
}

// lowerArith lowers "OP src, dst" (dst op= src). dst may be a register or memory
// (read-modify-write via scratch). If flags is set, the flag-setting variant
// (sop) is used so a following branch/CMOV can consume NZCV.
func (p *arm64) lowerArith(op, sop string, src, dst operand.Op, flags bool) {
	mnem := op
	synth := ""
	if flags {
		if sop == "" {
			// ORR/EOR have no arm64 flag-setting form. x86's logical ops clear
			// CF and OF and set only ZF/SF meaningfully, so a following TST of
			// the result reproduces every condition that can legitimately be
			// read here; flagProducers rejects any other consumer.
			synth = "TST"
		} else {
			mnem = sop
		}
	}
	if dmem, ok := dst.(operand.Mem); ok {
		// Read-modify-write; src is a register or immediate (never memory too).
		s := p.regOrImm(src)
		m := p.memAsm(dmem)
		p.emit("MOVD %s, %s", m, scratchVal)
		p.emit("%s %s, %s, %s", mnem, s, scratchVal, scratchVal)
		p.emit("MOVD %s, %s", scratchVal, m)
		if synth != "" {
			p.emit("TST %s, %s", scratchVal, scratchVal)
		}
		return
	}
	d := operandReg(dst)
	var s string
	if imm, ok := immAsm(src); ok {
		s = imm
	} else {
		s = p.valReg(src) // loads memory source into scratch if needed
	}
	p.emit("%s %s, %s, %s", mnem, s, d, d)
	if synth != "" {
		p.emit("TST %s, %s", d, d)
	}
}

// lowerIncDec lowers INC/DEC of a register or memory operand by 1.
// lowerArithW is lowerArith for 32-bit operations. Register destinations use
// the arm64 W-forms, which zero the upper 32 bits exactly as x86 does; memory
// destinations are read-modify-written at 32-bit width.
func (p *arm64) lowerArithW(op, sop string, src, dst operand.Op, flags bool) {
	mnem := op
	synth := ""
	if flags {
		if sop == "" {
			synth = "TSTW" // see lowerArith
		} else {
			mnem = sop
		}
	}
	if dmem, ok := dst.(operand.Mem); ok {
		s := p.regOrImm(src)
		m := p.memAsm(dmem)
		p.emit("MOVWU %s, %s", m, scratchVal)
		p.emit("%s %s, %s, %s", mnem, s, scratchVal, scratchVal)
		p.emit("MOVW %s, %s", scratchVal, m)
		if synth != "" {
			p.emit("TSTW %s, %s", scratchVal, scratchVal)
		}
		return
	}
	d := operandReg(dst)
	var s string
	if imm, ok := immAsm(src); ok {
		s = imm
	} else if m, ok := src.(operand.Mem); ok {
		p.emit("MOVWU %s, %s", p.memAsm(m), scratchVal)
		s = scratchVal
	} else {
		s = operandReg(src)
	}
	p.emit("%s %s, %s, %s", mnem, s, d, d)
	if synth != "" {
		p.emit("TSTW %s, %s", d, d)
	}
}

func (p *arm64) lowerIncDec(op, sop string, dst operand.Op, flags bool) {
	mnem := op
	if flags {
		mnem = sop
	}
	if dmem, ok := dst.(operand.Mem); ok {
		m := p.memAsm(dmem)
		p.emit("MOVD %s, %s", m, scratchVal)
		p.emit("%s $1, %s, %s", mnem, scratchVal, scratchVal)
		p.emit("MOVD %s, %s", scratchVal, m)
		return
	}
	d := operandReg(dst)
	p.emit("%s $1, %s, %s", mnem, d, d)
}

// lowerShift lowers "SHIFT count, dst" (count imm or register).
func (p *arm64) lowerShift(op string, count, dst operand.Op) {
	d := operandReg(dst)
	p.emit("%s %s, %s, %s", op, p.regOrImm(count), d, d)
}

// lowerROL lowers "ROLQ count, dst" using ROR by the two's-complement count
// (ROR by (64-count) == ROL by count; arm64 ROR uses the low 6 bits).
func (p *arm64) lowerROL(count, dst operand.Op) {
	d := operandReg(dst)
	if imm, ok := immAsm(count); ok {
		var n int
		if _, err := fmt.Sscanf(imm, "$%d", &n); err != nil {
			panic("arm64: bad ROL immediate " + imm)
		}
		p.emit("ROR $%d, %s, %s", (64-n)&63, d, d)
		return
	}
	p.emit("NEG %s, %s", operandReg(count), scratchVal)
	p.emit("ROR %s, %s, %s", scratchVal, d, d)
}

// srcRegInto returns a register name holding op's value, loading a memory operand
// into the given scratch register first. Callers pass a scratch that is free for
// the remainder of the lowering.
func (p *arm64) srcRegInto(op operand.Op, scratch string) string {
	if m, ok := op.(operand.Mem); ok {
		p.emit("MOVD %s, %s", p.memAsm(m), scratch)
		return scratch
	}
	return operandReg(op)
}

// lowerShiftX lowers a BMI2 flag-free shift "SHIFTX count, src, dst":
// dst = src <shift> count. The count register is masked mod 64, as on x86.
// A count register just loaded with a constant folds to an immediate shift.
func (p *arm64) lowerShiftX(op string, count, src, dst operand.Op) {
	s := p.srcRegInto(src, scratchVal)
	if n, ok := p.knownConst(count); ok {
		p.emit("%s $%d, %s, %s", op, n&63, s, operandReg(dst))
		return
	}
	p.emit("%s %s, %s, %s", op, operandReg(count), s, operandReg(dst))
}

// lowerRORX lowers "RORXQ imm, src, dst": dst = ror(src, imm) (flag-free).
func (p *arm64) lowerRORX(imm, src, dst operand.Op) {
	c, ok := immAsm(imm)
	if !ok {
		panic("arm64: RORXQ requires an immediate rotate")
	}
	n, err := strconv.ParseInt(strings.TrimPrefix(c, "$"), 0, 64)
	if err != nil {
		panic("arm64: bad RORX immediate " + c)
	}
	s := p.srcRegInto(src, scratchVal)
	p.emit("ROR $%d, %s, %s", n&63, s, operandReg(dst))
}

// lowerBZHI lowers "BZHIQ count, src, dst": dst = src & ((1<<count)-1). count is
// a register bit-count; zstd only uses counts < 64, for which the mask is exact
// (arm64 shifts mask the amount mod 64, so count == 64 is not handled). When the
// count register was just loaded with a constant, the mask is folded to an
// immediate AND (with x86's full n==0 / n>=64 semantics, which are exact).
func (p *arm64) lowerBZHI(count, src, dst operand.Op) {
	d := operandReg(dst)
	if n, ok := p.knownConst(count); ok {
		s := p.srcRegInto(src, d)
		switch {
		case n <= 0:
			p.emit("MOVD $0, %s", d)
		case n >= 64:
			if s != d {
				p.emit("MOVD %s, %s", s, d)
			}
		default:
			p.emit("AND $%d, %s, %s", (int64(1)<<uint(n))-1, s, d)
		}
		return
	}
	s := p.srcRegInto(src, scratchAddr)
	n := operandReg(count)
	p.emit("MOVD $1, %s", scratchVal)
	p.emit("LSL %s, %s, %s", n, scratchVal, scratchVal) // 1 << count
	p.emit("SUB $1, %s, %s", scratchVal, scratchVal)    // mask = (1<<count)-1
	p.emit("AND %s, %s, %s", scratchVal, s, d)
}

// lowerBEXTR lowers "BEXTRQ ctrl, src, dst": with ctrl[7:0]=start and
// ctrl[15:8]=len, dst = (src >> start) & ((1<<len)-1). start and len come from the
// ctrl register at run time; zstd's control values keep both < 64. start and len
// are extracted before any destination write so ctrl may alias dst. When the
// ctrl register was just loaded with a constant, the extract folds to a single
// UBFX (or LSR when the field reaches bit 63, or a zero move when empty).
func (p *arm64) lowerBEXTR(ctrl, src, dst operand.Op) {
	d := operandReg(dst)
	if c, ok := p.knownConst(ctrl); ok {
		start := c & 0xff
		length := (c >> 8) & 0xff
		s := p.srcRegInto(src, d)
		switch {
		case length == 0 || start >= 64:
			p.emit("MOVD $0, %s", d)
		case start+length >= 64:
			p.emit("LSR $%d, %s, %s", start, s, d)
		default:
			p.emit("UBFX $%d, %s, $%d, %s", start, s, length, d)
		}
		return
	}
	c := operandReg(ctrl)
	p.emit("UBFX $0, %s, $8, %s", c, scratchVal)  // start = ctrl[7:0]
	p.emit("UBFX $8, %s, $8, %s", c, scratchAddr) // len   = ctrl[15:8]
	s := p.srcRegInto(src, d)
	p.emit("LSR %s, %s, %s", scratchVal, s, d)                    // dst = src >> start
	p.emit("MOVD $1, %s", scratchVal)                             // start consumed; reuse
	p.emit("LSL %s, %s, %s", scratchAddr, scratchVal, scratchVal) // 1 << len
	p.emit("SUB $1, %s, %s", scratchVal, scratchVal)              // (1<<len)-1
	p.emit("AND %s, %s, %s", scratchVal, d, d)                    // dst &= mask
}

// lowerMULX lowers "MULXQ src, lo, hi" (BMI2, flag-free): the 128-bit product
// src * RDX has its low half written to lo and its high half to hi. The low half
// is staged in scratch so lo/hi may alias src or RDX.
func (p *arm64) lowerMULX(src, lo, hi operand.Op) {
	dx := rename(reg.RDX)
	s := p.srcRegInto(src, scratchAddr)
	p.emit("MUL %s, %s, %s", s, dx, scratchVal)       // low  -> scratch
	p.emit("UMULH %s, %s, %s", s, dx, operandReg(hi)) // high -> hi (s, dx intact)
	p.emit("MOVD %s, %s", scratchVal, operandReg(lo)) // low  -> lo
}

// lowerIMUL3 lowers "IMUL3Q imm, src, dst": dst = src * imm. x86 sign-extends the
// imm8/imm32 multiplier to 64 bits, so replicate that — otherwise a constant with
// bit 31 set (e.g. the 0x9E3779B1 golden-ratio multiplier) would differ from the
// amd64 result.
func (p *arm64) lowerIMUL3(imm, src, dst operand.Op) {
	c, ok := immAsm(imm)
	if !ok {
		panic("arm64: IMUL3Q requires an immediate multiplier")
	}
	v, err := strconv.ParseInt(strings.TrimPrefix(c, "$"), 0, 64)
	if err != nil {
		panic("arm64: bad IMUL3Q immediate " + c)
	}
	s := p.srcRegInto(src, scratchAddr)
	p.emit("MOVD $%d, %s", int64(int32(v)), scratchVal) // materialize sign-extended imm32
	p.emit("MUL %s, %s, %s", scratchVal, s, operandReg(dst))
}

// lowerIMUL lowers IMULQ in its 2-operand ("src, dst": dst *= src) and 1-operand
// ("src": RDX:RAX = RAX * src, signed) forms.
func (p *arm64) lowerIMUL(ops []operand.Op) {
	switch len(ops) {
	case 2:
		s := p.valReg(ops[0])
		d := operandReg(ops[1])
		p.emit("MUL %s, %s, %s", s, d, d)
	case 1:
		p.lowerWideMul("SMULH", ops[0])
	default:
		panic(fmt.Sprintf("arm64: IMULQ with %d operands not supported", len(ops)))
	}
}

// lowerWideMul lowers the single-operand MULQ/IMULQ: RDX:RAX = RAX * src, with
// hiOp = UMULH (unsigned) or SMULH (signed). The source is staged when it aliases
// RDX so writing the result cannot clobber it.
func (p *arm64) lowerWideMul(hiOp string, src operand.Op) {
	rax := rename(reg.RAX)
	rdx := rename(reg.RDX)
	s := p.srcRegInto(src, scratchVal)
	if s == rdx {
		p.emit("MOVD %s, %s", s, scratchVal)
		s = scratchVal
	}
	p.emit("%s %s, %s, %s", hiOp, s, rax, rdx) // RDX = high(RAX*src)
	p.emit("MUL %s, %s, %s", s, rax, rax)      // RAX = low(RAX*src)
}

func (p *arm64) lowerLEA(m operand.Mem, dst string) {
	if m.Symbol.Name != "" {
		panic("arm64: LEA of symbol not supported")
	}
	base := rename(m.Base)
	if m.Index != nil && m.Scale != 0 {
		sh := log2scale(m.Scale)
		if sh == 0 {
			p.emit("ADD %s, %s, %s", rename(m.Index), base, dst)
		} else {
			p.emit("ADD %s<<%d, %s, %s", rename(m.Index), sh, base, dst)
		}
		base = dst
	}
	if m.Disp != 0 {
		if m.Disp > 0 {
			p.emit("ADD $%d, %s, %s", m.Disp, base, dst)
		} else {
			p.emit("SUB $%d, %s, %s", -m.Disp, base, dst)
		}
		return
	}
	if base != dst {
		p.emit("MOVD %s, %s", base, dst)
	}
}

// lowerCompare emits an arm64 flag-setter for "CMP a, b" (flags from a - b),
// using the given compare mnemonic (cmp) and its negated-immediate counterpart
// (cmn), so callers select the operand width: CMP/CMN for 64-bit, CMPW/CMNW for
// 32-bit. At most one of a, b is a memory operand (loaded into scratchVal).
func (p *arm64) lowerCompare(cmp, cmn string, a, b operand.Op) {
	if imm, ok := immAsm(b); ok {
		aReg := p.valReg(a)
		if neg, val := negImm(imm); neg {
			p.emit("%s $%d, %s", cmn, val, aReg)
			return
		}
		p.emit("%s %s, %s", cmp, imm, aReg)
		return
	}
	// arm64 CMP Rm, Rn computes Rn - Rm; we want a - b, so Rn=a, Rm=b.
	if _, ok := a.(operand.Mem); ok {
		aReg := p.valReg(a)
		p.emit("%s %s, %s", cmp, operandReg(b), aReg)
		return
	}
	p.emit("%s %s, %s", cmp, p.valReg(b), operandReg(a))
}

// lowerTest emits an arm64 bitwise-test flag-setter for "TEST a, b" using the
// given mnemonic (TST for 64-bit, TSTW for 32-bit).
func (p *arm64) lowerTest(op string, a, b operand.Op) {
	aReg := p.valReg(a)
	p.emit("%s %s, %s", op, p.regOrImm(b), aReg)
}

// lowerSubwordCompareEqNe lowers a sub-32-bit CMP (bits is 8 or 16) whose only
// consumers are EQ/NE (see subwordSafeEqNe), by zero-extending both operands
// to the compared width before a full-width compare. Correct unconditionally:
// equality doesn't depend on sign, so no proof that the operands were already
// clean above that width is needed, unlike the general sub-word compare case.
func (p *arm64) lowerSubwordCompareEqNe(bits int, cmp string, a, b operand.Op) {
	ra, rb := p.materializeEqNe(bits, a, b)
	// arm64 CMP Rm, Rn computes Rn - Rm; we only need Z, so the operand order
	// does not matter here, but keep it consistent with lowerCompare (a - b).
	p.emit("%s %s, %s", cmp, rb, ra)
}

// lowerSubwordTestEqNe lowers a sub-32-bit TEST (bits is 8 or 16) whose only
// consumers are EQ/NE (i.e. only the Z flag is read), by zero-extending both
// operands to the tested width before a full-width TST. Correct
// unconditionally: with both operands clean above that width, the AND's upper
// bits are always zero, so Z exactly reflects whether the low bits are all
// zero, which is what EQ/NE tests.
func (p *arm64) lowerSubwordTestEqNe(bits int, a, b operand.Op) {
	ra, rb := p.materializeEqNe(bits, a, b)
	p.emit("TST %s, %s", rb, ra)
}

// zeroExtendEqNe materializes op, zero-extended to the given bit width (8 or
// 16), into the given scratch register and returns its name.
//
// A memory operand needs no masking: the zero-extending load of that exact
// width already produces the value. Callers must materialize a memory operand
// before any other operand, because computing an indexed address clobbers
// scratchAddr (see lowerSubwordCompareEqNe).
func (p *arm64) zeroExtendEqNe(bits int, op operand.Op, scratch string) string {
	mask := fmt.Sprintf("$0x%x", uint64(1)<<uint(bits)-1)
	if imm, ok := immAsm(op); ok {
		p.emit("MOVD %s, %s", imm, scratch)
		p.emit("AND %s, %s, %s", mask, scratch, scratch)
		return scratch
	}
	if m, ok := op.(operand.Mem); ok {
		ld := "MOVBU"
		if bits == 16 {
			ld = "MOVHU"
		}
		p.emit("%s %s, %s", ld, p.memAsm(m), scratch)
		return scratch
	}
	p.emit("AND %s, %s, %s", mask, operandReg(op), scratch)
	return scratch
}

// materializeEqNe zero-extends both operands of a sub-word compare/test into
// the two scratch registers. x86 permits at most one memory operand, and that
// one is materialized first: its address computation may use scratchAddr, which
// would otherwise clobber a value already placed there.
func (p *arm64) materializeEqNe(bits int, a, b operand.Op) (ra, rb string) {
	if _, aIsMem := a.(operand.Mem); aIsMem {
		ra = p.zeroExtendEqNe(bits, a, scratchAddr)
		rb = p.zeroExtendEqNe(bits, b, scratchVal)
		return ra, rb
	}
	rb = p.zeroExtendEqNe(bits, b, scratchVal)
	ra = p.zeroExtendEqNe(bits, a, scratchAddr)
	return ra, rb
}

// lowerSET lowers "SETcc dst". x86 SETcc writes only the low byte of dst, so
// the 0/1 is materialized with CSET in scratch and inserted into bits 7:0.
func (p *arm64) lowerSET(i *ir.Instruction) {
	cc, ok := armCond(strings.TrimPrefix(i.Opcode, "SET"))
	if !ok {
		panic(fmt.Sprintf("arm64: unsupported SETcc %q", i.Opcode))
	}
	if isHighByte(i.Operands[0]) {
		panic("arm64: SETcc high-byte destination not supported")
	}
	p.emit("CSET %s, %s", cc, scratchVal)
	p.emit("BFI $0, %s, $8, %s", scratchVal, operandReg(i.Operands[0]))
}

// lowerCMOV lowers "CMOVcc src, dst" to "CSEL cc, src, dst, dst".
func (p *arm64) lowerCMOV(i *ir.Instruction) {
	cond := cmovCond(i.Opcode)
	src := operandReg(i.Operands[0])
	dst := operandReg(i.Operands[1])
	p.emit("CSEL %s, %s, %s, %s", cond, src, dst, dst)
}

func negImm(imm string) (bool, int) {
	var v int
	if _, err := fmt.Sscanf(imm, "$%d", &v); err != nil {
		return false, 0
	}
	if v < 0 {
		return true, -v
	}
	return false, 0
}

func isConditionalBranch(i *ir.Instruction) bool {
	return i.Opcode != "JMP" && strings.HasPrefix(i.Opcode, "J")
}

// armCond maps an x86/Go condition suffix to the arm64 condition mnemonic shared
// by B.cond and CSEL. Mapping is by meaning: x86 and arm64 use opposite
// carry-flag conventions for subtraction, but the named conditions encode the
// intent, not the raw flag, so the same-named arm64 condition is correct.
func armCond(cc string) (string, bool) {
	switch cc {
	case "EQ", "E", "Z":
		return "EQ", true
	case "NE", "NZ":
		return "NE", true
	case "LT", "L":
		return "LT", true // signed <
	case "LE":
		return "LE", true // signed <=
	case "GT", "G":
		return "GT", true // signed >
	case "GE":
		return "GE", true // signed >=
	case "CS", "B", "LO", "NAE":
		return "LO", true // unsigned <
	case "CC", "AE", "HS", "NB":
		return "HS", true // unsigned >=
	case "HI", "A", "NBE":
		return "HI", true // unsigned >
	case "LS", "BE", "NA":
		return "LS", true // unsigned <=
	case "MI", "S":
		return "MI", true // negative
	case "PL", "NS":
		return "PL", true // non-negative
	case "OS":
		return "VS", true // overflow
	case "OC":
		return "VC", true // no overflow
	}
	return "", false
}

// branchMnemonic maps an x86 conditional-jump opcode (Go-canonical name or Intel
// alias) to the arm64 conditional branch with the same meaning.
func branchMnemonic(op string) string {
	if cc, ok := armCond(strings.TrimPrefix(op, "J")); ok {
		return "B" + cc
	}
	panic(fmt.Sprintf("arm64: unsupported conditional branch %q", op))
}

// cmovCond maps an x86 CMOVcc opcode to its arm64 CSEL condition, for the Q/L/W
// operand-size prefixes.
func cmovCond(op string) string {
	cc := op
	for _, prefix := range []string{"CMOVQ", "CMOVL", "CMOVW"} {
		if strings.HasPrefix(op, prefix) {
			cc = op[len(prefix):]
			break
		}
	}
	if a, ok := armCond(cc); ok {
		return a
	}
	panic(fmt.Sprintf("arm64: unsupported CMOV %q", op))
}

// flagProducers identifies, for every flag consumer (a conditional branch or a
// CMOVcc), the instruction that produces the NZCV it reads, and returns the set
// of node indices whose lowering must therefore emit a flag-setting variant.
//
// avo's IR does not model EFLAGS, so the producer/consumer link is recovered
// structurally: scanning back from a consumer, comments and flag-transparent
// instructions (moves, address computations, and other branches/CMOVs — which
// read flags but never write them) are skipped, and the first flag-affecting
// instruction is the producer. CMP*/TEST* already emit an unconditional
// flag-setter and need no mark; a producer whose lowering cannot carry flags
// (e.g. ORQ/XORQ/shifts) is unsupported and panics rather than let a branch run
// on stale flags. Reaching a label or the start of the block means the flags
// cross a control-flow edge, which this printer does not model; such a consumer
// is left unmarked (matching the previous behaviour).
//
// A single-instruction lookahead is insufficient because x86 permits
// flag-transparent instructions (a MOV, an LEA) between a producer and the
// branch that consumes it; those must be skipped, not treated as the producer.
func flagProducers(nodes []ir.Node) map[int]bool {
	setflags := make(map[int]bool)
	for j, n := range nodes {
		ins, ok := n.(*ir.Instruction)
		if !ok || !(strings.HasPrefix(ins.Opcode, "CMOV") || strings.HasPrefix(ins.Opcode, "SET") ||
			strings.HasPrefix(ins.Opcode, "ADC") || isConditionalBranch(ins)) {
			continue
		}
		for k := j - 1; k >= 0; k-- {
			if _, isComment := nodes[k].(*ir.Comment); isComment {
				continue
			}
			prev, isInstr := nodes[k].(*ir.Instruction)
			if !isInstr {
				break // label or other boundary: producer not in this straight-line run
			}
			if isFlagTransparent(prev.Opcode) {
				continue
			}
			mark, ok := flagSetter(prev.Opcode)
			if !ok {
				panic(fmt.Sprintf("arm64: %s consumes flags from %s, which the lowering cannot emit as a flag-setter", ins.Opcode, prev.Opcode))
			}
			if isLogicalFlagOp(prev.Opcode) {
				// These lower to op+TST, which reproduces ZF/SF but not CF/OF.
				// x86 logical ops clear CF/OF, so a carry/overflow consumer here
				// would be reading a constant -- almost certainly a bug, and not
				// something the TST substitution can express.
				if cond, isConsumer := consumerCondition(ins.Opcode); isConsumer {
					switch cond {
					case "EQ", "NE", "MI", "PL":
					default:
						panic(fmt.Sprintf("arm64: %s consumes condition %s from %s, but the logical-op lowering only reproduces ZF/SF", ins.Opcode, cond, prev.Opcode))
					}
				}
			}
			if mark {
				setflags[k] = true
			}
			break
		}
	}
	return setflags
}

// isFlagTransparent reports whether an opcode's lowering leaves NZCV unchanged.
// Moves and address computations never touch flags; conditional branches and
// CMOVcc read flags but do not write them, so they too are transparent to an
// earlier producer's flags.
func isFlagTransparent(op string) bool {
	switch op {
	// BMI2 flag-free variants (shifts and wide multiply) deliberately leave flags
	// untouched, unlike their non-BMI2 counterparts; their arm64 lowerings use
	// non-flag-setting ops too, so NZCV survives a producer across them.
	case "SHLXQ", "SHRXQ", "SARXQ", "RORXQ", "MULXQ":
		return true
	}
	switch {
	case strings.HasPrefix(op, "MOV"),
		strings.HasPrefix(op, "LEA"),
		strings.HasPrefix(op, "CMOV"),
		strings.HasPrefix(op, "SET"), // reads flags, never writes them
		strings.HasPrefix(op, "J"):   // JMP and the Jcc family
		return true
	}
	return false
}

// flagSetter classifies a flag-affecting opcode. mark is true when its lowering
// must be switched to a flag-setting variant; false when it already emits a
// setter unconditionally (CMP*/TEST*). ok is false for flag-affecting opcodes
// this printer cannot lower as a flag-setter, so callers fail loudly instead of
// silently branching on stale flags.
func flagSetter(op string) (mark, ok bool) {
	switch op {
	case "ADDQ", "SUBQ", "ANDQ", "INCQ", "DECQ",
		"ADDL", "SUBL", "ANDL", "INCL", "DECL":
		return true, true
	case "ORQ", "XORQ", "ORL", "XORL":
		// No arm64 ORRS/EORS; lowerArith appends a TST instead. Only valid for
		// Z/N-based consumers, which flagProducers verifies.
		return true, true
	case "CMPQ", "CMPL", "CMPW", "CMPB", "TESTQ", "TESTL", "TESTW", "TESTB":
		return false, true
	default:
		return false, false
	}
}

// isLogicalFlagOp reports whether an opcode is a bitwise OR/XOR, whose arm64
// lowering has no flag-setting form and instead appends a TST (see lowerArith).
func isLogicalFlagOp(op string) bool {
	switch op {
	case "ORQ", "XORQ", "ORL", "XORL":
		return true
	}
	return false
}

// isSubwordCompare reports whether opcode is a sub-32-bit CMP/TEST, the only
// class this printer treats as conditionally supported (see subwordSafeEqNe).
func isSubwordCompare(opcode string) bool {
	switch opcode {
	case "CMPW", "CMPB", "TESTW", "TESTB":
		return true
	}
	return false
}

// consumerCondition returns the arm64 condition an instruction consumes flags
// under, if it is a CMOVcc, SETcc, ADC (carry), or a conditional branch.
func consumerCondition(opcode string) (string, bool) {
	if strings.HasPrefix(opcode, "CMOV") {
		cc := opcode
		for _, prefix := range []string{"CMOVQ", "CMOVL", "CMOVW"} {
			if strings.HasPrefix(opcode, prefix) {
				cc = opcode[len(prefix):]
				break
			}
		}
		return armCond(cc)
	}
	if strings.HasPrefix(opcode, "SET") {
		return armCond(strings.TrimPrefix(opcode, "SET"))
	}
	if strings.HasPrefix(opcode, "ADC") {
		return "LO", true // consumes the carry (x86 borrow after a compare)
	}
	if opcode != "JMP" && strings.HasPrefix(opcode, "J") {
		return armCond(strings.TrimPrefix(opcode, "J"))
	}
	return "", false
}

// subwordSafeEqNe identifies, for every straight-line CMPW/CMPB/TESTW/TESTB
// producer, whether every consumer that reads its flags uses only EQ/NE.
// Sub-word CMP/TEST are otherwise unsupported (see lower()) because ordering
// conditions need sign-aware operand extension this printer does not model;
// equality doesn't depend on sign, so this narrow case can be lowered safely
// regardless of the surrounding dataflow (see lowerSubwordCompareEqNe).
//
// The scan mirrors flagProducers but runs forward: from each producer, walk
// over flag-transparent instructions (CMOVcc/Jcc chain onto the same flags)
// collecting every consumer found, until a non-flag-transparent instruction,
// a label, or the end of the block. A producer with multiple consumers (e.g.
// two chained conditional branches) requires ALL of them to be EQ/NE.
func subwordSafeEqNe(nodes []ir.Node) map[int]bool {
	safe := make(map[int]bool)
	for j, n := range nodes {
		ins, ok := n.(*ir.Instruction)
		if !ok || !isSubwordCompare(ins.Opcode) {
			continue
		}
		eqne, any := true, false
	consumers:
		for k := j + 1; k < len(nodes); k++ {
			switch next := nodes[k].(type) {
			case *ir.Comment:
				continue
			case *ir.Instruction:
				if cond, isConsumer := consumerCondition(next.Opcode); isConsumer {
					any = true
					if cond != "EQ" && cond != "NE" {
						eqne = false
					}
					continue // flag-transparent: keep scanning for chained consumers
				}
				any = true // another flag-affecting instruction: this producer's flags are dead
				break consumers
			default:
				break consumers // label: producer/consumer link does not cross blocks
			}
		}
		safe[j] = any && eqne
	}
	return safe
}
