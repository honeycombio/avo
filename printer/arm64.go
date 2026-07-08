package printer

import (
	"fmt"
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
	for _, s := range f.Sections {
		switch s := s.(type) {
		case *ir.Function:
			p.function(s)
		case *ir.Global:
			p.global(s)
		default:
			panic("unknown section type")
		}
	}
	return p.Result()
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

func (p *arm64) function(f *ir.Function) {
	// arm64 has no BMI2; skip those variants. The generator emits both the
	// generic-instruction ("_amd64") and BMI2 ("_bmi2") variants; arm64 uses the
	// generic ones, renamed with an _arm64 suffix.
	if strings.Contains(f.Name, "bmi2") {
		p.NL()
		p.Comment("skipped " + f.Name + " (BMI2 not available on arm64)")
		return
	}
	name := strings.Replace(f.Name, "_amd64", "_arm64", 1)

	p.NL()
	p.Comment(f.Stub())
	if len(f.ISA) > 0 {
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
	for idx := 0; idx < len(nodes); idx++ {
		switch n := nodes[idx].(type) {
		case ir.Label:
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
			case isConditionalBranch(n):
				p.emit("%s %s", branchMnemonic(n.Opcode), n.Operands[0].Asm())
			default:
				p.lower(n, setflags[idx])
			}
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
func (p *arm64) memAsm(m operand.Mem) string {
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

func (p *arm64) lower(i *ir.Instruction, flags bool) {
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
		p.lowerMove("MOVH", ops[0], ops[1])
	case "MOVB":
		if isHighByte(ops[0]) {
			// MOVB AH, dst : read bits 8-15 of the source register. arm64 has no
			// high-byte register, so extract the byte explicitly (zero-extended).
			if _, ok := ops[1].(operand.Mem); ok {
				panic("arm64: MOVB high-byte to memory not supported")
			}
			p.emit("UBFX $8, %s, $8, %s", rename(ops[0].(reg.Register)), operandReg(ops[1]))
			return
		}
		p.lowerMove("MOVB", ops[0], ops[1])
	case "MOVWQSX": // load/extend int16, sign-extend (mem or reg source)
		p.emit("MOVH %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVWQZX": // load/extend uint16, zero-extend
		p.emit("MOVHU %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVBQZX": // load/extend uint8, zero-extend
		p.emit("MOVBU %s, %s", p.srcAsm(ops[0]), operandReg(ops[1]))
	case "MOVUPS":
		p.lowerMOVUPS(ops[0], ops[1])

	// ---- arithmetic / logic (dst is last operand) ----
	case "ADDQ":
		p.lowerArith("ADD", "ADDS", ops[0], ops[1], flags)
	case "SUBQ":
		p.lowerArith("SUB", "SUBS", ops[0], ops[1], flags)
	case "ANDQ":
		p.lowerArith("AND", "ANDS", ops[0], ops[1], flags)
	case "ORQ":
		p.lowerArith("ORR", "", ops[0], ops[1], flags)
	case "XORQ":
		if ra, ok := ops[0].(reg.Register); ok {
			if rb, ok := ops[1].(reg.Register); ok && rename(ra) == rename(rb) {
				p.emit("MOVD $0, %s", rename(rb))
				return
			}
		}
		p.lowerArith("EOR", "", ops[0], ops[1], flags)
	case "INCQ":
		p.lowerIncDec("ADD", "ADDS", ops[0], flags)
	case "DECQ":
		p.lowerIncDec("SUB", "SUBS", ops[0], flags)
	case "DECL":
		p.lowerIncDec("SUBW", "SUBSW", ops[0], flags)
	case "NEGQ":
		p.emit("NEG %s, %s", operandReg(ops[0]), operandReg(ops[0]))
	case "SHRQ":
		p.lowerShift("LSR", ops[0], ops[1])
	case "SHLQ":
		p.lowerShift("LSL", ops[0], ops[1])
	case "SHRL":
		p.lowerShift("LSRW", ops[0], ops[1])
	case "SHLL":
		p.lowerShift("LSLW", ops[0], ops[1])
	case "ROLQ":
		p.lowerROL(ops[0], ops[1])

	case "LEAQ":
		p.lowerLEA(ops[0].(operand.Mem), operandReg(ops[1]))

	case "BTSQ":
		a := operandReg(ops[0])
		b := operandReg(ops[1])
		p.emit("MOVD $1, %s", scratchVal)
		p.emit("LSL %s, %s, %s", a, scratchVal, scratchVal)
		p.emit("ORR %s, %s, %s", scratchVal, b, b)

	case "BSRQ":
		src := operandReg(ops[0])
		dst := operandReg(ops[1])
		p.emit("CLZ %s, %s", src, scratchVal)
		p.emit("MOVD $63, %s", dst)
		p.emit("SUB %s, %s, %s", scratchVal, dst, dst)

	// ---- comparisons: always emit an arm64 flag-setter ----
	// The compare must run at the operand width: a 64-bit CMP of registers whose
	// upper bits are not provably zero sets flags from the wrong bits. arm64 has
	// a native 32-bit form (CMPW/TSTW); sub-32-bit widths would need the operands
	// extended for the consuming condition's signedness, which is not modelled, so
	// they fail loudly rather than silently comparing full 64-bit registers.
	case "CMPQ":
		p.lowerCompare("CMP", "CMN", ops[0], ops[1])
	case "CMPL":
		p.lowerCompare("CMPW", "CMNW", ops[0], ops[1])
	case "CMPW", "CMPB":
		panic(fmt.Sprintf("arm64: %s not supported (sub-32-bit compare needs width- and sign-correct operand extension)", i.Opcode))
	case "TESTQ":
		p.lowerTest("TST", ops[0], ops[1])
	case "TESTL":
		p.lowerTest("TSTW", ops[0], ops[1])
	case "TESTW", "TESTB":
		panic(fmt.Sprintf("arm64: %s not supported (sub-32-bit test needs width-correct operands)", i.Opcode))

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
	dmem, dstIsMem := dst.(operand.Mem)
	smem, srcIsMem := src.(operand.Mem)
	switch {
	case dstIsMem:
		if imm, ok := immAsm(src); ok {
			p.emit("MOVD %s, %s", imm, scratchVal)
			p.emit("%s %s, %s", op, scratchVal, p.memAsm(dmem))
			return
		}
		p.emit("%s %s, %s", op, operandReg(src), p.memAsm(dmem))
	case srcIsMem:
		p.emit("%s %s, %s", op, p.memAsm(smem), operandReg(dst))
	default:
		if imm, ok := immAsm(src); ok {
			p.emit("%s %s, %s", op, imm, operandReg(dst))
			return
		}
		p.emit("%s %s, %s", op, operandReg(src), operandReg(dst))
	}
}

// lowerMOVL lowers a 32-bit move. x86 MOVL zero-extends a register destination
// to 64 bits, so loads and register-to-register moves use MOVWU (zero-extend);
// Go arm64 MOVW would sign-extend. Stores write the low 32 bits.
func (p *arm64) lowerMOVL(src, dst operand.Op) {
	if dmem, ok := dst.(operand.Mem); ok {
		if imm, ok := immAsm(src); ok {
			p.emit("MOVD %s, %s", imm, scratchVal)
			p.emit("MOVW %s, %s", scratchVal, p.memAsm(dmem))
			return
		}
		p.emit("MOVW %s, %s", operandReg(src), p.memAsm(dmem))
		return
	}
	if smem, ok := src.(operand.Mem); ok {
		p.emit("MOVWU %s, %s", p.memAsm(smem), operandReg(dst))
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
	if dmem, ok := dst.(operand.Mem); ok {
		addr := p.memAddr(dmem)
		p.emit("VST1 [%s.B16], (%s)", operandReg(src), addr)
		return
	}
	if smem, ok := src.(operand.Mem); ok {
		addr := p.memAddr(smem)
		p.emit("VLD1 (%s), [%s.B16]", addr, operandReg(dst))
		return
	}
	panic("arm64: MOVUPS register-to-register not supported")
}

// lowerArith lowers "OP src, dst" (dst op= src). dst may be a register or memory
// (read-modify-write via scratch). If flags is set, the flag-setting variant
// (sop) is used so a following branch/CMOV can consume NZCV.
func (p *arm64) lowerArith(op, sop string, src, dst operand.Op, flags bool) {
	mnem := op
	if flags {
		if sop == "" {
			panic(fmt.Sprintf("arm64: %q has no flag-setting variant but a flag consumer follows", op))
		}
		mnem = sop
	}
	if dmem, ok := dst.(operand.Mem); ok {
		// Read-modify-write; src is a register or immediate (never memory too).
		s := p.regOrImm(src)
		m := p.memAsm(dmem)
		p.emit("MOVD %s, %s", m, scratchVal)
		p.emit("%s %s, %s, %s", mnem, s, scratchVal, scratchVal)
		p.emit("MOVD %s, %s", scratchVal, m)
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
}

// lowerIncDec lowers INC/DEC of a register or memory operand by 1.
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

// branchMnemonic maps an x86 conditional-jump opcode to the arm64 branch with
// the same meaning. avo uses the Go-canonical names (JEQ/JCS/JHI/...); the Intel
// aliases are accepted too. Mapping by meaning is correct despite x86 and arm64
// using opposite carry-flag conventions for subtraction, because the condition
// names encode the intent, not the raw flag.
func branchMnemonic(op string) string {
	switch op {
	case "JEQ", "JE", "JZ":
		return "BEQ"
	case "JNE", "JNZ":
		return "BNE"
	case "JLT", "JL":
		return "BLT" // signed <
	case "JLE":
		return "BLE" // signed <=
	case "JGT", "JG":
		return "BGT" // signed >
	case "JGE":
		return "BGE" // signed >=
	case "JCS", "JB":
		return "BLO" // unsigned <
	case "JCC", "JAE", "JHS":
		return "BHS" // unsigned >=
	case "JHI", "JA":
		return "BHI" // unsigned >
	case "JLS", "JBE":
		return "BLS" // unsigned <=
	case "JMI", "JS":
		return "BMI" // negative
	case "JPL", "JNS":
		return "BPL" // non-negative
	case "JOS":
		return "BVS" // overflow
	case "JOC":
		return "BVC" // no overflow
	default:
		panic(fmt.Sprintf("arm64: unsupported conditional branch %q", op))
	}
}

// cmovCond maps an x86 CMOVcc opcode to its arm64 CSEL condition. Only EQ/NE are
// exercised by the zstd generators today; the rest are by-construction.
func cmovCond(op string) string {
	switch op {
	case "CMOVQEQ", "CMOVLEQ":
		return "EQ"
	case "CMOVQNE", "CMOVLNE":
		return "NE"
	case "CMOVQHI", "CMOVLHI":
		return "HI"
	case "CMOVQCC", "CMOVQHS", "CMOVLCC", "CMOVLHS":
		return "HS"
	case "CMOVQCS", "CMOVQLO", "CMOVLCS", "CMOVLLO":
		return "LO"
	default:
		panic(fmt.Sprintf("arm64: unsupported CMOV %q", op))
	}
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
		if !ok || !(strings.HasPrefix(ins.Opcode, "CMOV") || isConditionalBranch(ins)) {
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
	switch {
	case strings.HasPrefix(op, "MOV"),
		strings.HasPrefix(op, "LEA"),
		strings.HasPrefix(op, "CMOV"),
		strings.HasPrefix(op, "J"): // JMP and the Jcc family
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
	case "ADDQ", "SUBQ", "ANDQ", "INCQ", "DECQ", "DECL":
		return true, true
	case "CMPQ", "CMPL", "CMPW", "CMPB", "TESTQ", "TESTL", "TESTW", "TESTB":
		return false, true
	default:
		return false, false
	}
}
