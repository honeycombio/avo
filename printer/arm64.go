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
//     as scratch for address computation and immediate-to-memory stores. x86 has
//     <=14 allocatable GP registers, arm64 has far more, so no re-allocation is
//     needed. Pseudo registers (FP/SB/SP) pass through unchanged.
//
//   - Memory operands: x86 base+index*scale+disp has no arm64 equivalent, so an
//     indexed operand is lowered to "ADD scratch, base, index<<log2(scale)"
//     followed by a simple disp(scratch) access.
//
//   - Flags: avo's IR does not model EFLAGS. x86 sets flags implicitly and a
//     following Jcc consumes them. We fuse a comparison instruction
//     (CMP*/TEST*) with the immediately-following conditional branch into an
//     arm64 compare + Bcc (or CBZ/CBNZ). This is valid because the amd64
//     generators emit the producer and consumer adjacently.
//
// Unsupported opcodes panic loudly, so growing the supported set (e.g. to cover
// the seqdec generator) surfaces exactly what is missing.
type arm64 struct {
	cfg Config
	prnt.Generator
}

// NewARM64Asm constructs a printer for writing Go arm64 assembly files by
// lowering an amd64 avo instruction stream. EXPERIMENTAL.
func NewARM64Asm(cfg Config) Printer { return &arm64{cfg: cfg} }

// armReg maps an x86 physical GP register index to an arm64 register name.
// x86 indices: 0=AX 1=CX 2=DX 3=BX 4=SP 5=BP 6=SI 7=DI 8..15=R8..R15.
// SP/BP (4,5) are not used as allocatable GP by avo here.
var armReg = map[reg.Index]string{
	0: "R0", 1: "R1", 2: "R2", 3: "R3",
	6: "R4", 7: "R5",
	8: "R6", 9: "R7", 10: "R8", 11: "R9", 12: "R10", 13: "R11", 14: "R12", 15: "R13",
}

const (
	// Scratch registers reserved for lowering; never produced by the x86 map.
	scratchAddr = "R14" // effective-address computation for indexed operands
	scratchVal  = "R15" // immediate materialization for store-to-memory
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
	p.NL()
	p.Comment(f.Stub())
	if len(f.ISA) > 0 {
		p.Comment("Requires: " + strings.Join(f.ISA, ", "))
	}
	p.Printf("TEXT %s%s(SB)", dot, f.Name)
	if f.Attributes != 0 {
		p.Printf(", %s", f.Attributes.Asm())
	}
	p.Printf(", %s\n", textsize(f))

	nodes := f.Nodes
	for idx := 0; idx < len(nodes); idx++ {
		switch n := nodes[idx].(type) {
		case ir.Label:
			p.Printf("%s:\n", n)
		case *ir.Comment:
			for _, line := range n.Lines {
				p.Printf("\t// %s\n", line)
			}
		case *ir.Instruction:
			// Fuse a comparison with its consuming conditional branch.
			if isCompare(n) {
				if br, ok := nextInstruction(nodes, idx); ok && br.IsConditional {
					p.lowerCompareBranch(n, br)
					idx = skipTo(nodes, idx, br)
					continue
				}
			}
			p.lower(n)
		default:
			panic("unexpected node type")
		}
	}
}

// emit prints a single arm64 instruction line.
func (p *arm64) emit(format string, args ...interface{}) {
	p.Printf("\t"+format+"\n", args...)
}

// rename returns the arm64 syntax for a register operand.
func rename(r reg.Register) string {
	if r.Kind() == reg.KindGP {
		if ph, ok := r.(reg.Physical); ok {
			if name, ok := armReg[ph.PhysicalIndex()]; ok {
				return name
			}
			panic(fmt.Sprintf("arm64: unmapped x86 GP register index %d", ph.PhysicalIndex()))
		}
		panic("arm64: non-physical register reached printer (run pass.Compile first)")
	}
	// Pseudo registers (FP, SB, SP, PC) share their tokens with arm64.
	return r.Asm()
}

// operandReg returns the arm64 register name for an operand expected to be a register.
func operandReg(op operand.Op) string {
	r, ok := op.(reg.Register)
	if !ok {
		panic(fmt.Sprintf("arm64: expected register operand, got %T", op))
	}
	return rename(r)
}

// isImm reports whether op is an immediate constant, returning its asm form ("$n").
func immAsm(op operand.Op) (string, bool) {
	if c, ok := op.(operand.Constant); ok {
		return c.Asm(), true
	}
	return "", false
}

// log2scale returns the shift amount for a memory scale (1,2,4,8).
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

// memAsm lowers a memory operand. It may emit address-computation instructions
// (for indexed operands) and returns the simple base+disp arm64 operand string.
func (p *arm64) memAsm(m operand.Mem) string {
	// Symbol-relative (FP params, SB globals): pass through; base is a pseudo.
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
		// base+disp
		if m.Disp != 0 {
			return fmt.Sprintf("%d(%s)", m.Disp, rename(m.Base))
		}
		return fmt.Sprintf("(%s)", rename(m.Base))
	}

	// base + index*scale (+ disp): compute effective base into scratch.
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

func (p *arm64) lower(i *ir.Instruction) {
	ops := i.Operands
	switch i.Opcode {
	case "RET":
		p.emit("RET")

	case "JMP":
		p.emit("JMP %s", ops[0].Asm())

	// ---- moves and loads ----
	case "MOVQ":
		p.lowerMove("MOVD", ops[0], ops[1])
	case "MOVL":
		p.lowerMove("MOVW", ops[0], ops[1])
	case "MOVW":
		p.lowerMove("MOVH", ops[0], ops[1])
	case "MOVB":
		p.lowerMove("MOVB", ops[0], ops[1])
	case "MOVWQSX": // load int16, sign-extend
		p.emit("MOVH %s, %s", p.memAsm(ops[0].(operand.Mem)), operandReg(ops[1]))
	case "MOVWQZX": // load uint16, zero-extend
		p.emit("MOVHU %s, %s", p.memAsm(ops[0].(operand.Mem)), operandReg(ops[1]))
	case "MOVBQZX": // load uint8, zero-extend
		p.emit("MOVBU %s, %s", p.memAsm(ops[0].(operand.Mem)), operandReg(ops[1]))

	// ---- arithmetic / logic (dst is last operand) ----
	case "ADDQ":
		p.lowerRRR("ADD", ops[0], ops[1])
	case "SUBQ":
		p.lowerRRR("SUB", ops[0], ops[1])
	case "ANDQ":
		p.lowerRRR("AND", ops[0], ops[1])
	case "ORQ":
		p.lowerRRR("ORR", ops[0], ops[1])
	case "XORQ":
		// xor reg with itself is the idiomatic zero.
		if ra, ok := ops[0].(reg.Register); ok {
			if rb, ok := ops[1].(reg.Register); ok && rename(ra) == rename(rb) {
				p.emit("MOVD $0, %s", rename(rb))
				return
			}
		}
		p.lowerRRR("EOR", ops[0], ops[1])
	case "INCQ":
		p.emit("ADD $1, %s, %s", operandReg(ops[0]), operandReg(ops[0]))
	case "DECQ":
		p.emit("SUB $1, %s, %s", operandReg(ops[0]), operandReg(ops[0]))
	case "SHRQ":
		p.lowerShift("LSR", ops[0], ops[1])
	case "SHLQ":
		p.lowerShift("LSL", ops[0], ops[1])

	case "LEAQ":
		p.lowerLEA(ops[0].(operand.Mem), operandReg(ops[1]))

	case "BTSQ":
		// b |= 1 << a  (avo uses this to compute 1<<actualTableLog after zeroing).
		a := operandReg(ops[0])
		b := operandReg(ops[1])
		p.emit("MOVD $1, %s", scratchVal)
		p.emit("LSL %s, %s, %s", a, scratchVal, scratchVal)
		p.emit("ORR %s, %s, %s", scratchVal, b, b)

	case "BSRQ":
		// dst = index of most-significant set bit = 63 - CLZ(src).
		src := operandReg(ops[0])
		dst := operandReg(ops[1])
		p.emit("CLZ %s, %s", src, scratchVal)
		p.emit("MOVD $63, %s", dst)
		p.emit("SUB %s, %s, %s", scratchVal, dst, dst)

	default:
		panic(fmt.Sprintf("arm64: unsupported opcode %q (operands: %s)", i.Opcode, joinOperands(ops)))
	}
}

// lowerMove handles MOV* of reg/imm/mem to reg/mem.
func (p *arm64) lowerMove(op string, src, dst operand.Op) {
	dmem, dstIsMem := dst.(operand.Mem)
	smem, srcIsMem := src.(operand.Mem)
	switch {
	case dstIsMem:
		// store: materialize an immediate source first.
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

// lowerRRR lowers "OP src, dst" (dst op= src) to arm64 "OP src, dst, dst".
func (p *arm64) lowerRRR(op string, src, dst operand.Op) {
	d := operandReg(dst)
	if imm, ok := immAsm(src); ok {
		p.emit("%s %s, %s, %s", op, imm, d, d)
		return
	}
	p.emit("%s %s, %s, %s", op, operandReg(src), d, d)
}

// lowerShift lowers "SHIFT count, dst" (count is imm or CL) to "SHIFT count, dst, dst".
func (p *arm64) lowerShift(op string, count, dst operand.Op) {
	d := operandReg(dst)
	if imm, ok := immAsm(count); ok {
		p.emit("%s %s, %s, %s", op, imm, d, d)
		return
	}
	p.emit("%s %s, %s, %s", op, operandReg(count), d, d)
}

// lowerLEA lowers an effective-address computation into dst.
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

// lowerCompareBranch fuses a comparison with its conditional branch.
func (p *arm64) lowerCompareBranch(cmp, br *ir.Instruction) {
	target := br.Operands[0].Asm()
	bcc := branchMnemonic(br.Opcode)
	a := cmp.Operands[0]
	b := cmp.Operands[1]

	if strings.HasPrefix(cmp.Opcode, "TEST") {
		// TEST a, b ; Jcc  -> typically a == b (self test of zero/nonzero).
		ra, aok := a.(reg.Register)
		rb, bok := b.(reg.Register)
		if aok && bok && rename(ra) == rename(rb) {
			switch br.Opcode {
			case "JZ", "JE":
				p.emit("CBZ %s, %s", rename(ra), target)
				return
			case "JNZ", "JNE":
				p.emit("CBNZ %s, %s", rename(ra), target)
				return
			}
		}
		p.emit("TST %s, %s", operandReg(b), operandReg(a))
		p.emit("%s %s", bcc, target)
		return
	}

	// CMP a, b ; Jcc -> flags from (a - b).
	if imm, ok := immAsm(b); ok {
		if neg, val := negImm(imm); neg {
			// a - (negative) ; use CMN a, |imm|  (sets Z iff a == imm).
			p.emit("CMN $%d, %s", val, operandReg(a))
			p.emit("%s %s", bcc, target)
			return
		}
		p.emit("CMP %s, %s", imm, operandReg(a))
		p.emit("%s %s", bcc, target)
		return
	}
	// arm64 CMP Rm, Rn computes Rn - Rm; we want a - b, so Rn=a, Rm=b.
	p.emit("CMP %s, %s", operandReg(b), operandReg(a))
	p.emit("%s %s", bcc, target)
}

// negImm parses an immediate asm string ("$-1") and reports whether it is
// negative, returning the absolute value.
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

func branchMnemonic(op string) string {
	switch op {
	case "JL":
		return "BLT"
	case "JLE":
		return "BLE"
	case "JG":
		return "BGT"
	case "JGE":
		return "BGE"
	case "JE", "JZ":
		return "BEQ"
	case "JNE", "JNZ":
		return "BNE"
	case "JB":
		return "BLO"
	case "JBE":
		return "BLS"
	case "JA":
		return "BHI"
	case "JAE":
		return "BHS"
	default:
		panic(fmt.Sprintf("arm64: unsupported conditional branch %q", op))
	}
}

func isCompare(i *ir.Instruction) bool {
	return strings.HasPrefix(i.Opcode, "CMP") || strings.HasPrefix(i.Opcode, "TEST")
}

// nextInstruction returns the next *ir.Instruction after index idx, skipping
// comments (but not labels, which break fusion).
func nextInstruction(nodes []ir.Node, idx int) (*ir.Instruction, bool) {
	for j := idx + 1; j < len(nodes); j++ {
		switch n := nodes[j].(type) {
		case *ir.Comment:
			continue
		case *ir.Instruction:
			return n, true
		default:
			return nil, false
		}
	}
	return nil, false
}

// skipTo returns the index of node target starting from idx.
func skipTo(nodes []ir.Node, idx int, target *ir.Instruction) int {
	for j := idx + 1; j < len(nodes); j++ {
		if nodes[j] == target {
			return j
		}
	}
	return idx
}
