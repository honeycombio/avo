//go:build ignore

// This program generates functions that exercise the EXPERIMENTAL arm64
// lowering printer. The same avo source is emitted to both amd64 and arm64; the
// test file runs each generated function against a pure-Go reference (via
// testing/quick), so on each architecture CI verifies that the lowering is
// semantics-preserving. It deliberately covers the operand-width and
// sub-register cases that produce plausible-but-wrong asm (MOVL zero-extension,
// high-byte reads) and the conditional mappings real generators may not hit.
package main

import (
	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	// Add: ADDQ.
	TEXT("Add", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		y := Load(Param("y"), GP64())
		ADDQ(x, y)
		Store(y, ReturnIndex(0))
		RET()
	}

	// Sub: SUBQ.
	TEXT("Sub", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		y := Load(Param("y"), GP64())
		SUBQ(y, x) // x -= y
		Store(x, ReturnIndex(0))
		RET()
	}

	// Neg: NEGQ.
	TEXT("Neg", NOSPLIT, "func(x uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		NEGQ(x)
		Store(x, ReturnIndex(0))
		RET()
	}

	// AndOrXor: ANDQ, ORQ, XORQ combined.
	TEXT("AndOrXor", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		y := Load(Param("y"), GP64())
		a := GP64()
		MOVQ(x, a)
		ANDQ(y, a) // a = x & y
		o := GP64()
		MOVQ(x, o)
		ORQ(y, o)  // o = x | y
		XORQ(o, a) // a = (x&y) ^ (x|y)
		Store(a, ReturnIndex(0))
		RET()
	}

	// Variable shifts by CL: SHLQ, SHRQ, ROLQ.
	TEXT("ShlCL", NOSPLIT, "func(x, n uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		n := GP64()
		Load(Param("n"), n)
		MOVQ(n, reg.RCX)
		SHLQ(reg.CL, x)
		Store(x, ReturnIndex(0))
		RET()
	}
	TEXT("ShrCL", NOSPLIT, "func(x, n uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		n := GP64()
		Load(Param("n"), n)
		MOVQ(n, reg.RCX)
		SHRQ(reg.CL, x)
		Store(x, ReturnIndex(0))
		RET()
	}
	TEXT("RolCL", NOSPLIT, "func(x, n uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		n := GP64()
		Load(Param("n"), n)
		MOVQ(n, reg.RCX)
		ROLQ(reg.CL, x)
		Store(x, ReturnIndex(0))
		RET()
	}

	// ZeroExt32: MOVL register move must zero the upper 32 bits (x86 semantics).
	// The case where Go arm64 MOVW would wrongly sign-extend.
	TEXT("ZeroExt32", NOSPLIT, "func(x uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		y := GP64()
		MOVL(x.As32(), y.As32()) // y = uint64(uint32(x))
		// Keep x live so the truncating move cannot be coalesced/elided; the sum
		// observes whether the upper 32 bits of y were correctly zeroed.
		ADDQ(x, y)
		Store(y, ReturnIndex(0))
		RET()
	}

	// HighByte: read bits 8-15 via the AH-style high-byte register.
	TEXT("HighByte", NOSPLIT, "func(x uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		y := GP64()
		MOVB(x.As8H(), y.As8L()) // y_low = (x >> 8) & 0xff
		MOVBQZX(y.As8(), y)      // zero-extend on both architectures
		Store(y, ReturnIndex(0))
		RET()
	}

	// LoadIdx: indexed memory load (base+index*scale), lowered via a scratch ADD.
	TEXT("LoadIdx", NOSPLIT, "func(p *[8]uint64, i uint64) uint64")
	{
		p := Load(Param("p"), GP64())
		i := Load(Param("i"), GP64())
		v := GP64()
		MOVQ(operand.Mem{Base: p, Index: i, Scale: 8}, v)
		Store(v, ReturnIndex(0))
		RET()
	}

	// Copy16: MOVUPS 16-byte copy through a vector register.
	TEXT("Copy16", NOSPLIT, "func(dst, src *[16]byte)")
	{
		dst := Load(Param("dst"), GP64())
		src := Load(Param("src"), GP64())
		x := XMM()
		MOVUPS(operand.Mem{Base: src}, x)
		MOVUPS(x, operand.Mem{Base: dst})
		RET()
	}

	// Condition helpers: each returns 1 if the condition holds, else 0,
	// exercising a distinct branch mnemonic (including JGE, unused by zstd).
	branch := func(name string, jmp func(operand.Op)) {
		TEXT(name, NOSPLIT, "func(a, b uint64) uint64")
		a := Load(Param("a"), GP64())
		b := Load(Param("b"), GP64())
		res := GP64()
		CMPQ(a, b)
		jmp(operand.LabelRef(name + "_yes"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef(name + "_end"))
		Label(name + "_yes")
		MOVQ(operand.U64(1), res)
		Label(name + "_end")
		Store(res, ReturnIndex(0))
		RET()
	}
	branch("LessS", JLT)      // signed <
	branch("GreaterEqS", JGE) // signed >= (not used by zstd)
	branch("LessU", JCS)      // unsigned <
	branch("AboveU", JHI)     // unsigned >
	branch("Equal", JEQ)

	// SelectEq: CMOVQEQ — res = (a==b) ? c : d.
	TEXT("SelectEq", NOSPLIT, "func(a, b, c, d uint64) uint64")
	{
		a := Load(Param("a"), GP64())
		b := Load(Param("b"), GP64())
		c := Load(Param("c"), GP64())
		res := Load(Param("d"), GP64())
		CMPQ(a, b)
		CMOVQEQ(c, res)
		Store(res, ReturnIndex(0))
		RET()
	}

	// selBranch emits "res = (op sets ZF) ? 1 : 0" with a flag-transparent MOVQ
	// wedged between the flag-setting op and the JEQ that consumes it. x86 leaves
	// flags untouched across a MOV, so a lowering that only inspects the single
	// instruction following the producer fails to emit a flag-setting variant and
	// the branch then reads stale NZCV.
	selBranch := func(name string, produce func(res reg.GPVirtual)) {
		TEXT(name, NOSPLIT, "func(x, y uint64) uint64")
		res := GP64()
		produce(res)
		MOVQ(operand.U64(0), res)            // flag-transparent gap instruction
		JEQ(operand.LabelRef(name + "_yes")) // consumes ZF from the producer above
		JMP(operand.LabelRef(name + "_end"))
		Label(name + "_yes")
		MOVQ(operand.U64(1), res)
		Label(name + "_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// SubGapEq: SUBQ producer separated from JEQ by a MOV. res = (x-y==0).
	selBranch("SubGapEq", func(res reg.GPVirtual) {
		x := Load(Param("x"), GP64())
		y := Load(Param("y"), GP64())
		SUBQ(y, x) // x -= y; ZF set iff x == y
	})

	// DecGapZero: DECQ producer separated from JEQ by a MOV. res = (x-1==0).
	// Covers the lowerIncDec path (the loop-counter idiom) as well as lowerArith.
	selBranch("DecGapZero", func(res reg.GPVirtual) {
		x := Load(Param("x"), GP64())
		DECQ(x) // x--; ZF set iff x == 1
	})

	// cmpLow32 emits "res = (cmp of low 32 bits) ? 1 : 0" via a 32-bit compare on
	// operands whose upper 32 bits differ, so folding CMPL to a 64-bit CMP is
	// observably wrong. jmp selects the (signed or unsigned) branch under test.
	cmpLow32 := func(name string, jmp func(operand.Op)) {
		TEXT(name, NOSPLIT, "func(a, b uint64) uint64")
		a := GP64()
		Load(Param("a"), a)
		b := GP64()
		Load(Param("b"), b)
		res := GP64()
		CMPL(a.As32(), b.As32())
		jmp(operand.LabelRef(name + "_yes"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef(name + "_end"))
		Label(name + "_yes")
		MOVQ(operand.U64(1), res)
		Label(name + "_end")
		Store(res, ReturnIndex(0))
		RET()
	}
	cmpLow32("CmpL32Eq", JEQ)    // low32(a) == low32(b)
	cmpLow32("CmpL32LessS", JLT) // int32(a) < int32(b)
	cmpLow32("CmpL32LessU", JCS) // uint32(a) < uint32(b)

	// TestL32: TESTL sets ZF from the low-32-bit AND; the 64-bit fold is wrong
	// when the operands share set bits only above bit 31. res = (low32 AND == 0).
	TEXT("TestL32", NOSPLIT, "func(a, m uint64) uint64")
	{
		a := GP64()
		Load(Param("a"), a)
		m := GP64()
		Load(Param("m"), m)
		res := GP64()
		TESTL(a.As32(), m.As32())
		JEQ(operand.LabelRef("TestL32_zero"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef("TestL32_end"))
		Label("TestL32_zero")
		MOVQ(operand.U64(1), res)
		Label("TestL32_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// CmpW16Eq: res = (low 16 bits of a == low 16 bits of b) via a 16-bit
	// compare on operands whose upper 48 bits differ, exercising the EQ/NE-only
	// zero-extend fallback for sub-32-bit compares: a naive 64-bit fold would
	// see the differing upper bits and wrongly report inequality.
	TEXT("CmpW16Eq", NOSPLIT, "func(a, b uint64) uint64")
	{
		a := GP64()
		Load(Param("a"), a)
		b := GP64()
		Load(Param("b"), b)
		res := GP64()
		CMPW(a.As16(), b.As16())
		JEQ(operand.LabelRef("CmpW16Eq_yes"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef("CmpW16Eq_end"))
		Label("CmpW16Eq_yes")
		MOVQ(operand.U64(1), res)
		Label("CmpW16Eq_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// CmpB8Ne: same idea as CmpW16Eq but 8-bit (CMPB), consumed by JNE instead
	// of JEQ.
	TEXT("CmpB8Ne", NOSPLIT, "func(a, b uint64) uint64")
	{
		a := GP64()
		Load(Param("a"), a)
		b := GP64()
		Load(Param("b"), b)
		res := GP64()
		CMPB(a.As8(), b.As8())
		JNE(operand.LabelRef("CmpB8Ne_diff"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef("CmpB8Ne_end"))
		Label("CmpB8Ne_diff")
		MOVQ(operand.U64(1), res)
		Label("CmpB8Ne_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// TestW16Eq: TESTW sets ZF from the low-16-bit AND; the 64-bit fold is wrong
	// when the operands share set bits only above bit 15. res = (low16 AND == 0).
	TEXT("TestW16Eq", NOSPLIT, "func(a, m uint64) uint64")
	{
		a := GP64()
		Load(Param("a"), a)
		m := GP64()
		Load(Param("m"), m)
		res := GP64()
		TESTW(a.As16(), m.As16())
		JEQ(operand.LabelRef("TestW16Eq_zero"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef("TestW16Eq_end"))
		Label("TestW16Eq_zero")
		MOVQ(operand.U64(1), res)
		Label("TestW16Eq_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// ---- multiplication ----

	// IMul2: two-operand IMULQ (dst *= src), low 64 bits.
	TEXT("IMul2", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		y := Load(Param("y"), GP64())
		IMULQ(x, y) // y *= x
		Store(y, ReturnIndex(0))
		RET()
	}

	// IMul3: three-operand IMUL3Q (dst = src * imm), low 64 bits.
	TEXT("IMul3", NOSPLIT, "func(x uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		dst := GP64()
		IMUL3Q(operand.Imm(0x9E3779B1), x, dst) // dst = x * prime
		Store(dst, ReturnIndex(0))
		RET()
	}

	// MulWide: one-operand MULQ (RDX:RAX = RAX * src), unsigned 128-bit product.
	TEXT("MulWide", NOSPLIT, "func(x, y uint64) (lo uint64, hi uint64)")
	{
		Load(Param("x"), reg.RAX)
		Load(Param("y"), reg.RSI)
		MULQ(reg.RSI)
		Store(reg.RAX, ReturnIndex(0)) // low
		Store(reg.RDX, ReturnIndex(1)) // high
		RET()
	}

	// IMulWide: one-operand IMULQ (RDX:RAX = RAX * src), signed 128-bit product.
	TEXT("IMulWide", NOSPLIT, "func(x, y int64) (lo uint64, hi uint64)")
	{
		Load(Param("x"), reg.RAX)
		Load(Param("y"), reg.RSI)
		IMULQ(reg.RSI)
		Store(reg.RAX, ReturnIndex(0)) // low
		Store(reg.RDX, ReturnIndex(1)) // high (signed)
		RET()
	}

	// MulX: BMI2 MULXQ (flag-free wide multiply, implicit RDX factor).
	TEXT("MulX", NOSPLIT, "func(x, y uint64) (lo uint64, hi uint64)")
	{
		Load(Param("x"), reg.RSI)
		Load(Param("y"), reg.RDX)        // implicit MULX factor
		MULXQ(reg.RSI, reg.RAX, reg.RBX) // {RBX,RAX} = RSI * RDX
		Store(reg.RAX, ReturnIndex(0))   // low
		Store(reg.RBX, ReturnIndex(1))   // high
		RET()
	}

	// ---- BMI2 flag-free shifts / rotate ----
	shiftX := func(name string, op func(count, src, dst operand.Op)) {
		TEXT(name, NOSPLIT, "func(x, n uint64) uint64")
		x := Load(Param("x"), GP64())
		n := Load(Param("n"), GP64())
		dst := GP64()
		op(n, x, dst)
		Store(dst, ReturnIndex(0))
		RET()
	}
	shiftX("ShlX", func(count, src, dst operand.Op) { SHLXQ(count, src, dst) })
	shiftX("ShrX", func(count, src, dst operand.Op) { SHRXQ(count, src, dst) })
	shiftX("SarX", func(count, src, dst operand.Op) { SARXQ(count, src, dst) })

	// RorX: BMI2 RORXQ with an immediate rotate.
	TEXT("RorX", NOSPLIT, "func(x uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		dst := GP64()
		RORXQ(operand.Imm(56), x, dst) // dst = ror(x, 56)
		Store(dst, ReturnIndex(0))
		RET()
	}

	// ---- BMI2 bit-field ops ----

	// Bzhi: BZHIQ zeroes bits at index n and above (n < 64).
	TEXT("Bzhi", NOSPLIT, "func(x, n uint64) uint64")
	{
		x := Load(Param("x"), GP64())
		n := Load(Param("n"), GP64())
		dst := GP64()
		BZHIQ(n, x, dst) // dst = x & ((1<<n)-1)
		Store(dst, ReturnIndex(0))
		RET()
	}

	// bextr: BEXTRQ extracts `length` bits starting at `start` (control built the
	// way zstd does, MOVQ of start|(length<<8) into a register).
	bextr := func(name string, start, length uint32) {
		TEXT(name, NOSPLIT, "func(x uint64) uint64")
		x := Load(Param("x"), GP64())
		ctrl := GP64()
		MOVQ(operand.U32(start|(length<<8)), ctrl)
		dst := GP64()
		BEXTRQ(ctrl, x, dst)
		Store(dst, ReturnIndex(0))
		RET()
	}
	bextr("Bextr88", 8, 8)    // zstd's exact usage
	bextr("Bextr4_12", 4, 12) // asymmetric start/len

	// ---- CMOVcc conditions newly enabled by the completed table ----
	selectCC := func(name string, cmov func(src, dst operand.Op)) {
		TEXT(name, NOSPLIT, "func(a, b, c, d uint64) uint64")
		a := Load(Param("a"), GP64())
		b := Load(Param("b"), GP64())
		c := Load(Param("c"), GP64())
		res := Load(Param("d"), GP64())
		CMPQ(a, b)
		cmov(c, res) // res = cond(a,b) ? c : d
		Store(res, ReturnIndex(0))
		RET()
	}
	selectCC("SelLtS", func(s, d operand.Op) { CMOVQLT(s, d) }) // signed <
	selectCC("SelLeS", func(s, d operand.Op) { CMOVQLE(s, d) }) // signed <=
	selectCC("SelGtS", func(s, d operand.Op) { CMOVQGT(s, d) }) // signed >
	selectCC("SelGeS", func(s, d operand.Op) { CMOVQGE(s, d) }) // signed >=
	selectCC("SelLsU", func(s, d operand.Op) { CMOVQLS(s, d) }) // unsigned <=
	selectCC("SelMi", func(s, d operand.Op) { CMOVQMI(s, d) })  // sign set
	selectCC("SelPl", func(s, d operand.Op) { CMOVQPL(s, d) })  // sign clear

	Generate()
}
