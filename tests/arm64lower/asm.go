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
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
	"github.com/mmcloughlin/avo/tests/arm64lower/propspec"
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
	// Constant-fold boundary cases: field reaching bit 63 (LSR form), field
	// crossing past bit 63 (effective width shrinks), empty field, and start
	// past bit 63 (both constant zero).
	bextr("Bextr56_8", 56, 8)
	bextr("Bextr8_56", 8, 56)
	bextr("Bextr8_60", 8, 60)
	bextr("Bextr0_0", 0, 0)
	bextr("Bextr70_8", 70, 8)

	// BZHI/SHLX/SHRX with the count loaded as an adjacent constant, exercising
	// the immediate fold on arm64 (real BMI2 register semantics on amd64).
	constCount := func(name string, val uint32, op func(count, src, dst operand.Op)) {
		TEXT(name, NOSPLIT, "func(x uint64) uint64")
		x := Load(Param("x"), GP64())
		cnt := GP64()
		MOVQ(operand.U32(val), cnt)
		dst := GP64()
		op(cnt, x, dst)
		Store(dst, ReturnIndex(0))
		RET()
	}
	constCount("BzhiConst13", 13, func(c, s, d operand.Op) { BZHIQ(c, s, d) })
	constCount("BzhiConst0", 0, func(c, s, d operand.Op) { BZHIQ(c, s, d) })
	constCount("BzhiConst64", 64, func(c, s, d operand.Op) { BZHIQ(c, s, d) })
	constCount("ShlXConst9", 9, func(c, s, d operand.Op) { SHLXQ(c, s, d) })
	constCount("ShrXConst9", 9, func(c, s, d operand.Op) { SHRXQ(c, s, d) })

	// ---- huff0-style idioms: SETcc, ADC carry-accumulate, BSWAPL, ADDB ----

	// SetGe: XOR-zeroed destination + SETGE, the huff0 exhausted-flag pattern.
	TEXT("SetGe", NOSPLIT, "func(a, b uint64) uint64")
	{
		a, b, dst := GP64(), GP64(), GP64()
		Load(Param("a"), a)
		Load(Param("b"), b)
		XORL(dst.As32(), dst.As32())
		CMPQ(a, b)
		SETGE(dst.As8())
		Store(dst, ReturnIndex(0))
		RET()
	}

	// AdcAccum: the huff0 fillFast32 exhausted counter: acc's low byte += (x<4),
	// on an arbitrary accumulator (upper bits preserved, byte wrap exact).
	TEXT("AdcAccum", NOSPLIT, "func(x, acc uint64) uint64")
	{
		x, acc := GP64(), GP64()
		Load(Param("x"), x)
		Load(Param("acc"), acc)
		CMPQ(x, operand.Imm(4))
		ADCB(operand.I8(0), acc.As8())
		Store(acc, ReturnIndex(0))
		RET()
	}

	// BswapL: 32-bit byte reverse, zero-extending like x86.
	TEXT("BswapL", NOSPLIT, "func(x uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		BSWAPL(x.As32())
		Store(x, ReturnIndex(0))
		RET()
	}

	// AddByte: exact x86 byte add on arbitrary values: y's low byte gets
	// (x+y) mod 256, all other bits of y preserved.
	TEXT("AddByte", NOSPLIT, "func(x, y uint64) uint64")
	{
		x, y := GP64(), GP64()
		Load(Param("x"), x)
		Load(Param("y"), y)
		ADDB(x.As8(), y.As8())
		Store(y, ReturnIndex(0))
		RET()
	}

	// MovbHighDst: MOVB into AH: replaces bits 15:8 of y with x's low byte,
	// preserving everything else (the huff0 byte-packing idiom).
	TEXT("MovbHighDst", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := GP64()
		Load(Param("x"), x)
		y := reg.RAX // fixed: high-byte access requires AX-DX
		Load(Param("y"), y)
		MOVB(x.As8(), y.As8H())
		Store(y, ReturnIndex(0))
		RET()
	}

	// MovbLowPreserve: MOVB from AH to a low byte: replaces bits 7:0 of y with
	// bits 15:8 of x, preserving y's upper bits.
	TEXT("MovbLowPreserve", NOSPLIT, "func(x, y uint64) uint64")
	{
		x := reg.RCX // fixed: high-byte access requires AX-DX
		Load(Param("x"), x)
		y := GP64()
		Load(Param("y"), y)
		MOVB(x.As8H(), y.As8())
		Store(y, ReturnIndex(0))
		RET()
	}

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

	// ---- extension, 32-bit ALU, rotates, bit ops (s2 + general coverage) ----

	un := func(name string, emit func(x, dst reg.GPVirtual)) {
		TEXT(name, NOSPLIT, "func(x uint64) uint64")
		x, dst := GP64(), GP64()
		Load(Param("x"), x)
		emit(x, dst)
		Store(dst, ReturnIndex(0))
		RET()
	}
	un("SxL", func(x, d reg.GPVirtual) { MOVLQSX(x.As32(), d) })
	un("SxB", func(x, d reg.GPVirtual) { MOVBQSX(x.As8(), d) })
	un("SxBL", func(x, d reg.GPVirtual) { MOVBLSX(x.As8(), d.As32()) })
	un("SxWL", func(x, d reg.GPVirtual) { MOVWLSX(x.As16(), d.As32()) })
	un("NegL", func(x, d reg.GPVirtual) { MOVQ(x, d); NEGL(d.As32()) })
	un("NotL", func(x, d reg.GPVirtual) { MOVQ(x, d); NOTL(d.As32()) })
	un("NotQ", func(x, d reg.GPVirtual) { MOVQ(x, d); NOTQ(d) })
	un("RolL7", func(x, d reg.GPVirtual) { MOVQ(x, d); ROLL(operand.U8(7), d.As32()) })
	un("RorQ9", func(x, d reg.GPVirtual) { MOVQ(x, d); RORQ(operand.U8(9), d) })
	un("RorL9", func(x, d reg.GPVirtual) { MOVQ(x, d); RORL(operand.U8(9), d.As32()) })
	un("BtrQ5", func(x, d reg.GPVirtual) {
		MOVQ(x, d)
		n := GP64()
		MOVQ(operand.U64(5), n)
		BTRQ(n, d)
	})
	un("BtcQ5", func(x, d reg.GPVirtual) {
		MOVQ(x, d)
		n := GP64()
		MOVQ(operand.U64(5), n)
		BTCQ(n, d)
	})
	un("PopcntQ", func(x, d reg.GPVirtual) { POPCNTQ(x, d) })
	un("SarQ3", func(x, d reg.GPVirtual) { MOVQ(x, d); SARQ(operand.U8(3), d) })
	un("SarL3", func(x, d reg.GPVirtual) { MOVQ(x, d); SARL(operand.U8(3), d.As32()) })
	un("IncL", func(x, d reg.GPVirtual) { MOVQ(x, d); INCL(d.As32()) })
	un("ShlB2", func(x, d reg.GPVirtual) { MOVQ(x, d); SHLB(operand.U8(2), d.As8()) })
	un("BsfQ", func(x, d reg.GPVirtual) { BSFQ(x, d) })
	un("TzcntQ", func(x, d reg.GPVirtual) { TZCNTQ(x, d) })

	bin := func(name string, emit func(x, y, dst reg.GPVirtual)) {
		TEXT(name, NOSPLIT, "func(x, y uint64) uint64")
		x, y, dst := GP64(), GP64(), GP64()
		Load(Param("x"), x)
		Load(Param("y"), y)
		emit(x, y, dst)
		Store(dst, ReturnIndex(0))
		RET()
	}
	bin("AddL", func(x, y, d reg.GPVirtual) { MOVQ(x, d); ADDL(y.As32(), d.As32()) })
	bin("SubL", func(x, y, d reg.GPVirtual) { MOVQ(x, d); SUBL(y.As32(), d.As32()) })
	bin("AndL", func(x, y, d reg.GPVirtual) { MOVQ(x, d); ANDL(y.As32(), d.As32()) })
	bin("OrL", func(x, y, d reg.GPVirtual) { MOVQ(x, d); ORL(y.As32(), d.As32()) })
	bin("ImulL", func(x, y, d reg.GPVirtual) { MOVQ(x, d); IMULL(y.As32(), d.As32()) })
	bin("XchgQ", func(x, y, d reg.GPVirtual) {
		a, b := GP64(), GP64()
		MOVQ(x, a)
		MOVQ(y, b)
		XCHGQ(a, b)
		// Return a after the swap, which must be y.
		MOVQ(a, d)
	})

	// LeaL: 32-bit address arithmetic, truncated and zero-extended.
	TEXT("LeaL", NOSPLIT, "func(x, y uint64) uint64")
	{
		x, y, d := GP64(), GP64(), GP64()
		Load(Param("x"), x)
		Load(Param("y"), y)
		LEAL(operand.Mem{Base: x, Index: y, Scale: 4, Disp: 7}, d.As32())
		Store(d, ReturnIndex(0))
		RET()
	}

	// VecXor: MOVOU load, PXOR against itself to zero, MOVOU store. Verifies the
	// 128-bit move and vector-xor lowerings end to end.
	TEXT("VecZero", NOSPLIT, "func(dst *[16]byte)")
	{
		dst := GP64()
		Load(Param("dst"), dst)
		z := XMM()
		PXOR(z, z)
		MOVOU(z, operand.Mem{Base: dst})
		RET()
	}

	// VecCopyIdx: 128-bit move through an indexed operand, the shape zstd's
	// match-copy loop emits. arm64 has no register-offset form for these, so the
	// address must be materialized.
	TEXT("VecCopyIdx", NOSPLIT, "func(dst, src *[32]byte, i uint64)")
	{
		dst, src, i := GP64(), GP64(), GP64()
		Load(Param("dst"), dst)
		Load(Param("src"), src)
		Load(Param("i"), i)
		v := XMM()
		MOVOU(operand.Mem{Base: src, Index: i, Scale: 1}, v)
		MOVOU(v, operand.Mem{Base: dst, Index: i, Scale: 1})
		RET()
	}

	// VecCopy: MOVOU load + MOVOU store with a displacement on each side.
	TEXT("VecCopy", NOSPLIT, "func(dst, src *[32]byte)")
	{
		dst, src := GP64(), GP64()
		Load(Param("dst"), dst)
		Load(Param("src"), src)
		v := XMM()
		MOVOU(operand.Mem{Base: src, Disp: 16}, v)
		MOVOU(v, operand.Mem{Base: dst, Disp: 16})
		RET()
	}

	// Randomized programs. Each chains operations from propspec, whose Go
	// references the test replays over the same sequence. This is what catches
	// the interaction bugs a hand-written case has to be thought of first: an
	// operand width that only matters after a particular predecessor, or a flag
	// producer separated from its consumer.
	for n := 0; n < propspec.NumPrograms; n++ {
		TEXT(fmt.Sprintf("Prop%d", n), NOSPLIT, "func(x, y uint64) uint64")
		acc, y := GP64(), GP64()
		Load(Param("x"), acc)
		Load(Param("y"), y)
		for _, op := range propspec.Program(n, propspec.ProgramLength) {
			propspec.Ops[op].Emit(acc, y)
		}
		Store(acc, ReturnIndex(0))
		RET()
	}

	// ---- regression cases for lowering bugs found in review ----

	// MovbzxHigh: a byte-extend from a high-byte source must read bits 15:8. AH
	// renames to the same arm64 register as AL, so a plain byte load silently
	// reads 7:0 instead. The 32-bit destination form is used deliberately: with a
	// 64-bit destination x86-64 needs a REX prefix, and REX redefines that
	// register slot as SPL, so AH is unreachable there.
	TEXT("MovbzxHigh", NOSPLIT, "func(x uint64) uint64")
	{
		x := reg.RAX // high-byte access requires AX-DX
		Load(Param("x"), x)
		d := GP64()
		MOVBLZX(x.As8H(), d.As32())
		Store(d, ReturnIndex(0))
		RET()
	}

	// CmovL32: a 32-bit CMOV writes its destination on BOTH condition outcomes,
	// zero-extending it. Here the condition is false, so a 64-bit CSEL would
	// leave the destination's upper half intact where x86 clears it.
	TEXT("CmovL32", NOSPLIT, "func(x, y uint64) uint64")
	{
		x, y, d := GP64(), GP64(), GP64()
		Load(Param("x"), x)
		Load(Param("y"), y)
		MOVQ(y, d)
		CMPQ(x, x)                  // equal, so NE is false
		CMOVLNE(x.As32(), d.As32()) // no move, but x86 still zero-extends d
		Store(d, ReturnIndex(0))
		RET()
	}

	// BextrMem: BEXTR with a runtime control and an indexed memory source. The
	// control fields and the source address both want a scratch register, and
	// staging them in the wrong order makes the shift count the address.
	TEXT("BextrMem", NOSPLIT, "func(p *[4]uint64, i uint64, ctrl uint64) uint64")
	{
		ptr, i, ctrl, d := GP64(), GP64(), GP64(), GP64()
		Load(Param("p"), ptr)
		Load(Param("i"), i)
		Load(Param("ctrl"), ctrl)
		BEXTRQ(ctrl, operand.Mem{Base: ptr, Index: i, Scale: 8}, d)
		Store(d, ReturnIndex(0))
		RET()
	}

	// XorSelfEq: x86's XOR-self zeroing idiom also sets ZF, and a consumer may
	// read it. The arm64 shortcut replaces the XOR with a move, which sets no
	// flags, so it has to supply them separately.
	TEXT("XorSelfEq", NOSPLIT, "func(x uint64) uint64")
	{
		x, res := GP64(), GP64()
		Load(Param("x"), x)
		XORQ(x, x)
		JEQ(operand.LabelRef("xorself_yes"))
		MOVQ(operand.U64(0), res)
		JMP(operand.LabelRef("xorself_end"))
		Label("xorself_yes")
		MOVQ(operand.U64(1), res)
		Label("xorself_end")
		Store(res, ReturnIndex(0))
		RET()
	}

	// MovLNegImm: a 32-bit move zero-extends, so a negative immediate must land
	// as its unsigned 32-bit value rather than sign-extended across all 64 bits.
	TEXT("MovLNegImm", NOSPLIT, "func() uint64")
	{
		d := GP64()
		MOVL(operand.I32(-1), d.As32())
		Store(d, ReturnIndex(0))
		RET()
	}

	Generate()
}
