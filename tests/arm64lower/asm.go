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

	Generate()
}
