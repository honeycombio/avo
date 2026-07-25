// Package propspec defines the operation vocabulary used to build randomized
// programs for the arm64 lowering's property test.
//
// Each Op pairs an avo emitter with a pure-Go reference that models the exact
// x86 semantics of what the emitter produces, including partial-register and
// flag behaviour. Keeping the two in one struct is deliberate: the emitter and
// its reference are the two halves of a differential test, and separating them
// invites drift.
//
// The generator (asm.go) and the test both derive the same programs from the
// same seeds via Program, so neither needs to record what the other did.
package propspec

import (
	"math/bits"
	"math/rand"

	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

// Op is one step of a random program: acc = f(acc, y).
type Op struct {
	Name string
	// Emit appends the operation to the avo program being built, updating acc
	// in place. y is a second live value the operation may read.
	Emit func(acc, y reg.GPVirtual)
	// Ref is the pure-Go model of Emit's x86 semantics.
	Ref func(acc, y uint64) uint64
}

// Ops is the vocabulary. It deliberately over-samples the classes that have
// produced silent miscompiles in this lowering: partial-register writes
// (MOVB/ADDB/SHLB), operand-width truncation (32-bit ALU, MOVL), and
// flag-producer/consumer pairs (CMOVcc, SETcc).
var Ops = []Op{
	{
		Name: "AddQ",
		Emit: func(acc, y reg.GPVirtual) { ADDQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc + y },
	},
	{
		Name: "SubQ",
		Emit: func(acc, y reg.GPVirtual) { SUBQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc - y },
	},
	{
		Name: "AndQ",
		Emit: func(acc, y reg.GPVirtual) { ANDQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc & y },
	},
	{
		Name: "OrQ",
		Emit: func(acc, y reg.GPVirtual) { ORQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc | y },
	},
	{
		Name: "XorQ",
		Emit: func(acc, y reg.GPVirtual) { XORQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc ^ y },
	},
	{
		Name: "ImulQ",
		Emit: func(acc, y reg.GPVirtual) { IMULQ(y, acc) },
		Ref:  func(acc, y uint64) uint64 { return acc * y },
	},
	// 32-bit ALU: the result zero-extends into the 64-bit register.
	{
		Name: "AddL",
		Emit: func(acc, y reg.GPVirtual) { ADDL(y.As32(), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(acc) + uint32(y)) },
	},
	{
		Name: "SubL",
		Emit: func(acc, y reg.GPVirtual) { SUBL(y.As32(), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(acc) - uint32(y)) },
	},
	{
		Name: "AndL",
		Emit: func(acc, y reg.GPVirtual) { ANDL(y.As32(), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(acc) & uint32(y)) },
	},
	{
		Name: "ImulL",
		Emit: func(acc, y reg.GPVirtual) { IMULL(y.As32(), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(acc) * uint32(y)) },
	},
	// Shifts and rotates, immediate forms.
	{
		Name: "ShlQ3",
		Emit: func(acc, y reg.GPVirtual) { SHLQ(operand.U8(3), acc) },
		Ref:  func(acc, y uint64) uint64 { return acc << 3 },
	},
	{
		Name: "ShrQ7",
		Emit: func(acc, y reg.GPVirtual) { SHRQ(operand.U8(7), acc) },
		Ref:  func(acc, y uint64) uint64 { return acc >> 7 },
	},
	{
		Name: "SarQ5",
		Emit: func(acc, y reg.GPVirtual) { SARQ(operand.U8(5), acc) },
		Ref:  func(acc, y uint64) uint64 { return uint64(int64(acc) >> 5) },
	},
	{
		Name: "ShlL9",
		Emit: func(acc, y reg.GPVirtual) { SHLL(operand.U8(9), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(acc) << 9) },
	},
	{
		Name: "RolQ13",
		Emit: func(acc, y reg.GPVirtual) { ROLQ(operand.U8(13), acc) },
		Ref:  func(acc, y uint64) uint64 { return bits.RotateLeft64(acc, 13) },
	},
	{
		Name: "RolL5",
		Emit: func(acc, y reg.GPVirtual) { ROLL(operand.U8(5), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(bits.RotateLeft32(uint32(acc), 5)) },
	},
	// Unary.
	{
		Name: "NotQ",
		Emit: func(acc, y reg.GPVirtual) { NOTQ(acc) },
		Ref:  func(acc, y uint64) uint64 { return ^acc },
	},
	{
		Name: "NegQ",
		Emit: func(acc, y reg.GPVirtual) { NEGQ(acc) },
		Ref:  func(acc, y uint64) uint64 { return -acc },
	},
	{
		Name: "PopcntQ",
		Emit: func(acc, y reg.GPVirtual) { POPCNTQ(acc, acc) },
		Ref:  func(acc, y uint64) uint64 { return uint64(bits.OnesCount64(acc)) },
	},
	{
		Name: "TzcntQ",
		Emit: func(acc, y reg.GPVirtual) { TZCNTQ(acc, acc) },
		Ref:  func(acc, y uint64) uint64 { return uint64(bits.TrailingZeros64(acc)) },
	},
	{
		Name: "BswapL",
		Emit: func(acc, y reg.GPVirtual) { BSWAPL(acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(bits.ReverseBytes32(uint32(acc))) },
	},
	// Width truncation and extension.
	{
		// Copies y's low half into acc, zero-extending. Deliberately not a
		// self-move: avo elides "MOVL AX, AX" as redundant, even though on x86
		// it clears the upper 32 bits.
		Name: "MovLFromY",
		Emit: func(acc, y reg.GPVirtual) { MOVL(y.As32(), acc.As32()) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint32(y)) },
	},
	{
		Name: "SxL",
		Emit: func(acc, y reg.GPVirtual) { MOVLQSX(acc.As32(), acc) },
		Ref:  func(acc, y uint64) uint64 { return uint64(int64(int32(acc))) },
	},
	{
		Name: "ZxW",
		Emit: func(acc, y reg.GPVirtual) { MOVWQZX(acc.As16(), acc) },
		Ref:  func(acc, y uint64) uint64 { return uint64(uint16(acc)) },
	},
	// Partial-register writes: only the addressed byte changes.
	{
		Name: "MovBLow",
		Emit: func(acc, y reg.GPVirtual) { MOVB(y.As8(), acc.As8()) },
		Ref:  func(acc, y uint64) uint64 { return acc&^0xff | y&0xff },
	},
	{
		Name: "AddB",
		Emit: func(acc, y reg.GPVirtual) { ADDB(y.As8(), acc.As8()) },
		Ref:  func(acc, y uint64) uint64 { return acc&^0xff | (acc+y)&0xff },
	},
	{
		Name: "ShlB2",
		Emit: func(acc, y reg.GPVirtual) { SHLB(operand.U8(2), acc.As8()) },
		Ref:  func(acc, y uint64) uint64 { return acc&^0xff | (acc&0xff)<<2&0xff },
	},
	// Flag producer/consumer pairs. Two of the three miscompiles found in this
	// lowering lived here, so both CMOVcc and SETcc are represented.
	{
		Name: "CmovBelow",
		Emit: func(acc, y reg.GPVirtual) {
			CMPQ(acc, y)
			CMOVQCS(y, acc) // acc = acc < y ? y : acc
		},
		Ref: func(acc, y uint64) uint64 {
			if acc < y {
				return y
			}
			return acc
		},
	},
	{
		Name: "CmovEq",
		Emit: func(acc, y reg.GPVirtual) {
			CMPQ(acc, y)
			CMOVQEQ(y, acc)
		},
		Ref: func(acc, y uint64) uint64 { return acc },
	},
	{
		Name: "SetLess",
		Emit: func(acc, y reg.GPVirtual) {
			CMPQ(acc, y)
			SETLT(acc.As8()) // signed <, written into acc's low byte
		},
		Ref: func(acc, y uint64) uint64 {
			v := uint64(0)
			if int64(acc) < int64(y) {
				v = 1
			}
			return acc&^0xff | v
		},
	},
	{
		Name: "XorSelfZero",
		Emit: func(acc, y reg.GPVirtual) { XORQ(acc, acc) },
		Ref:  func(acc, y uint64) uint64 { return 0 },
	},
}

// Program returns the op indices making up program n. Both the generator and
// the test call this, so the two always agree without recording anything.
func Program(n, length int) []int {
	r := rand.New(rand.NewSource(int64(n) * 7919))
	out := make([]int, length)
	for i := range out {
		out[i] = r.Intn(len(Ops))
	}
	return out
}

// NumPrograms is how many random programs the suite builds.
const NumPrograms = 64

// ProgramLength is how many operations each one chains together. Long enough to
// interleave widths and flag pairs, short enough to point at a culprit when a
// program fails.
const ProgramLength = 12
