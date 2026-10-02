//go:build ignore

package main

import (
	. "github.com/honeycombio/avo/build"
	. "github.com/honeycombio/avo/operand"
)

func main() {
	TEXT("Issue89", NOSPLIT, "func() uint64")
	x := GP64()
	MOVQ(U32(42), x)
	for i := 0; i < 100; i++ {
		zero := GP64()
		XORQ(zero, zero)
		ADDQ(zero, x)
	}
	Store(x, ReturnIndex(0))
	RET()
	Generate()
}
