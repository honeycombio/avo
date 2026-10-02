//go:build ignore

package main

import (
	. "github.com/honeycombio/avo/build"
	. "github.com/honeycombio/avo/operand"
	. "github.com/honeycombio/avo/reg"
)

func main() {
	TEXT("Issue65", NOSPLIT, "func()")
	VINSERTI128(Imm(1), Y0.AsX(), Y1, Y2)
	RET()
	Generate()
}
