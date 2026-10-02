//go:build ignore

package main

import . "github.com/honeycombio/avo/build"

func main() {
	Package("github.com/honeycombio/avo/tests/fixedbugs/issue62")
	Implement("private")
	RET()
	Generate()
}
