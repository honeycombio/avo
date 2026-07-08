package printer_test

import (
	"strings"
	"testing"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/printer"
)

// TestARM64PrefersBMI2Twin checks the arm64 lowering picks the BMI2 variant over
// its generic twin (BMI2 idioms lower to faster native arm64), while a function
// with only one variant is still lowered.
func TestARM64PrefersBMI2Twin(t *testing.T) {
	ctx := build.NewContext()
	for _, name := range []string{"twin_amd64", "twin_bmi2", "solo_amd64"} {
		ctx.Function(name)
		ctx.SignatureExpr("func()")
		ctx.RET()
	}

	out := Print(t, ctx, printer.NewARM64Asm)

	if n := strings.Count(out, "TEXT ·twin_arm64(SB)"); n != 1 {
		t.Errorf("expected exactly one twin_arm64 definition, got %d:\n%s", n, out)
	}
	if !strings.Contains(out, "skipped twin_amd64") {
		t.Errorf("expected generic twin_amd64 to be skipped:\n%s", out)
	}
	if strings.Contains(out, "TEXT ·twin_bmi2(SB)") {
		t.Errorf("BMI2 variant should be renamed, not emitted as twin_bmi2:\n%s", out)
	}
	if !strings.Contains(out, "TEXT ·solo_arm64(SB)") {
		t.Errorf("expected single-variant solo_amd64 lowered to solo_arm64:\n%s", out)
	}
}
