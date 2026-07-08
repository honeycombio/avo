package printer_test

import (
	"strings"
	"testing"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/printer"
	"github.com/mmcloughlin/avo/reg"
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

// TestARM64SubwordCompareGuard checks the sub-32-bit CMPW guard: lowering
// succeeds when the only consumer is EQ/NE (lowerSubwordCompareEqNe applies),
// and still panics for an ordering consumer (JLT), which would need
// sign-aware operand extension this printer does not model.
func TestARM64SubwordCompareGuard(t *testing.T) {
	t.Run("EqNeAllowed", func(t *testing.T) {
		ctx := build.NewContext()
		ctx.Function("f")
		ctx.SignatureExpr("func()")
		ctx.CMPW(reg.RAX.As16(), reg.RCX.As16())
		ctx.JEQ(operand.LabelRef("yes"))
		ctx.Label("yes")
		ctx.RET()

		out := Print(t, ctx, printer.NewARM64Asm)
		if !strings.Contains(out, "CMP") {
			t.Errorf("expected a lowered CMP, got:\n%s", out)
		}
	})

	t.Run("OrderingPanics", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("expected lowering CMPW followed by JLT to panic")
			}
		}()

		ctx := build.NewContext()
		ctx.Function("f")
		ctx.SignatureExpr("func()")
		ctx.CMPW(reg.RAX.As16(), reg.RCX.As16())
		ctx.JLT(operand.LabelRef("yes"))
		ctx.Label("yes")
		ctx.RET()

		Print(t, ctx, printer.NewARM64Asm)
	})
}
