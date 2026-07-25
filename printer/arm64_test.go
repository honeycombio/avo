package printer_test

import (
	"strings"
	"testing"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/printer"
	"github.com/mmcloughlin/avo/reg"
)

// twinTestContext builds a file with a generic/BMI2 twin pair and a
// single-variant function, shared by both twin-preference tests below.
func twinTestContext() *build.Context {
	ctx := build.NewContext()
	for _, name := range []string{"twin_amd64", "twin_bmi2", "solo_amd64"} {
		ctx.Function(name)
		ctx.SignatureExpr("func()")
		ctx.RET()
	}
	return ctx
}

// printARM64 prints ctx with the arm64 lowering printer under the given config.
func printARM64(t *testing.T, ctx *build.Context, cfg printer.Config) string {
	t.Helper()
	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}
	b, err := printer.NewARM64Asm(cfg).Print(f)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestARM64PrefersGenericTwinByDefault checks that with the default config the
// arm64 lowering picks a function's generic twin over its BMI2 one: BMI2 x86
// code is tuned for x86 and is not reliably faster once mechanically lowered
// (e.g. BEXTR packs into one instruction what arm64 needs two UBFX to unpack),
// so this is the safe default absent a measured reason to prefer BMI2. A
// function with only one variant is lowered regardless.
func TestARM64PrefersGenericTwinByDefault(t *testing.T) {
	out := printARM64(t, twinTestContext(), printer.NewDefaultConfig())

	if n := strings.Count(out, "TEXT ·twin_arm64(SB)"); n != 1 {
		t.Errorf("expected exactly one twin_arm64 definition, got %d:\n%s", n, out)
	}
	if !strings.Contains(out, "skipped twin_bmi2") {
		t.Errorf("expected BMI2 twin_bmi2 to be skipped by default:\n%s", out)
	}
	if strings.Contains(out, "TEXT ·twin_amd64(SB)") {
		t.Errorf("generic variant should be renamed, not emitted as twin_amd64:\n%s", out)
	}
	if !strings.Contains(out, "TEXT ·solo_arm64(SB)") {
		t.Errorf("expected single-variant solo_amd64 lowered to solo_arm64:\n%s", out)
	}
}

// TestARM64PreferBMI2TwinOptIn checks the ARM64PreferBMI2 config opt-in flips
// the choice to the BMI2 twin, for callers who have actually measured it to be
// faster for their case.
func TestARM64PreferBMI2TwinOptIn(t *testing.T) {
	cfg := printer.NewDefaultConfig()
	cfg.ARM64PreferBMI2 = true
	out := printARM64(t, twinTestContext(), cfg)

	if n := strings.Count(out, "TEXT ·twin_arm64(SB)"); n != 1 {
		t.Errorf("expected exactly one twin_arm64 definition, got %d:\n%s", n, out)
	}
	if !strings.Contains(out, "skipped twin_amd64") {
		t.Errorf("expected generic twin_amd64 to be skipped when BMI2 is preferred:\n%s", out)
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

// TestARM64CrossLabelFlagsRejected checks that a conditional branch whose flags
// are produced before a label fails generation instead of silently emitting a
// non-flag-setting op. The lowering recovers the producer/consumer link by
// scanning backwards through a straight-line run; when a label intervenes the
// flags arrive along a control-flow edge it does not model, and quietly leaving
// the producer unmarked would let the branch read whatever NZCV survived.
func TestARM64CrossLabelFlagsRejected(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("crosslabel")
	ctx.SignatureExpr("func(x, y uint64) uint64")
	x, y := reg.RAX, reg.RCX
	ctx.SUBQ(y, x) // producer
	ctx.Label("join")
	ctx.JEQ(operand.LabelRef("yes")) // consumer, separated by the label
	ctx.Label("yes")
	ctx.RET()

	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic for flags read across a label, got none")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "across a label") {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
}

// TestARM64CallRejected checks that a CALL fails generation. The lowering emits
// NOFRAME leaf functions and uses caller-saved registers (including R16/R17 and
// the two scratch registers) without preserving them, all of which is only
// sound while the function makes no calls.
func TestARM64CallRejected(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("calls")
	ctx.SignatureExpr("func()")
	ctx.CALL(operand.LabelRef("somewhere"))
	ctx.RET()

	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic for CALL, got none")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "CALL") {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
}

// TestARM64GOAMD64ConditionalsEvaluated checks that a GOAMD64 #ifdef/#else pair
// contributes only its else arm to arm64 output. Generators emit these
// directives as comments, so passing them through would either duplicate the
// work or, if the comment markers are never stripped, run both arms.
func TestARM64GOAMD64ConditionalsEvaluated(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("cond")
	ctx.SignatureExpr("func()")
	ctx.Comment("#ifdef GOAMD64_v3")
	ctx.TZCNTQ(reg.RAX, reg.RCX) // amd64-only arm
	ctx.Comment("#else")
	ctx.BSFQ(reg.RAX, reg.RDX) // the arm arm64 must take
	ctx.Comment("#endif")
	ctx.RET()

	out := printARM64(t, ctx, printer.NewGoRunConfig())

	if strings.Contains(out, "#ifdef") || strings.Contains(out, "#else") || strings.Contains(out, "#endif") {
		t.Errorf("GOAMD64 directives survived into arm64 output:\n%s", out)
	}
	// BSFQ targets RDX (R2); TZCNTQ targets RCX (R1). Only the else arm should
	// have been lowered.
	if !strings.Contains(out, "R2") {
		t.Errorf("else arm (BSFQ -> R2) missing from output:\n%s", out)
	}
	if strings.Contains(out, "R1") {
		t.Errorf("then arm (TZCNTQ -> R1) should have been skipped:\n%s", out)
	}
}
