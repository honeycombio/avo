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

// TestARM64FlagSemanticGuards covers sequences the lowering must refuse rather
// than miscompile. Each pairs a flag producer with a consumer whose condition
// the arm64 translation cannot faithfully reproduce, because the two
// architectures disagree about what the flag means after that producer.
func TestARM64FlagSemanticGuards(t *testing.T) {
	cases := []struct {
		name  string
		build func(ctx *build.Context)
		want  string
	}{
		{
			// x86 TEST forces CF=0 and so does arm64 TST, meaning the raw bits
			// agree; the meaning-inverting condition map (correct after a
			// compare) then flips the answer. JA is never taken after this TEST
			// on x86, but BHI would always be taken.
			name: "carry condition after TEST",
			build: func(ctx *build.Context) {
				ctx.TESTQ(reg.RAX, reg.RCX)
				ctx.JHI(operand.LabelRef("l"))
			},
			want: "only the ZF/SF conditions",
		},
		{
			// x86 INC leaves CF alone, so this reads the compare's borrow. The
			// arm64 ADDS it lowers to overwrites C with the increment's carry.
			name: "carry condition across INC",
			build: func(ctx *build.Context) {
				ctx.CMPQ(reg.RAX, reg.RCX)
				ctx.INCQ(reg.RDX)
				ctx.JCS(operand.LabelRef("l"))
			},
			want: "preserves CF on x86",
		},
		{
			// ADC's lowering hardcodes the borrow convention, which only holds
			// after a compare or subtract. After an addition both architectures
			// use the same carry-out sense, so it would increment inversely.
			name: "ADC after an addition",
			build: func(ctx *build.Context) {
				ctx.ADDQ(reg.RAX, reg.RCX)
				ctx.ADCB(operand.I8(0), reg.DL)
			},
			want: "borrow-producing compare or subtract",
		},
		{
			// A signed sub-word compare whose consumer sits past a
			// flag-transparent MOV. The zero-extending translation is only valid
			// for EQ/NE, so this must not be silently blessed.
			name: "signed sub-word compare past a MOV",
			build: func(ctx *build.Context) {
				ctx.CMPB(reg.AL, reg.CL)
				ctx.MOVQ(operand.U64(1), reg.RAX)
				ctx.JLT(operand.LabelRef("l"))
			},
			want: "sub-32-bit compare",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := build.NewContext()
			ctx.Function("guard")
			ctx.SignatureExpr("func()")
			c.build(ctx)
			ctx.Label("l")
			ctx.RET()
			f, errs := ctx.Result()
			if errs != nil {
				t.Fatal(errs)
			}
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected a panic, got none")
				}
				msg, ok := r.(string)
				if !ok || !strings.Contains(msg, c.want) {
					t.Fatalf("panic %q does not mention %q", r, c.want)
				}
			}()
			_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
		})
	}
}
