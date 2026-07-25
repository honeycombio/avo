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

// TestARM64AddCarryGuard checks that a carry-condition consumer after an
// addition fails generation. x86 and arm64 both set carry-out on an add, so the
// condition map -- which translates by post-compare meaning, where the two use
// opposite borrow conventions -- would invert the test.
func TestARM64AddCarryGuard(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("addcarry")
	ctx.SignatureExpr("func()")
	ctx.ADDQ(reg.RAX, reg.RCX)
	ctx.JCS(operand.LabelRef("wrap"))
	ctx.Label("wrap")
	ctx.RET()

	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic for a carry condition after ADD")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "only a compare or subtract") {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
}

// TestARM64HighByteEncodability checks that byte extends from a high-byte
// register are refused where x86-64 cannot encode them. Any REX-carrying form
// renames that operand to SPL, so emitting a faithful arm64 extract would make
// the two architectures compute different values from one program.
func TestARM64HighByteEncodability(t *testing.T) {
	cases := []struct {
		name  string
		build func(ctx *build.Context)
		want  string
	}{
		{
			name:  "64-bit destination",
			build: func(ctx *build.Context) { ctx.MOVBQZX(reg.AH, reg.RBX) },
			want:  "forces a REX prefix",
		},
		{
			name:  "extended destination",
			build: func(ctx *build.Context) { ctx.MOVBLZX(reg.AH, reg.R9L) },
			want:  "forces a REX prefix",
		},
		// The pairings below all encode on amd64 rather than being rejected --
		// the assembler silently substitutes SPL -- so each is a case where the
		// two architectures would otherwise compute different things.
		{
			name:  "byte move to extended register",
			build: func(ctx *build.Context) { ctx.MOVB(reg.AH, reg.R8B) },
			want:  "forces a REX prefix",
		},
		{
			name:  "byte move from extended register",
			build: func(ctx *build.Context) { ctx.MOVB(reg.R8B, reg.AH) },
			want:  "forces a REX prefix",
		},
		{
			name:  "byte move through extended base",
			build: func(ctx *build.Context) { ctx.MOVB(reg.AH, operand.Mem{Base: reg.R8}) },
			want:  "forces a REX prefix",
		},
		{
			// SIL is not an extended register, but it exists only under REX:
			// without the prefix that encoding names AH.
			name:  "byte add against SIL",
			build: func(ctx *build.Context) { ctx.ADDB(reg.SIB, reg.AH) },
			want:  "forces a REX prefix",
		},
		{
			name: "sub-word compare against extended register",
			build: func(ctx *build.Context) {
				ctx.CMPB(reg.AH, reg.R8B)
				ctx.JEQ(operand.LabelRef("hb_yes"))
				ctx.Label("hb_yes")
			},
			want: "forces a REX prefix",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := build.NewContext()
			ctx.Function("hb")
			ctx.SignatureExpr("func()")
			c.build(ctx)
			ctx.RET()
			f, errs := ctx.Result()
			if errs != nil {
				t.Fatal(errs)
			}
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected a panic")
				}
				if msg, ok := r.(string); !ok || !strings.Contains(msg, c.want) {
					t.Fatalf("unexpected panic: %v", r)
				}
			}()
			_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
		})
	}
}

// TestARM64NestedForeignConditional checks that an #endif closes the innermost
// conditional even when a GOAMD64 one is nested inside a directive this printer
// does not evaluate. Resolving them out of order would drop live code.
func TestARM64NestedForeignConditional(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("nested")
	ctx.SignatureExpr("func()")
	ctx.Comment("#ifdef SOMETHING_ELSE")
	ctx.Comment("#ifdef GOAMD64_v3")
	ctx.MOVQ(operand.U64(1), reg.RAX) // dead on arm64
	ctx.Comment("#endif")
	ctx.MOVQ(operand.U64(2), reg.RCX) // live: inside the foreign conditional only
	ctx.Comment("#endif")
	ctx.RET()

	out := printARM64(t, ctx, printer.NewGoRunConfig())
	if strings.Contains(out, "$0x0000000000000001") {
		t.Errorf("dead GOAMD64 arm was emitted:\n%s", out)
	}
	if !strings.Contains(out, "$0x0000000000000002") {
		t.Errorf("live code after the inner #endif was dropped:\n%s", out)
	}
}

// TestARM64ConstWindowStopsAtDirective checks that the constant window does not
// carry a folded value across a preprocessor directive this printer cannot
// evaluate. If the MOV that established the constant sits inside an arm the
// assembler later drops, a fold emitted after the #endif is unconditional and
// wrong.
//
// The address cache used to need the same guard and had its own subtest here.
// It is gone: it bought two instructions across the whole of zstd and huff0,
// because almost every access folds into the instruction and never materializes
// an address, and it cost two silent miscompiles. The constant window stays --
// it is not merely an optimization. It is the only path that implements x86's
// saturating semantics for BZHI/BEXTR counts >= 64; deleting it turned correct
// programs into wrong ones, which the differential suite caught on arm64.
func TestARM64ConstWindowStopsAtDirective(t *testing.T) {
	for _, c := range []struct{ name, open, close string }{
		{"Plain", "#ifdef FOO", "#endif"},
		// Go's assembler tokenizes the '#' separately, so this is live too.
		{"Spaced", "# ifdef FOO", "# endif"},
	} {
		t.Run(c.name, func(t *testing.T) {
			ctx := build.NewContext()
			ctx.Function("cw")
			ctx.SignatureExpr("func()")
			ctx.MOVQ(operand.U64(5), reg.RCX)
			ctx.Comment(c.open)
			ctx.MOVQ(operand.U64(9), reg.RCX)
			ctx.Comment(c.close)
			ctx.SHLXQ(reg.RCX, reg.RAX, reg.RAX)
			ctx.RET()

			out := printARM64(t, ctx, printer.NewGoRunConfig())
			// Either folded constant would be wrong: $9 came from inside the arm,
			// $5 assumes the arm was not taken. The shift must read the register.
			if strings.Contains(out, "LSL $9") || strings.Contains(out, "LSL $5") {
				t.Errorf("constant folded across a preprocessor directive:\n%s", out)
			}
		})
	}
}

// dispatch loop, and tests here that fail when that barrier is removed.

// TestARM64NoProducerRejected checks that a flag consumer with no producer
// anywhere in the function fails generation rather than branching on whatever
// NZCV the caller happened to leave.
func TestARM64NoProducerRejected(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("noprod")
	ctx.SignatureExpr("func()")
	ctx.JEQ(operand.LabelRef("np_yes"))
	ctx.Label("np_yes")
	ctx.RET()

	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic for a consumer with no producer")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "no producer") {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
}

// TestARM64DirectiveNodeIntegrity covers the two ways liveNodes could drop
// content silently. Both concern the same asymmetry: this printer resolves
// GOAMD64 conditionals while the amd64 side keeps them, so anything discarded
// along with a resolved directive is discarded from only one of the two
// binaries.
func TestARM64DirectiveNodeIntegrity(t *testing.T) {
	t.Run("MixedComment", func(t *testing.T) {
		// The #endif is owned and resolved away; the #include is not, and would
		// survive into the amd64 output while vanishing here.
		ctx := build.NewContext()
		ctx.Function("mixed")
		ctx.SignatureExpr("func()")
		ctx.Comment("#ifdef GOAMD64_v3")
		ctx.Comment("#endif", "#include \"extra.h\"")
		ctx.RET()

		f, errs := ctx.Result()
		if errs != nil {
			t.Fatal(errs)
		}
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected a panic for a comment mixing owned and foreign directives")
			}
			if msg, ok := r.(string); !ok || !strings.Contains(msg, "one directive per comment") {
				t.Fatalf("unexpected panic: %v", r)
			}
		}()
		_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
	})

	t.Run("UnclosedConditional", func(t *testing.T) {
		// Without the guard this emits a TEXT block with an empty body -- the
		// RET included -- and says nothing about it.
		ctx := build.NewContext()
		ctx.Function("unclosed")
		ctx.SignatureExpr("func()")
		ctx.Comment("#ifdef GOAMD64_v3")
		ctx.RET()

		f, errs := ctx.Result()
		if errs != nil {
			t.Fatal(errs)
		}
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected a panic for an unclosed GOAMD64 conditional")
			}
			if msg, ok := r.(string); !ok || !strings.Contains(msg, "unclosed") {
				t.Fatalf("unexpected panic: %v", r)
			}
		}()
		_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
	})

	t.Run("ForeignUnclosedIsAllowed", func(t *testing.T) {
		// A conditional this printer does not own is the assembler's business;
		// leaving it open must not be rejected here.
		ctx := build.NewContext()
		ctx.Function("foreignopen")
		ctx.SignatureExpr("func()")
		ctx.Comment("#ifdef SOMETHING_ELSE")
		ctx.RET()

		out := printARM64(t, ctx, printer.NewGoRunConfig())
		if !strings.Contains(out, "RET") {
			t.Errorf("expected the body to survive a foreign conditional:\n%s", out)
		}
	})
}

// TestARM64SpacedDirectivesResolved covers directives written with a space
// after the '#'. Go's assembler tokenizes the '#' separately, so "# else" is a
// live directive; an exact-string match misses it, and missing an #else means
// the evaluator never toggles the arm, so BOTH arms get dropped.
func TestARM64SpacedDirectivesResolved(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("spaced")
	ctx.SignatureExpr("func()")
	ctx.Comment("# ifdef GOAMD64_v3")
	ctx.ADDQ(operand.U32(10), reg.RAX) // amd64-only arm
	ctx.Comment("# else")
	ctx.ADDQ(operand.U32(10), reg.RCX) // the arm arm64 must take
	ctx.Comment("# endif")
	ctx.RET()

	out := printARM64(t, ctx, printer.NewGoRunConfig())
	if !strings.Contains(out, "ADD $0x0000000a, R1, R1") {
		t.Errorf("spaced #else was not resolved; the live arm is missing:\n%s", out)
	}
	if strings.Contains(out, "R0, R0") {
		t.Errorf("spaced #ifdef was not resolved; the dead arm survived:\n%s", out)
	}
}

// TestARM64ForeignElseMixedComment covers ownership being a property of the
// preprocessor stack rather than of the text. A comment can carry an #else
// belonging to a FOREIGN conditional alongside a directive this printer owns;
// dropping the node whole then moves code inside an arm it was never in, with
// balanced directives on both sides so no assembler ever complains.
func TestARM64ForeignElseMixedComment(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("foreignelse")
	ctx.SignatureExpr("func()")
	ctx.Comment("#ifdef MYFLAG")
	ctx.ADDQ(operand.U32(1), reg.RAX)
	ctx.Comment("#else", "#ifdef GOAMD64_v4") // the #else is MYFLAG's, not ours
	ctx.ADDQ(operand.U32(50), reg.RAX)
	ctx.Comment("#endif")
	ctx.ADDQ(operand.U32(10), reg.RAX)
	ctx.Comment("#endif")
	ctx.RET()

	f, errs := ctx.Result()
	if errs != nil {
		t.Fatal(errs)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected a panic for a comment mixing an owned directive with a foreign #else")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "one directive per comment") {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
}

// TestARM64DoubleShiftRejected checks the three-operand SHL/SHR forms, which
// Go's assembler encodes as the double-precision shifts SHLD/SHRD -- a
// different instruction, whose destination is the third operand.
func TestARM64DoubleShiftRejected(t *testing.T) {
	for _, op := range []string{"SHLQ", "SHRQ", "SHLL", "SHRL"} {
		t.Run(op, func(t *testing.T) {
			ctx := build.NewContext()
			ctx.Function("dbl")
			ctx.SignatureExpr("func()")
			switch op {
			case "SHLQ":
				ctx.SHLQ(operand.U8(8), reg.RDX, reg.RAX)
			case "SHRQ":
				ctx.SHRQ(operand.U8(8), reg.RDX, reg.RAX)
			case "SHLL":
				ctx.SHLL(operand.U8(8), reg.EDX, reg.EAX)
			case "SHRL":
				ctx.SHRL(operand.U8(8), reg.EDX, reg.EAX)
			}
			ctx.RET()

			f, errs := ctx.Result()
			if errs != nil {
				t.Fatal(errs)
			}
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected a panic for three-operand %s", op)
				}
				if msg, ok := r.(string); !ok || !strings.Contains(msg, "double-precision shift") {
					t.Fatalf("unexpected panic: %v", r)
				}
			}()
			_, _ = printer.NewARM64Asm(printer.NewGoRunConfig()).Print(f)
		})
	}
}

// TestARM64PseudoSPLocalsOffset checks that avo's frame-relative locals land
// above the saved link register. x86 puts local 0 at SP+0 because CALL has
// already pushed the return address above the frame; arm64 spills the link
// register to the bottom of the frame, so locals start at 8(RSP).
func TestARM64PseudoSPLocalsOffset(t *testing.T) {
	ctx := build.NewContext()
	ctx.Function("locals")
	ctx.SignatureExpr("func()")
	ctx.MOVQ(reg.RAX, operand.Mem{Base: reg.StackPointer})          // avo local 0
	ctx.MOVQ(operand.Mem{Base: reg.StackPointer, Disp: 8}, reg.RCX) // avo local 8
	ctx.RET()

	out := printARM64(t, ctx, printer.NewGoRunConfig())
	if strings.Contains(out, ", (RSP)") || strings.Contains(out, "(RSP), ") && strings.Contains(out, " (RSP)") {
		t.Errorf("a local was emitted at 0(RSP), which is the saved link register:\n%s", out)
	}
	if !strings.Contains(out, "8(RSP)") || !strings.Contains(out, "16(RSP)") {
		t.Errorf("expected locals shifted to 8(RSP) and 16(RSP):\n%s", out)
	}
}
