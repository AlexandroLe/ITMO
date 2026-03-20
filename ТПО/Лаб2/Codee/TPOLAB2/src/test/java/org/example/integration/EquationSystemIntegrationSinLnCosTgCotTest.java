package org.example.integration;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.example.EquationSystem;
import org.example.function.AbstractMathFunction;
import org.example.log.Ln;
import org.example.trig.Cos;
import org.example.trig.Cot;
import org.example.trig.Sin;
import org.example.trig.Tan;

import java.math.BigDecimal;
import java.math.MathContext;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

public class EquationSystemIntegrationSinLnCosTgCotTest {

    private static final BigDecimal EPS = new BigDecimal("0.000000000000001");
    private static final MathContext MC = MathContext.DECIMAL128;

    @Test
    public void testLeftBranchAtMinusPiOverFour() {
        assertLeftBranch(
                new BigDecimal("-0.7853981633974483"),   // -π/4
                new BigDecimal("1.414213562373095"),     // sec
                new BigDecimal("-1.414213562373095")     // csc
        );
    }

    @Test
    public void testLeftBranchAtMinusOne() {
        assertLeftBranch(
                new BigDecimal("-1"),
                new BigDecimal("1.8508157176809255"),     // sec(-1)
                new BigDecimal("-1.1883951057781212")     // csc(-1), approx
        );
    }

    @Test
    public void testLeftBranchAtMinusTwo() {
        assertLeftBranch(
                new BigDecimal("-2"),
                new BigDecimal("-2.402997961722380"),     // sec(-2)
                new BigDecimal("-1.0997501702946164")     // csc(-2)
        );
    }


    @Test
    public void testRightBranchAtTwo() {
        assertRightBranch(
                new BigDecimal("2"),
                new BigDecimal("0.6309297535714574"),     // log3(2)
                new BigDecimal("0.43067655807339306")     // log5(2)
        );
    }

    @Test
    public void testRightBranchAtE() {
        assertRightBranch(
                new BigDecimal("2.718281828459045"),
                new BigDecimal("0.9102392266268373"),     // log3(e)
                new BigDecimal("0.6213349345596119")      // log5(e)
        );
    }

    private void assertLeftBranch(BigDecimal x,
                                  BigDecimal secVal,
                                  BigDecimal cscVal) {

        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Ln ln = new Ln();
        Tan tan = new Tan(sin, cos);
        Cot cot = new Cot(tan);

        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal sinVal = sin.calculate(x, EPS);
        BigDecimal cosVal = cos.calculate(x, EPS);
        BigDecimal tanVal = tan.calculate(x, EPS);
        BigDecimal cotVal = cot.calculate(x, EPS);

        when(sec.calculate(eq(x), eq(EPS))).thenReturn(secVal);
        when(csc.calculate(eq(x), eq(EPS))).thenReturn(cscVal);

        EquationSystem system = new EquationSystem(
                sin, cos, tan, cot, sec, csc,
                ln, log3, log5
        );

        BigDecimal result = system.calculate(x, EPS);

        BigDecimal step1 = cscVal.divide(tanVal, MC);
        BigDecimal step2 = step1.subtract(cosVal, MC);
        BigDecimal step3 = step2.pow(3, MC);
        BigDecimal step4 = secVal.subtract(sinVal, MC);
        BigDecimal step5 = step3.subtract(step4, MC);
        BigDecimal step6 = step5.pow(2, MC);
        BigDecimal step7 = step6.multiply(cotVal.pow(3, MC), MC);
        BigDecimal step8 = step7.multiply(sinVal, MC);
        BigDecimal expected = step8.pow(3, MC).pow(3, MC);

        BigDecimal delta = result.subtract(expected).abs();
        assertTrue(delta.compareTo(EPS) <= 0,
                "expected=" + expected + ", actual=" + result + ", delta=" + delta);
    }

    private void assertRightBranch(BigDecimal x,
                                   BigDecimal log3Val,
                                   BigDecimal log5Val) {

        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Ln ln = new Ln();
        Tan tan = new Tan(sin, cos);
        Cot cot =  new Cot(tan);

        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal lnVal = ln.calculate(x, EPS);

        when(log3.calculate(eq(x), eq(EPS))).thenReturn(log3Val);
        when(log5.calculate(eq(x), eq(EPS))).thenReturn(log5Val);

        EquationSystem system = new EquationSystem(
                sin, cos, tan, cot, sec, csc,
                ln, log3, log5
        );

        BigDecimal result = system.calculate(x, EPS);

        BigDecimal part1 = log5Val.multiply(log5Val, MC).add(log3Val, MC);
        BigDecimal part2 = log3Val.divide(log3Val, MC);
        BigDecimal left = part1.multiply(part2, MC).add(log3Val, MC);
        BigDecimal right = lnVal.subtract(log5Val.divide(log5Val, MC), MC);
        BigDecimal expected = left.multiply(right, MC);

        BigDecimal delta = result.subtract(expected).abs();
        assertTrue(delta.compareTo(EPS) <= 0,
                "expected=" + expected + ", actual=" + result + ", delta=" + delta);
    }

}