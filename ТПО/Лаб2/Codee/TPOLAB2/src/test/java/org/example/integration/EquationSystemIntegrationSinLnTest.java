package org.example.integration;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.example.EquationSystem;
import org.example.function.AbstractMathFunction;
import org.example.log.Ln;
import org.example.trig.Sin;

import java.math.BigDecimal;
import java.math.MathContext;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

public class EquationSystemIntegrationSinLnTest {

    private static final BigDecimal EPS = new BigDecimal("0.000000000000001");
    private static final MathContext MC = MathContext.DECIMAL128;

    @Test
    public void testLeftBranchAtMinusPiOverFour() {

        BigDecimal x = new BigDecimal("-0.7853981633974483"); // -π/4

        Sin sin = new Sin();
        Ln ln = new Ln();

        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal sinVal = sin.calculate(x, EPS);
        BigDecimal cosVal = new BigDecimal("0.7071067811865476");
        BigDecimal tanVal = new BigDecimal("-1.0");
        BigDecimal cotVal = new BigDecimal("-1.0");
        BigDecimal secVal = new BigDecimal("1.414213562373095");
        BigDecimal cscVal = new BigDecimal("-1.414213562373095");

        when(cos.calculate(eq(x), eq(EPS))).thenReturn(cosVal);
        when(tan.calculate(eq(x), eq(EPS))).thenReturn(tanVal);
        when(cot.calculate(eq(x), eq(EPS))).thenReturn(cotVal);
        when(sec.calculate(eq(x), eq(EPS))).thenReturn(secVal);
        when(csc.calculate(eq(x), eq(EPS))).thenReturn(cscVal);

        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

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
        assertTrue(delta.compareTo(EPS) <= 0);
    }

    @Test
    public void testLeftBranchAtMinusOne() {

        BigDecimal x = new BigDecimal("-1");

        Sin sin = new Sin();
        Ln ln = new Ln();

        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal sinVal = sin.calculate(x, EPS);

        BigDecimal cosVal = new BigDecimal(Double.toString(Math.cos(x.doubleValue())));
        BigDecimal tanVal = new BigDecimal(Double.toString(Math.tan(x.doubleValue())));
        BigDecimal cotVal = BigDecimal.ONE.divide(tanVal, MC);
        BigDecimal secVal = BigDecimal.ONE.divide(cosVal, MC);
        BigDecimal cscVal = BigDecimal.ONE.divide(sinVal, MC);

        when(cos.calculate(eq(x), eq(EPS))).thenReturn(cosVal);
        when(tan.calculate(eq(x), eq(EPS))).thenReturn(tanVal);
        when(cot.calculate(eq(x), eq(EPS))).thenReturn(cotVal);
        when(sec.calculate(eq(x), eq(EPS))).thenReturn(secVal);
        when(csc.calculate(eq(x), eq(EPS))).thenReturn(cscVal);

        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

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
        assertTrue(delta.compareTo(EPS) <= 0);
    }

    @Test
    public void testRightBranchAtTwo() {

        BigDecimal x = new BigDecimal("2");

        Sin sin = new Sin();
        Ln ln = new Ln();

        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal lnVal = ln.calculate(x, EPS);
        BigDecimal log3Val = new BigDecimal("0.6309297535714574");
        BigDecimal log5Val = new BigDecimal("0.43067655807339306");

        when(log3.calculate(eq(x), eq(EPS))).thenReturn(log3Val);
        when(log5.calculate(eq(x), eq(EPS))).thenReturn(log5Val);

        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

        BigDecimal result = system.calculate(x, EPS);

        BigDecimal part1 = log5Val.multiply(log5Val, MC).add(log3Val, MC);
        BigDecimal part2 = log3Val.divide(log3Val, MC);
        BigDecimal left = part1.multiply(part2, MC).add(log3Val, MC);
        BigDecimal right = lnVal.subtract(log5Val.divide(log5Val, MC), MC);
        BigDecimal expected = left.multiply(right, MC);

        BigDecimal delta = result.subtract(expected).abs();
        assertTrue(delta.compareTo(EPS) <= 0);
    }

    @Test
    public void testRightBranchAtE() {

        BigDecimal x = new BigDecimal("2.718281828459045");

        Sin sin = new Sin();
        Ln ln = new Ln();

        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        BigDecimal lnVal = ln.calculate(x, EPS);
        BigDecimal log3Val = new BigDecimal("0.9102392266268373");
        BigDecimal log5Val = new BigDecimal("0.6213349345596118");

        when(log3.calculate(eq(x), eq(EPS))).thenReturn(log3Val);
        when(log5.calculate(eq(x), eq(EPS))).thenReturn(log5Val);

        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

        BigDecimal result = system.calculate(x, EPS);

        BigDecimal part1 = log5Val.multiply(log5Val, MC).add(log3Val, MC);
        BigDecimal part2 = log3Val.divide(log3Val, MC);
        BigDecimal left = part1.multiply(part2, MC).add(log3Val, MC);
        BigDecimal right = lnVal.subtract(log5Val.divide(log5Val, MC), MC);
        BigDecimal expected = left.multiply(right, MC);

        BigDecimal delta = result.subtract(expected).abs();
        assertTrue(delta.compareTo(EPS) <= 0);
    }
}