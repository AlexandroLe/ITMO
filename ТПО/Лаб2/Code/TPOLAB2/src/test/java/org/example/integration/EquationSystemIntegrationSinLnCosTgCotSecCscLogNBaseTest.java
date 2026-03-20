package org.example.integration;

import org.junit.jupiter.api.Test;
import org.example.EquationSystem;
import org.example.log.Ln;
import org.example.log.LogNBase;
import org.example.trig.*;

import java.math.BigDecimal;
import java.math.MathContext;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class EquationSystemIntegrationSinLnCosTgCotSecCscLogNBaseTest {

    private static final BigDecimal EPS = new BigDecimal("0.000000000000001");
    private static final MathContext MC = MathContext.DECIMAL128;

    @Test
    public void testLeftBranchAtMinusPiOverFour() {
        assertLeftBranch(
                new BigDecimal("-0.7853981633974483")   // -π/4
        );
    }

    @Test
    public void testLeftBranchAtMinusOne() {
        assertLeftBranch(
                new BigDecimal("-1")
        );
    }

    @Test
    public void testLeftBranchAtMinusTwo() {
        assertLeftBranch(
                new BigDecimal("-2")
        );
    }


    @Test
    public void testRightBranchAtTwo() {
        assertRightBranch(
                new BigDecimal("2")
        );
    }

    @Test
    public void testRightBranchAtE() {
        assertRightBranch(
                new BigDecimal("2.718281828459045")
        );
    }

    private void assertLeftBranch(BigDecimal x) {

        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Ln ln = new Ln();
        Tan tan = new Tan(sin, cos);
        Cot cot = new Cot(tan);
        Sec sec = new Sec(cos);
        Csc csc = new Csc(sin);
        LogNBase log3 = new LogNBase(ln, 3);
        LogNBase log5 = new LogNBase(ln, 5);

        BigDecimal sinVal = sin.calculate(x, EPS);
        BigDecimal cosVal = cos.calculate(x, EPS);
        BigDecimal tanVal = tan.calculate(x, EPS);
        BigDecimal cotVal = cot.calculate(x, EPS);
        BigDecimal secVal = sec.calculate(x, EPS);
        BigDecimal cscVal = csc.calculate(x, EPS);

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

    private void assertRightBranch(BigDecimal x) {

        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Ln ln = new Ln();
        Tan tan = new Tan(sin, cos);
        Cot cot =  new Cot(tan);
        Sec sec = new Sec(cos);
        Csc csc = new Csc(sin);
        LogNBase log3 = new LogNBase(ln, 3);
        LogNBase log5 = new LogNBase(ln, 5);

        BigDecimal lnVal = ln.calculate(x, EPS);
        BigDecimal log3Val = log3.calculate(x, EPS);
        BigDecimal log5Val = log5.calculate(x, EPS);

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