package org.example.unit.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.trig.Sin;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SinTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.0000001;
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);
    private static final BigDecimal PI = new BigDecimal(Math.PI, MC);
    private static final BigDecimal TWO = new BigDecimal("2");
    private static final BigDecimal HALF_PI = PI.divide(TWO, MC);
    private Sin sin;

    @BeforeEach
    void init() {
        sin = new Sin();
    }

    @Test
    void shouldCalculateForZero() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(ZERO, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiHalf() {
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(HALF_PI, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePiHalf() {
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(HALF_PI.negate(), PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPi() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(PI, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePi() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(PI.negate(), PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC);
        BigDecimal expected = new BigDecimal("0.5").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiFour() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal expected = new BigDecimal("0.70710678").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC);
        BigDecimal expected = new BigDecimal("0.8660254").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC).negate();
        BigDecimal expected = new BigDecimal("-0.5").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForTwoPi() {
        BigDecimal twoPi = TWO.multiply(PI, MC);
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(twoPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForFivePiHalf() {
        BigDecimal fivePiHalf = new BigDecimal("5").multiply(HALF_PI, MC);
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(fivePiHalf, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForLargeValue() {
        BigDecimal hundredPi = new BigDecimal("100").multiply(PI, MC);
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sin.calculate(hundredPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldBeOddFunction() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal positive = sin.calculate(arg, PRECISION);
        BigDecimal negative = sin.calculate(arg.negate(), PRECISION);
        assertEquals(positive.doubleValue(), negative.negate().doubleValue(), DELTA);
    }
}