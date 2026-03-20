package org.example.unit.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.trig.Cos;
import org.example.trig.Sin;
import org.example.trig.Tan;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.*;

class TanTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.000001; 
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);
    private static final BigDecimal PI = new BigDecimal(Math.PI, MC);
    private static final BigDecimal TWO = new BigDecimal("2");
    private static final BigDecimal HALF_PI = PI.divide(TWO, MC);
    private static final BigDecimal THREE = new BigDecimal("3");
    private Tan tan;

    @BeforeEach
    void init() {
        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        tan = new Tan(sin, cos);
    }

    @Test
    void shouldCalculateForZero() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(ZERO, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiFour() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePiFour() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC).negate();
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC);
        BigDecimal expected = new BigDecimal("0.5773503").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC);
        BigDecimal expected = new BigDecimal("1.7320508").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldNotCalculateForPiHalf() {
        BigDecimal arg = HALF_PI;
        assertThrows(ArithmeticException.class, () -> tan.calculate(arg, PRECISION));
    }

    @Test
    void shouldNotCalculateForNegativePiHalf() {
        BigDecimal arg = HALF_PI.negate();
        assertThrows(ArithmeticException.class, () -> tan.calculate(arg, PRECISION));
    }

    @Test
    void shouldNotCalculateForThreePiHalf() {
        BigDecimal threePiHalf = THREE.multiply(HALF_PI, MC);
        assertThrows(ArithmeticException.class, () -> tan.calculate(threePiHalf, PRECISION));
    }

    @Test
    void shouldCalculateForPiPlusPiFour() {
        BigDecimal arg = PI.add(PI.divide(new BigDecimal("4"), MC), MC);
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC).negate();
        BigDecimal expected = new BigDecimal("-1.7320508").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateNearAsymptoteFromLeft() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = HALF_PI.subtract(epsilon, MC);
        BigDecimal result = tan.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) > 0);
    }

    @Test
    void shouldCalculateNearAsymptoteFromRight() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = HALF_PI.add(epsilon, MC);
        BigDecimal result = tan.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) < 0);
    }

    @Test
    void shouldCalculateForLargeValue() {
        BigDecimal hundredPi = new BigDecimal("100").multiply(PI, MC);
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = tan.calculate(hundredPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldBeOddFunction() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal positive = tan.calculate(arg, PRECISION);
        BigDecimal negative = tan.calculate(arg.negate(), PRECISION);
        assertEquals(positive.doubleValue(), negative.negate().doubleValue(), DELTA);
    }

    @Test
    void shouldHavePeriodPi() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal argPlusPi = arg.add(PI, MC);
        BigDecimal expected = tan.calculate(arg, PRECISION);
        BigDecimal actual = tan.calculate(argPlusPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }
}