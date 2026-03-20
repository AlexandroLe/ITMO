package org.example.unit.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.trig.Csc;
import org.example.trig.Sin;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.*;

class CscTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.000001;
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);
    private static final BigDecimal PI = new BigDecimal(Math.PI, MC);
    private static final BigDecimal TWO = new BigDecimal("2");
    private static final BigDecimal HALF_PI = PI.divide(TWO, MC);
    private static final BigDecimal THREE = new BigDecimal("3");
    private Csc csc;

    @BeforeEach
    void init() {
        Sin sin = new Sin();
        csc = new Csc(sin);
    }

    @Test
    void shouldCalculateForPiHalf() {
        BigDecimal arg = HALF_PI;
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForNegativePiHalf() {
        BigDecimal arg = HALF_PI.negate();
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForThreePiHalf() {
        BigDecimal threePiHalf = THREE.multiply(HALF_PI, MC);
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(threePiHalf, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC);
        BigDecimal expected = new BigDecimal("2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiFour() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal expected = new BigDecimal("1.4142136").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC);
        BigDecimal expected = new BigDecimal("1.1547005").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldNotCalculateForZero() {
        assertThrows(ArithmeticException.class, () -> csc.calculate(ZERO, PRECISION));
    }

    @Test
    void shouldNotCalculateForPi() {
        assertThrows(ArithmeticException.class, () -> csc.calculate(PI, PRECISION));
    }

    @Test
    void shouldNotCalculateForNegativePi() {
        assertThrows(ArithmeticException.class, () -> csc.calculate(PI.negate(), PRECISION));
    }

    @Test
    void shouldNotCalculateForTwoPi() {
        BigDecimal twoPi = TWO.multiply(PI, MC);
        assertThrows(ArithmeticException.class, () -> csc.calculate(twoPi, PRECISION));
    }

    @Test
    void shouldNotCalculateForLargeValue() {
        BigDecimal hundredPi = new BigDecimal("100").multiply(PI, MC);
        assertThrows(ArithmeticException.class, () -> csc.calculate(hundredPi, PRECISION));
    }

    @Test
    void shouldCalculateForNegativePiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC).negate();
        BigDecimal expected = new BigDecimal("-2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = csc.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateNearAsymptoteFromRight() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = epsilon;
        BigDecimal result = csc.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) > 0);
    }

    @Test
    void shouldCalculateNearAsymptoteFromLeft() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = PI.subtract(epsilon, MC);
        BigDecimal result = csc.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) > 0);
    }

    @Test
    void shouldCalculateNearAsymptoteFromLeftNegative() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = PI.negate().add(epsilon, MC);
        BigDecimal result = csc.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) < 0);
    }

    @Test
    void shouldBeOddFunction() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal positive = csc.calculate(arg, PRECISION);
        BigDecimal negative = csc.calculate(arg.negate(), PRECISION);
        assertEquals(positive.doubleValue(), negative.negate().doubleValue(), DELTA);
    }

    @Test
    void shouldHavePeriodTwoPi() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal argPlusTwoPi = arg.add(TWO.multiply(PI, MC), MC);
        BigDecimal expected = csc.calculate(arg, PRECISION);
        BigDecimal actual = csc.calculate(argPlusTwoPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }
}