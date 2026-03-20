package org.example.unit.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.trig.Cos;
import org.example.trig.Sec;
import org.example.trig.Sin;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.*;

class SecTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.000001;
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);
    private static final BigDecimal PI = new BigDecimal(Math.PI, MC);
    private static final BigDecimal TWO = new BigDecimal("2");
    private static final BigDecimal HALF_PI = PI.divide(TWO, MC);
    private static final BigDecimal THREE = new BigDecimal("3");
    private Sec sec;

    @BeforeEach
    void init() {
        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        sec = new Sec(cos);
    }

    @Test
    void shouldCalculateForZero() {
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(ZERO, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPi() {
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(PI, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC);
        BigDecimal expected = new BigDecimal("2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiFour() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal expected = new BigDecimal("1.4142136").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForPiSix() {
        BigDecimal arg = PI.divide(new BigDecimal("6"), MC);
        BigDecimal expected = new BigDecimal("1.1547005").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldNotCalculateForPiHalf() {
        assertThrows(ArithmeticException.class, () -> sec.calculate(HALF_PI, PRECISION));
    }

    @Test
    void shouldNotCalculateForNegativePiHalf() {
        assertThrows(ArithmeticException.class, () -> sec.calculate(HALF_PI.negate(), PRECISION));
    }

    @Test
    void shouldNotCalculateForThreePiHalf() {
        BigDecimal threePiHalf = THREE.multiply(HALF_PI, MC);
        assertThrows(ArithmeticException.class, () -> sec.calculate(threePiHalf, PRECISION));
    }

    @Test
    void shouldCalculateForNegativePiThree() {
        BigDecimal arg = PI.divide(new BigDecimal("3"), MC).negate();
        BigDecimal expected = new BigDecimal("2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateNearAsymptoteFromLeft() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = HALF_PI.subtract(epsilon, MC);
        BigDecimal result = sec.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) > 0);
    }

    @Test
    void shouldCalculateNearAsymptoteFromRight() {
        BigDecimal epsilon = new BigDecimal("0.0001");
        BigDecimal arg = HALF_PI.add(epsilon, MC);
        BigDecimal result = sec.calculate(arg, PRECISION);
        assertTrue(result.compareTo(BigDecimal.ZERO) < 0);
    }

    @Test
    void shouldBeEvenFunction() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal positive = sec.calculate(arg, PRECISION);
        BigDecimal negative = sec.calculate(arg.negate(), PRECISION);
        assertEquals(positive.doubleValue(), negative.doubleValue(), DELTA);
    }

    @Test
    void shouldHavePeriodTwoPi() {
        BigDecimal arg = PI.divide(new BigDecimal("4"), MC);
        BigDecimal argPlusTwoPi = arg.add(TWO.multiply(PI, MC), MC);
        BigDecimal expected = sec.calculate(arg, PRECISION);
        BigDecimal actual = sec.calculate(argPlusTwoPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForLargeValue() {
        BigDecimal hundredPi = new BigDecimal("100").multiply(PI, MC);
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = sec.calculate(hundredPi, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }
}