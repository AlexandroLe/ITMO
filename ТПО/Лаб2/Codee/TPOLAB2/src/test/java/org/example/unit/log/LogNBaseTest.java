package org.example.unit.log;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.log.Ln;
import org.example.log.LogNBase;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.*;

class LogNBaseTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.000001;
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);

    private LogNBase log3;
    private LogNBase log5;

    @BeforeEach
    void init() {
        Ln ln = new Ln();
        log3 = new LogNBase(ln, 3);
        log5 = new LogNBase(ln, 5);
    }

    @Test
    void shouldThrowExceptionForInvalidBase() {
        Ln ln = new Ln();
        assertThrows(IllegalArgumentException.class, () -> new LogNBase(ln, 0));
        assertThrows(IllegalArgumentException.class, () -> new LogNBase(ln, -5));
        assertThrows(IllegalArgumentException.class, () -> new LogNBase(ln, 1));
    }

    @Test
    void shouldNotCalculateForZero() {
        assertThrows(ArithmeticException.class, () -> log3.calculate(ZERO, PRECISION));
    }

    @Test
    void shouldNotCalculateForNegative() {
        assertThrows(ArithmeticException.class, () -> log3.calculate(new BigDecimal("-1"), PRECISION));
    }

    @Test
    void shouldCalculateForOne() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(ONE, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForBaseThree() {
        BigDecimal arg = new BigDecimal("3");
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForBaseFive() {
        BigDecimal arg = new BigDecimal("5");
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log5.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForBaseNine() {
        BigDecimal arg = new BigDecimal("9");
        BigDecimal expected = new BigDecimal("2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForBaseTwentySeven() {
        BigDecimal arg = new BigDecimal("27");
        BigDecimal expected = new BigDecimal("3").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForOneThird() {
        BigDecimal arg = ONE.divide(new BigDecimal("3"), MC);
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForOneNinth() {
        BigDecimal arg = ONE.divide(new BigDecimal("9"), MC);
        BigDecimal expected = new BigDecimal("-2").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForFiveBaseThree() {
        BigDecimal arg = new BigDecimal("5");
        BigDecimal expected = new BigDecimal("1.4649735").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log3.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldThrowForThousandBaseThree() {
        BigDecimal arg = new BigDecimal("1000");

        assertThrows(ArithmeticException.class,
                () -> log3.calculate(arg, PRECISION));
    }

    @Test
    void shouldThrowForOneThousandthBaseThree() {
        BigDecimal arg = new BigDecimal("0.001");

        assertThrows(ArithmeticException.class,
                () -> log3.calculate(arg, PRECISION));
    }

    @Test
    void shouldCalculateForDifferentBases() {
        BigDecimal arg = new BigDecimal("8");
        LogNBase log2 = new LogNBase(new Ln(), 2);

        BigDecimal expected = new BigDecimal("3").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = log2.calculate(arg, PRECISION);

        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldBeNegativeForLessThanOne() {
        BigDecimal arg = new BigDecimal("0.5");
        BigDecimal result = log3.calculate(arg, PRECISION);

        assertTrue(result.compareTo(ZERO) < 0);
    }

    @Test
    void shouldBePositiveForGreaterThanOne() {
        BigDecimal arg = new BigDecimal("2");
        BigDecimal result = log3.calculate(arg, PRECISION);

        assertTrue(result.compareTo(ZERO) > 0);
    }
}