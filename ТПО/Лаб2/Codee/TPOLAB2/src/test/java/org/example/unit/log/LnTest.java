package org.example.unit.log;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.log.Ln;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static java.math.RoundingMode.HALF_EVEN;
import static org.junit.jupiter.api.Assertions.*;

class LnTest {

    private static final BigDecimal PRECISION = new BigDecimal("0.0000001");
    private static final double DELTA = 0.000001;
    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_EVEN);
    private Ln ln;

    @BeforeEach
    void init() {
        ln = new Ln();
    }

    @Test
    void shouldNotCalculateForZero() {
        assertThrows(ArithmeticException.class, () -> ln.calculate(ZERO, PRECISION));
    }

    @Test
    void shouldNotCalculateForNegative() {
        assertThrows(ArithmeticException.class, () -> ln.calculate(new BigDecimal("-1"), PRECISION));
    }

    @Test
    void shouldNotCalculateForNegativeFive() {
        assertThrows(ArithmeticException.class, () -> ln.calculate(new BigDecimal("-5"), PRECISION));
    }

    @Test
    void shouldCalculateForOne() {
        BigDecimal expected = ZERO.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(ONE, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForE() {
        BigDecimal e = new BigDecimal("2.718281828459045");
        BigDecimal expected = ONE.setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(e, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForTen() {
        BigDecimal arg = BigDecimal.TEN;
        BigDecimal expected = new BigDecimal("2.3025851").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForOneOverE() {
        BigDecimal e = new BigDecimal("2.718281828459045");
        BigDecimal arg = ONE.divide(e, MC);
        BigDecimal expected = ONE.negate().setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForHalf() {
        BigDecimal arg = new BigDecimal("0.5");
        BigDecimal expected = new BigDecimal("-0.6931472").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForOneTenth() {
        BigDecimal arg = new BigDecimal("0.1");
        BigDecimal expected = new BigDecimal("-2.3025851").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForTwo() {
        BigDecimal arg = new BigDecimal("2");
        BigDecimal expected = new BigDecimal("0.6931472").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldCalculateForThree() {
        BigDecimal arg = new BigDecimal("3");
        BigDecimal expected = new BigDecimal("1.0986123").setScale(PRECISION.scale(), HALF_EVEN);
        BigDecimal actual = ln.calculate(arg, PRECISION);
        assertEquals(expected.doubleValue(), actual.doubleValue(), DELTA);
    }

    @Test
    void shouldBeIncreasing() {
        BigDecimal x1 = new BigDecimal("0.5");
        BigDecimal x2 = new BigDecimal("1");
        BigDecimal x3 = new BigDecimal("2");
        
        BigDecimal y1 = ln.calculate(x1, PRECISION);
        BigDecimal y2 = ln.calculate(x2, PRECISION);
        BigDecimal y3 = ln.calculate(x3, PRECISION);
        
        assertTrue(y1.compareTo(y2) < 0);
        assertTrue(y2.compareTo(y3) < 0);
    }

    @Test
    void shouldBeNegativeForXLessThanOne() {
        BigDecimal arg = new BigDecimal("0.5");
        BigDecimal result = ln.calculate(arg, PRECISION);
        assertTrue(result.compareTo(ZERO) < 0);
    }

    @Test
    void shouldBePositiveForXGreaterThanOne() {
        BigDecimal arg = new BigDecimal("2");
        BigDecimal result = ln.calculate(arg, PRECISION);
        assertTrue(result.compareTo(ZERO) > 0);
    }

    @Test
    void shouldHavePropertyLnAB() {
        BigDecimal a = new BigDecimal("2");
        BigDecimal b = new BigDecimal("3");
        
        BigDecimal lnA = ln.calculate(a, PRECISION);
        BigDecimal lnB = ln.calculate(b, PRECISION);
        BigDecimal lnAB = ln.calculate(a.multiply(b, MC), PRECISION);
        
        assertEquals(lnA.add(lnB, MC).doubleValue(), lnAB.doubleValue(), DELTA * 10);
    }

    @Test
    void shouldHavePropertyLnDiv() {
        BigDecimal a = new BigDecimal("6");
        BigDecimal b = new BigDecimal("2");
        
        BigDecimal lnA = ln.calculate(a, PRECISION);
        BigDecimal lnB = ln.calculate(b, PRECISION);
        BigDecimal lnDiv = ln.calculate(a.divide(b, MC), PRECISION);
        
        assertEquals(lnA.subtract(lnB, MC).doubleValue(), lnDiv.doubleValue(), DELTA * 10);
    }

    @Test
    void shouldHavePropertyLnPow() {
        BigDecimal a = new BigDecimal("2");
        int n = 3;
        BigDecimal aPowN = a.pow(n, MC);
        
        BigDecimal lnA = ln.calculate(a, PRECISION);
        BigDecimal lnAPowN = ln.calculate(aPowN, PRECISION);
        
        assertEquals(lnA.multiply(new BigDecimal(n), MC).doubleValue(), lnAPowN.doubleValue(), DELTA * 10);
    }

    @Test
    void shouldCalculateWithDifferentPrecisions() {
        BigDecimal arg = new BigDecimal("2");
        BigDecimal lowPrecision = new BigDecimal("0.01");
        BigDecimal highPrecision = new BigDecimal("0.00000001");
        
        BigDecimal resultLow = ln.calculate(arg, lowPrecision);
        BigDecimal resultHigh = ln.calculate(arg, highPrecision);
        
        assertNotEquals(resultLow, resultHigh);
        assertTrue(resultHigh.subtract(resultLow, MC).abs().compareTo(lowPrecision) < 0);
    }

    @Test
    void shouldFailForThousand() {
        assertThrows(ArithmeticException.class,
            () -> ln.calculate(new BigDecimal("1000"), PRECISION));
    }

    @Test
    void shouldFailWithCorrectMessageForLargeX() {
        ArithmeticException ex = assertThrows(ArithmeticException.class,
            () -> ln.calculate(new BigDecimal("1000"), PRECISION));
        assertTrue(ex.getMessage().contains("did not converge"));
    }

    @Test
    void shouldFailForVerySmallX() {
        assertThrows(ArithmeticException.class,
            () -> ln.calculate(new BigDecimal("0.001"), PRECISION));
    }

}