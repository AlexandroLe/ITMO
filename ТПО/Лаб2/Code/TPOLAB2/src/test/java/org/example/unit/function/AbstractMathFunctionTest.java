package org.example.unit.function;

import org.junit.jupiter.api.Test;
import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

class AbstractMathFunctionTest {

    private static final BigDecimal VALID_EPS = new BigDecimal("0.5");

    private final AbstractMathFunction function = new AbstractMathFunction() {
        @Override
        public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
            return BigDecimal.ONE;
        }
    };

    @Test
    void nullX() {
        assertThrows(NullPointerException.class,
                () -> function.doCalculate(null, VALID_EPS));
    }

    @Test
    void nullEps() {
        assertThrows(NullPointerException.class,
                () -> function.doCalculate(BigDecimal.ONE, null));
    }

    @Test
    void epsLessOrEqualZero() {
        assertThrows(ArithmeticException.class,
                () -> function.doCalculate(BigDecimal.ONE, BigDecimal.ZERO));

        assertThrows(ArithmeticException.class,
                () -> function.doCalculate(BigDecimal.ONE, new BigDecimal("-1")));
    }

    @Test
    void epsGreaterThanOne() {
        assertThrows(ArithmeticException.class,
                () -> function.doCalculate(BigDecimal.ONE, new BigDecimal("2")));
    }

    @Test
    void validInput() {
        assertDoesNotThrow(() ->
                function.doCalculate(BigDecimal.ONE, VALID_EPS));
    }
}