package org.example.unit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.example.EquationSystem;
import org.example.log.*;
import org.example.trig.*;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

class EquationSystemTest {

    private static final BigDecimal EPS = new BigDecimal("0.000001");

    private EquationSystem system;

    @BeforeEach
    void init() {
        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Tan tan = new Tan(sin, cos);
        Cot cot = new Cot(tan);
        Sec sec = new Sec(cos);
        Csc csc = new Csc(sin);

        Ln ln = new Ln();
        LogNBase log3 = new LogNBase(ln, 3);
        LogNBase log5 = new LogNBase(ln, 5);

        system = new EquationSystem(
                sin, cos, tan, cot, sec, csc,
                ln, log3, log5
        );
    }

    @Test
    void zeroValue() {
        assertThrows(ArithmeticException.class,
            () -> system.calculate(BigDecimal.ZERO, EPS));
    }

    @Test
    void negativeValues() {
        assertAll(
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("-0.5"), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("-1"), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("-2"), EPS))
        );
    }

    @Test
    void tanDiscontinuity() {
        BigDecimal x = BigDecimal.valueOf(-Math.PI / 2);
        assertThrows(ArithmeticException.class,
                () -> system.calculate(x, EPS));

        BigDecimal x2 = BigDecimal.valueOf(-3 * Math.PI / 2);
        assertThrows(ArithmeticException.class,
                () -> system.calculate(x2, EPS));
    }

    @Test
    void cotDiscontinuity() {
        BigDecimal x = BigDecimal.valueOf(-Math.PI);
        assertThrows(ArithmeticException.class,
                () -> system.calculate(x, EPS));
    }

    @Test
    void secDiscontinuity() {
        BigDecimal x = BigDecimal.valueOf(-Math.PI / 2);
        assertThrows(ArithmeticException.class,
                () -> system.calculate(x, EPS));
    }

    @Test
    void cscDiscontinuity() {
        BigDecimal x = BigDecimal.valueOf(-Math.PI);
        assertThrows(ArithmeticException.class,
                () -> system.calculate(x, EPS));
    }

    @Test
    void specialNegativeValues() {
        assertAll(
                () -> assertDoesNotThrow(() -> system.calculate(BigDecimal.valueOf(-Math.PI / 4), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(BigDecimal.valueOf(-Math.PI / 6), EPS))
        );
    }

    @Test
    void xEqualsOne() {
        assertThrows(ArithmeticException.class,
                () -> system.calculate(BigDecimal.ONE, EPS));
    }

    @Test
    void validPositiveValues() {
        assertAll(
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("5"), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("3"), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("15"), EPS))
        );
    }

    @Test
    void fractionalPositiveValues() {
        assertAll(
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("0.5"), EPS)),
                () -> assertDoesNotThrow(() -> system.calculate(new BigDecimal("0.2"), EPS))
        );
    }

    @Test
    void nearZeroPositiveShouldThrowException() {
        assertThrows(ArithmeticException.class,
                () -> system.calculate(new BigDecimal("0.0001"), EPS),
                "Для значений, близких к нулю, должно выбрасываться исключение");
    }

    @Test
    void largePositiveShouldThrowException() {
        assertThrows(ArithmeticException.class,
                () -> system.calculate(new BigDecimal("1000"), EPS),
                "Для больших положительных значений должно выбрасываться исключение");
    }
}