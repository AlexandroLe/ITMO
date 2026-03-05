package org.example.math;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.Arguments;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class CosineParameterizedTest {

    private static final MathContext MC = new MathContext(50);

    @ParameterizedTest(name = "[{index}] x={0}")
    @MethodSource("valuesProvider")
    void parameterizedCosTests(BigDecimal x) {

        BigDecimal actual = Cosine.cos(x);

        double expectedDouble = Math.cos(x.doubleValue());
        BigDecimal expected = BigDecimal.valueOf(expectedDouble);

        BigDecimal tolerance = x.abs().compareTo(new BigDecimal("1000")) > 0
                ? new BigDecimal("1e-9")  // для больших аргументов
                : new BigDecimal("1e-12"); // для обычных

        BigDecimal diff = actual.subtract(expected, MC).abs();

        assertTrue(diff.compareTo(tolerance) <= 0,
                () -> "x=" + x + " expected=" + expected + " actual=" + actual +
                        " diff=" + diff + " tolerance=" + tolerance);

        assertTrue(Cosine.getLastTermsCount() > 0, "Used terms count should be > 0");
        assertTrue(Cosine.getLastTermsCount() <= 1000, "Used terms count should not exceed maxTerms");
    }

    static Stream<Arguments> valuesProvider() {

        BigDecimal PI = new BigDecimal(Math.PI, MC);

        return Stream.of(

                // 0
                Arguments.of(BigDecimal.ZERO),

                // ±π
                Arguments.of(PI),
                Arguments.of(PI.negate()),

                // ±π/2
                Arguments.of(PI.divide(new BigDecimal("2"), MC)),
                Arguments.of(PI.divide(new BigDecimal("2"), MC).negate()),

                // ±π/4
                Arguments.of(PI.divide(new BigDecimal("4"), MC)),
                Arguments.of(PI.divide(new BigDecimal("4"), MC).negate()),

                // ±π/3
                Arguments.of(PI.divide(new BigDecimal("3"), MC)),
                Arguments.of(PI.divide(new BigDecimal("3"), MC).negate()),

                // π/6
                Arguments.of(PI.divide(new BigDecimal("6"), MC)),

                // 3π/4
                Arguments.of(PI.multiply(new BigDecimal("3"), MC)
                               .divide(new BigDecimal("4"), MC)),

                // 53π/180
                Arguments.of(PI.multiply(new BigDecimal("53"), MC)
                               .divide(new BigDecimal("180"), MC)),

                // 190π/180
                Arguments.of(PI.multiply(new BigDecimal("190"), MC)
                               .divide(new BigDecimal("180"), MC)),

                // -261π/180
                Arguments.of(PI.multiply(new BigDecimal("-261"), MC)
                               .divide(new BigDecimal("180"), MC)),

                // ±10⁻¹⁰
                Arguments.of(new BigDecimal("1e-10")),
                Arguments.of(new BigDecimal("-1e-10")),

                // ±10⁶
                Arguments.of(new BigDecimal("1e6")),
                Arguments.of(new BigDecimal("-1e6"))
        );
    }
}