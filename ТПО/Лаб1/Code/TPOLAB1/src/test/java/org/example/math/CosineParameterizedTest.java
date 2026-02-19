package org.example.math;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.Arguments;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class CosineParameterizedTest {

    @ParameterizedTest(name = "[{index}] x={0}, expect={1}, tol={2}")
    @MethodSource("valuesProvider")
    void parameterizedCosTests(double x, Double expected, double tol) {
        double actual = Cosine.cos(x);

        if (expected == null) {
            assertTrue(Double.isNaN(actual), () -> "Expected NaN for input: " + x + ", but was: " + actual);
        } else {
            assertEquals(expected, actual, tol, () -> String.format("x=%.18g, expected=%.18g, actual=%.18g", x, expected, actual));
        }
    }

    static Stream<Arguments> valuesProvider() {
        return Stream.of(
                Arguments.of(0.0, Math.cos(0.0), 1e-15),

                Arguments.of(1e-6, Math.cos(1e-6), 1e-12),
                Arguments.of(-1e-6, Math.cos(-1e-6), 1e-12),

                Arguments.of(1e-308, Math.cos(1e-308), 1e-15),

                Arguments.of(Math.PI / 2.0, Math.cos(Math.PI / 2.0), 1e-12),
                Arguments.of(-Math.PI / 2.0, Math.cos(-Math.PI / 2.0), 1e-12),
                Arguments.of(Math.PI, Math.cos(Math.PI), 1e-12),
                Arguments.of(-Math.PI, Math.cos(-Math.PI), 1e-12),
                Arguments.of(3.0 * Math.PI / 4.0, Math.cos(3.0 * Math.PI / 4.0), 1e-12),

                Arguments.of(1e6, Math.cos(1e6), 1e-9),
                Arguments.of(-1e6, Math.cos(-1e6), 1e-9),

                Arguments.of(2.0 * Math.PI, Math.cos(2.0 * Math.PI), 1e-15),

                Arguments.of(Double.NaN, null, 0.0),
                Arguments.of(Double.POSITIVE_INFINITY, null, 0.0),
                Arguments.of(Double.NEGATIVE_INFINITY, null, 0.0)
        );
    }
}
