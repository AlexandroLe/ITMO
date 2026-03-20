package org.example.log;

import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Ln extends AbstractMathFunction {

    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    private int lastTermsCount = 0;

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return calculate(x, eps, DEFAULT_MAX_ITERATIONS);
    }

    public BigDecimal calculate(BigDecimal x, BigDecimal eps, int maxIterations) {
        if (x.compareTo(BigDecimal.ZERO) <= 0) {
            throw new ArithmeticException("ln undefined for x <= 0: x=" + x);
        }

        isValid(eps);

        if (x.compareTo(BigDecimal.ONE) == 0) {
            lastTermsCount = 0;
            return BigDecimal.ZERO.setScale(eps.scale(), RoundingMode.HALF_EVEN);
        }

        int scale = eps.scale() + 2;
        RoundingMode rm = RoundingMode.HALF_EVEN;

        BigDecimal z = x.subtract(BigDecimal.ONE).divide(x.add(BigDecimal.ONE), scale, rm);
        BigDecimal z2 = z.pow(2);
        BigDecimal term = z;
        BigDecimal result = BigDecimal.ZERO;
        int i = 1;
        int iteration = 0;

        do {
            if (iteration >= maxIterations) {
                throw new ArithmeticException("Ln calculation did not converge within " + maxIterations + " iterations");
            }

            result = result.add(term.divide(BigDecimal.valueOf(i), scale, rm));
            term = term.multiply(z2);
            i += 2;
            iteration++;
        } while (term.abs().compareTo(eps) > 0);

        lastTermsCount = iteration;
        return result.multiply(BigDecimal.valueOf(2)).setScale(eps.scale(), rm);
    }

    private void isValid(BigDecimal eps) {
        if (eps == null || eps.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Precision must be positive");
        }
    }

    public int getLastTermsCount() {
        return lastTermsCount;
    }
}