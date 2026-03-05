package org.example.math;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

public final class Cosine {

    private static final MathContext MC = new MathContext(34, RoundingMode.HALF_UP); // ~double precision
    private static final BigDecimal TWO = new BigDecimal("2");
    private static final BigDecimal PI = new BigDecimal(Math.PI, MC);
    private static final BigDecimal TWO_PI = PI.multiply(TWO, MC);

    private static int lastTermsCount = 0;

    private Cosine() {}

    public static BigDecimal cos(BigDecimal x) {
        return cos(x, new BigDecimal("1e-20"), 1000);
    }

    public static BigDecimal cos(BigDecimal x, BigDecimal eps, int maxTerms) {

        BigDecimal y = x.remainder(TWO_PI, MC);

        if (y.compareTo(PI) > 0) {
            y = y.subtract(TWO_PI, MC);
        }
        if (y.compareTo(PI.negate()) <= 0) {
            y = y.add(TWO_PI, MC);
        }

        boolean negate = false;
        BigDecimal halfPi = PI.divide(TWO, MC);

        if (y.compareTo(halfPi) > 0) {
            y = PI.subtract(y, MC);
            negate = true;
        } else if (y.compareTo(halfPi.negate()) < 0) {
            y = PI.negate().subtract(y, MC);
            negate = true;
        }

        BigDecimal term = BigDecimal.ONE;
        BigDecimal sum = term;
        BigDecimal x2 = y.multiply(y, MC);

        lastTermsCount = 1;

        for (int k = 1; k <= maxTerms; k++) {
            BigDecimal denom = new BigDecimal((2L * k - 1L) * (2L * k));
            term = term.multiply(x2.negate(), MC)
                       .divide(denom, MC);
            sum = sum.add(term, MC);
            lastTermsCount++;

            if (term.abs().compareTo(eps) < 0) {
                break;
            }
        }

        return negate ? sum.negate(MC) : sum;
    }

    public static int getLastTermsCount() {
        return lastTermsCount;
    }
}