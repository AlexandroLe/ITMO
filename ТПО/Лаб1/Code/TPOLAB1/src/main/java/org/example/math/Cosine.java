package org.example.math;

/**
 * Косинус через ряд Тейлора с редукцией аргумента.
 * Поведение аналогично Math.cos для NaN/Infinity (возвращает Double.NaN).
 */
public final class Cosine {

    private static final double TWO_PI = 2.0 * Math.PI;

    private Cosine() {}

    public static double cos(double x) {
        return cos(x, 1e-15, 1000);
    }

    public static double cos(double x, double eps, int maxTerms) {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            return Double.NaN;
        }

        double y = x % TWO_PI;
        if (y > Math.PI) y -= TWO_PI;
        if (y <= -Math.PI) y += TWO_PI;

        boolean negate = false;
        if (y > Math.PI / 2.0) {
            y = Math.PI - y;
            negate = true;
        } else if (y < -Math.PI / 2.0) {
            y = -Math.PI - y;
            negate = true;
        }

        double term = 1.0;    // k = 0
        double sum = term;
        double x2 = y * y;

        for (int k = 1; k <= maxTerms; k++) {
            double denom = (2.0 * k - 1.0) * (2.0 * k);
            term *= -x2 / denom;
            sum += term;
            if (Math.abs(term) < eps) break;
        }

        return negate ? -sum : sum;
    }
}
