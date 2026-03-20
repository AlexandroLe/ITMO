package org.example.trig;



import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

public class Sin extends AbstractMathFunction {

    private static final MathContext mc = new MathContext(25, RoundingMode.HALF_UP);
    private static final BigDecimal PI = new BigDecimal(Math.PI, mc);
    private static final BigDecimal TWO_PI = PI.multiply(new BigDecimal("2"), mc);

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal reduced = x.remainder(TWO_PI, mc);
        
        if (reduced.compareTo(PI) > 0) {
            reduced = reduced.subtract(TWO_PI, mc);
        }
        else if (reduced.compareTo(PI.negate()) < 0) {
            reduced = reduced.add(TWO_PI, mc);
        }

        BigDecimal result = BigDecimal.ZERO;
        BigDecimal term = reduced;
        int n = 1;

        while (term.abs().compareTo(eps) > 0) {
            result = result.add(term, mc);

            BigDecimal numerator = term.multiply(reduced, mc).multiply(reduced, mc).negate();
            BigDecimal denominator = BigDecimal.valueOf((2L * n) * (2L * n + 1));

            term = numerator.divide(denominator, mc);
            n++;
        }

        return result;
    }
}