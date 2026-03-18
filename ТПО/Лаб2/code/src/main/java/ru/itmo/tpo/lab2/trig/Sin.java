package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Sin implements MathFunction {

    private static final MathContext mc = new MathContext(25, RoundingMode.HALF_UP);

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {

        BigDecimal result = BigDecimal.ZERO;
        BigDecimal term = x;
        int n = 1;

        while (term.abs().compareTo(eps) > 0) {
            result = result.add(term, mc);

            BigDecimal numerator = term.multiply(x, mc).multiply(x, mc).negate();
            BigDecimal denominator = BigDecimal.valueOf((2L * n) * (2L * n + 1));

            term = numerator.divide(denominator, mc);
            n++;
        }

        return result;
    }
}