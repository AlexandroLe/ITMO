package ru.itmo.tpo.lab2.log;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Ln implements MathFunction {

    private static final MathContext mc = new MathContext(25);

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {

        if (x.compareTo(BigDecimal.ZERO) <= 0) {
            throw new ArithmeticException("ln undefined for x <= 0: x=" + x);
        }

        BigDecimal one = BigDecimal.ONE;
        BigDecimal t = x.subtract(one).divide(x.add(one), mc);

        BigDecimal result = BigDecimal.ZERO;
        BigDecimal term = t;
        int n = 1;

        while (term.abs().compareTo(eps) > 0) {
            result = result.add(term.divide(BigDecimal.valueOf(n), mc), mc);
            term = term.multiply(t, mc).multiply(t, mc);
            n += 2;
        }

        return result.multiply(BigDecimal.valueOf(2), mc);
    }
}