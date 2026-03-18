package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Csc implements MathFunction {

    private final Sin sin;
    private static final MathContext mc = new MathContext(25);

    public Csc(Sin sin) {
        this.sin = sin;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal sinVal = sin.calculate(x, eps);

        if (sinVal.abs().compareTo(eps) < 0) {
            return new BigDecimal("1E100");
        }

        return BigDecimal.ONE.divide(sinVal, mc);
    }
}