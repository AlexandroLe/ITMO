package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Sec implements MathFunction {

    private final Cos cos;
    private static final MathContext mc = new MathContext(25);

    public Sec(Cos cos) {
        this.cos = cos;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal cosVal = cos.calculate(x, eps);

        if (cosVal.abs().compareTo(eps) < 0) {
            throw new ArithmeticException("sec undefined: cos(x)=0 at x=" + x);
        }

        return BigDecimal.ONE.divide(cosVal, mc);
    }
}