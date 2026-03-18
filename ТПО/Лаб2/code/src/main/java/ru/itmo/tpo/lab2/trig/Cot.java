package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Cot implements MathFunction {

    private final Tan tan;
    private static final MathContext mc = new MathContext(25);

    public Cot(Tan tan) {
        this.tan = tan;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal tanVal = tan.calculate(x, eps);

        if (tanVal.abs().compareTo(eps) < 0) {
            throw new ArithmeticException("cot undefined: tan(x)=0 at x=" + x);
        }

        return BigDecimal.ONE.divide(tanVal, mc);
    }
}