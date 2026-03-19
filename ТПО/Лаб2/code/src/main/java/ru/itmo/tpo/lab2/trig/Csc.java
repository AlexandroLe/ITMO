package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.AbstractMathFunction;
import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Csc extends AbstractMathFunction {

    private final Sin sin;
    private static final MathContext mc = new MathContext(25);

    public Csc(Sin sin) {
        this.sin = sin;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal sinVal = sin.calculate(x, eps);

        if (sinVal.abs().compareTo(eps) < 0) {
            throw new ArithmeticException("csc undefined: sin(x)=0 at x=" + x);
        }

        return BigDecimal.ONE.divide(sinVal, mc);
    }
}