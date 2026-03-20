package org.example.trig;

import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;

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