package org.example.trig;

import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;

public class Sec extends AbstractMathFunction {

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