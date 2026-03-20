package org.example.trig;

import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;

public class Cot extends AbstractMathFunction {

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