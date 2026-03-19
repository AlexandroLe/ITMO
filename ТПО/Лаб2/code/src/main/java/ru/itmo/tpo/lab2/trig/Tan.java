package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.AbstractMathFunction;
import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Tan extends AbstractMathFunction {

    private final Sin sin;
    private final Cos cos;
    private static final MathContext mc = new MathContext(25);

    public Tan(Sin sin, Cos cos) {
        this.sin = sin;
        this.cos = cos;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        BigDecimal cosVal = cos.calculate(x, eps);

        if (cosVal.abs().compareTo(eps) < 0) {
            throw new ArithmeticException("tan undefined: cos(x)=0 at x=" + x);
        }

        return sin.calculate(x, eps)
                .divide(cosVal, mc);
    }
}