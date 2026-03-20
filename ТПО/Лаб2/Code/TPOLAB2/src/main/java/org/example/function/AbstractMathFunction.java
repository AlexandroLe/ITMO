package org.example.function;

import java.math.BigDecimal;

public abstract class AbstractMathFunction implements MathFunction {

    @Override
    public final BigDecimal doCalculate(BigDecimal x, BigDecimal eps) {

        if (x == null) {
            throw new NullPointerException("x is null");
        }

        if (eps == null) {
            throw new NullPointerException("eps is null");
        }

        if (eps.compareTo(BigDecimal.ZERO) <= 0 ||
                eps.compareTo(BigDecimal.ONE) > 0) {
            throw new ArithmeticException("eps must be in (0, 1]");
        }

        return calculate(x, eps);
    }

    public abstract BigDecimal calculate(BigDecimal x, BigDecimal eps);
}