package org.example.log;

import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;

public class LogNBase extends AbstractMathFunction {

    private final Ln ln;
    private final BigDecimal base;
    private static final MathContext mc = new MathContext(25);

    public LogNBase(Ln ln, double base) {
        if (base <= 0 || base == 1) {
            throw new IllegalArgumentException("Invalid base");
        }
        this.ln = ln;
        this.base = BigDecimal.valueOf(base);
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {

        BigDecimal lnX = ln.calculate(x, eps);
        BigDecimal lnBase = ln.calculate(base, eps);

        if (lnBase.abs().compareTo(eps) < 0) {
            throw new ArithmeticException("log base leads to division by zero");
        }

        return lnX.divide(lnBase, mc);
    }
}