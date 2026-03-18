package ru.itmo.tpo.lab2.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.*;

public class Cos implements MathFunction {

    private final Sin sin;
    private static final MathContext mc = new MathContext(25);
    private static final BigDecimal HALF_PI = BigDecimal.valueOf(Math.PI / 2);

    public Cos(Sin sin) {
        this.sin = sin;
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return sin.calculate(x.add(HALF_PI, mc), eps);
    }
}