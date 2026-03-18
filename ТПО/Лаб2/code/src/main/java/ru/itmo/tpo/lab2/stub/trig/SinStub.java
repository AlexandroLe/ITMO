package ru.itmo.tpo.lab2.stub.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class SinStub implements MathFunction {

    private final Map<BigDecimal, BigDecimal> table = new HashMap<>();

    public SinStub() {
        table.put(BigDecimal.ZERO, BigDecimal.ZERO);
        table.put(BigDecimal.valueOf(Math.PI / 2), BigDecimal.ONE);
        table.put(BigDecimal.valueOf(-Math.PI / 2), BigDecimal.ONE.negate());
        table.put(BigDecimal.valueOf(Math.PI), BigDecimal.ZERO);
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return table.getOrDefault(x, BigDecimal.ZERO);
    }
}