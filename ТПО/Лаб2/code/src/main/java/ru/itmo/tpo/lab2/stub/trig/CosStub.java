package ru.itmo.tpo.lab2.stub.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class CosStub implements MathFunction {

    private final Map<BigDecimal, BigDecimal> table = new HashMap<>();

    public CosStub() {
        table.put(BigDecimal.ZERO, BigDecimal.ONE);
        table.put(BigDecimal.valueOf(Math.PI / 2), BigDecimal.ZERO);
        table.put(BigDecimal.valueOf(Math.PI), BigDecimal.ONE.negate());
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return table.getOrDefault(x, BigDecimal.ZERO);
    }
}