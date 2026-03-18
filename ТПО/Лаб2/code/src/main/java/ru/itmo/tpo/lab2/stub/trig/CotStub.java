package ru.itmo.tpo.lab2.stub.trig;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class CotStub implements MathFunction {

    private final Map<BigDecimal, BigDecimal> table = new HashMap<>();

    public CotStub() {
        table.put(BigDecimal.valueOf(Math.PI / 4), BigDecimal.ONE);
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return table.getOrDefault(x, BigDecimal.ONE);
    }
}