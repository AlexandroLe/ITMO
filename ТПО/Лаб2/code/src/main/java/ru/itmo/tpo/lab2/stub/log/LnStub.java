package ru.itmo.tpo.lab2.stub.log;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class LnStub implements MathFunction {

    private final Map<BigDecimal, BigDecimal> table = new HashMap<>();

    public LnStub() {
        table.put(BigDecimal.ONE, BigDecimal.ZERO);
        table.put(BigDecimal.valueOf(2), BigDecimal.valueOf(Math.log(2)));
        table.put(BigDecimal.valueOf(3), BigDecimal.valueOf(Math.log(3)));
        table.put(BigDecimal.valueOf(5), BigDecimal.valueOf(Math.log(5)));
    }

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        if (x.compareTo(BigDecimal.ZERO) <= 0) {
            return BigDecimal.valueOf(Double.NaN);
        }
        return table.getOrDefault(x, BigDecimal.ZERO);
    }
}