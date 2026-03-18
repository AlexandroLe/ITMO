package ru.itmo.tpo.lab2.stub.log;

import ru.itmo.tpo.lab2.function.MathFunction;

import java.math.BigDecimal;

public class LogNBaseStub implements MathFunction {

    @Override
    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {
        return BigDecimal.ONE;
    }
}