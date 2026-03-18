package ru.itmo.tpo.lab2.function;

import java.math.BigDecimal;

public interface MathFunction {
    BigDecimal calculate(BigDecimal x, BigDecimal eps);
}