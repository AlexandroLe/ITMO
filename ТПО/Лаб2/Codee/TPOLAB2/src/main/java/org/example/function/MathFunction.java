package org.example.function;

import java.math.BigDecimal;

public interface MathFunction {
    BigDecimal doCalculate(BigDecimal x, BigDecimal eps);
}