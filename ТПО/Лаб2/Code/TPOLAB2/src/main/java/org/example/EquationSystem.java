package org.example;

import org.example.function.AbstractMathFunction;

import java.math.*;

public class EquationSystem {

    private static final MathContext mc = new MathContext(25);

    private final AbstractMathFunction sin;
    private final AbstractMathFunction cos;
    private final AbstractMathFunction tan;
    private final AbstractMathFunction cot;
    private final AbstractMathFunction sec;
    private final AbstractMathFunction csc;

    private final AbstractMathFunction ln;
    private final AbstractMathFunction log3;
    private final AbstractMathFunction log5;

    public EquationSystem(AbstractMathFunction sin, AbstractMathFunction cos, AbstractMathFunction tan,
                          AbstractMathFunction cot, AbstractMathFunction sec, AbstractMathFunction csc,
                          AbstractMathFunction ln, AbstractMathFunction log3, AbstractMathFunction log5) {

        this.sin = sin;
        this.cos = cos;
        this.tan = tan;
        this.cot = cot;
        this.sec = sec;
        this.csc = csc;
        this.ln = ln;
        this.log3 = log3;
        this.log5 = log5;
    }

    public BigDecimal calculate(BigDecimal x, BigDecimal eps) {

        if (x.compareTo(BigDecimal.ZERO) <= 0) {

            BigDecimal sinVal = sin.calculate(x, eps);
            BigDecimal cosVal = cos.calculate(x, eps);
            BigDecimal tanVal = tan.calculate(x, eps);
            BigDecimal cotVal = cot.calculate(x, eps);
            BigDecimal secVal = sec.calculate(x, eps);
            BigDecimal cscVal = csc.calculate(x, eps);

            if (tanVal.abs().compareTo(eps) < 0) {
                throw new ArithmeticException("tan(x) ≈ 0 → division by zero, x=" + x);
            }

            BigDecimal first =
                    cscVal.divide(tanVal, mc)
                            .subtract(cosVal, mc)
                            .pow(3, mc);

            BigDecimal second =
                    secVal.subtract(sinVal, mc);

            BigDecimal combined =
                    first.subtract(second, mc)
                            .pow(2, mc)
                            .multiply(cotVal.pow(3, mc), mc)
                            .multiply(sinVal, mc);

            return combined.pow(3, mc).pow(3, mc);

        } else {

            BigDecimal l5 = log5.calculate(x, eps);
            BigDecimal l3 = log3.calculate(x, eps);
            BigDecimal lnVal = ln.calculate(x, eps);

            if (l3.abs().compareTo(eps) < 0) {
                throw new ArithmeticException("log3(x)=0 → division by zero, x=" + x);
            }

            if (l5.abs().compareTo(eps) < 0) {
                throw new ArithmeticException("log5(x)=0 → division by zero, x=" + x);
            }

            BigDecimal part1 =
                    l5.multiply(l5, mc)
                            .add(l3, mc);

            BigDecimal part2 =
                    l3.divide(l3, mc);

            BigDecimal left =
                    part1.multiply(part2, mc)
                            .add(l3, mc);

            BigDecimal right =
                    lnVal.subtract(
                            l5.divide(l5, mc), mc
                    );

            return left.multiply(right, mc);
        }
    }
}