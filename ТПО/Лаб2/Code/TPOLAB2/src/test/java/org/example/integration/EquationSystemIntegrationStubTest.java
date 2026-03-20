package org.example.integration;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.example.EquationSystem;
import org.example.function.AbstractMathFunction;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

public class EquationSystemIntegrationStubTest {

    private static final MathContext MC = new MathContext(25, RoundingMode.HALF_UP);
    private static final BigDecimal EPS = new BigDecimal("0.000001");




    @Test
    public void testLeftBranchWithStubs() {

        // given
        BigDecimal x = new BigDecimal("-0.7853981633974483"); // -π/4

        // Создаём заглушки для всех девяти модулей
        AbstractMathFunction sin = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction ln = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        // Точные значения тригонометрических функций для x = -π/4
        BigDecimal sinVal = new BigDecimal("-0.7071067811865475");
        BigDecimal cosVal = new BigDecimal("0.7071067811865476");
        BigDecimal tanVal = new BigDecimal("-1.0");
        BigDecimal cotVal = new BigDecimal("-1.0");
        BigDecimal secVal = new BigDecimal("1.414213562373095");
        BigDecimal cscVal = new BigDecimal("-1.414213562373095");

        // Настраиваем поведение заглушек
        when(sin.calculate(eq(x), eq(EPS))).thenReturn(sinVal);
        when(cos.calculate(eq(x), eq(EPS))).thenReturn(cosVal);
        when(tan.calculate(eq(x), eq(EPS))).thenReturn(tanVal);
        when(cot.calculate(eq(x), eq(EPS))).thenReturn(cotVal);
        when(sec.calculate(eq(x), eq(EPS))).thenReturn(secVal);
        when(csc.calculate(eq(x), eq(EPS))).thenReturn(cscVal);

        // Собираем систему из заглушек
        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

        // when
        BigDecimal result = system.calculate(x, EPS);


        // then — вычисляем ожидаемое значение вручную по той же формуле
        BigDecimal step1 = cscVal.divide(tanVal, MC);
        BigDecimal step2 = step1.subtract(cosVal, MC);
        BigDecimal step3 = step2.pow(3, MC);
        BigDecimal step4 = secVal.subtract(sinVal, MC);
        BigDecimal step5 = step3.subtract(step4, MC);
        BigDecimal step6 = step5.pow(2, MC);
        BigDecimal step7 = step6.multiply(cotVal.pow(3, MC), MC);
        BigDecimal step8 = step7.multiply(sinVal, MC);
        BigDecimal step9 = step8.pow(3, MC);
        BigDecimal expected = step9.pow(3, MC);

        assertEquals(expected, result);
    }

    @Test
    public void testRightBranchWithStubs() {
        // given
        BigDecimal x = new BigDecimal("2");

        AbstractMathFunction sin = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cos = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction tan = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction cot = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction sec = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction csc = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction ln = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log3 = Mockito.mock(AbstractMathFunction.class);
        AbstractMathFunction log5 = Mockito.mock(AbstractMathFunction.class);

        // Точные значения логарифмических функций для x = 2
        BigDecimal lnVal = new BigDecimal("0.6931471805599453");
        BigDecimal log3Val = new BigDecimal("0.6309297535714574");
        BigDecimal log5Val = new BigDecimal("0.43067655807339306");

        when(ln.calculate(eq(x), eq(EPS))).thenReturn(lnVal);
        when(log3.calculate(eq(x), eq(EPS))).thenReturn(log3Val);
        when(log5.calculate(eq(x), eq(EPS))).thenReturn(log5Val);

        EquationSystem system = new EquationSystem(sin, cos, tan, cot, sec, csc, ln, log3, log5);

        // when
        BigDecimal result = system.calculate(x, EPS);

        // then — вычисляем ожидаемое значение по формуле для правой ветки
        BigDecimal part1 = log5Val.multiply(log5Val, MC).add(log3Val, MC);
        BigDecimal part2 = log3Val.divide(log3Val, MC); // всегда 1
        BigDecimal left = part1.multiply(part2, MC).add(log3Val, MC);
        BigDecimal right = lnVal.subtract(log5Val.divide(log5Val, MC), MC);
        BigDecimal expected = left.multiply(right, MC);

        assertEquals(expected, result);
    }
}