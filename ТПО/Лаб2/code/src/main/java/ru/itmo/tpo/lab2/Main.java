package ru.itmo.tpo.lab2;

import ru.itmo.tpo.lab2.trig.*;
import ru.itmo.tpo.lab2.log.*;
import ru.itmo.tpo.lab2.stub.trig.*;
import ru.itmo.tpo.lab2.stub.log.*;

import ru.itmo.tpo.lab2.util.CsvWriter;

import java.math.BigDecimal;

public class Main {

    public static void main(String[] args) {

        BigDecimal eps = new BigDecimal("0.00001");

        // Классы с реализацией
        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Tan tan = new Tan(sin, cos);
        Cot cot = new Cot(tan);
        Sec sec = new Sec(cos);
        Csc csc = new Csc(sin);

        Ln ln = new Ln();
        LogNBase log3 = new LogNBase(ln, 3);
        LogNBase log5 = new LogNBase(ln, 5);

        // Сборка приложения на заглушках (нужна доработка заглушек, много деления на 0)
        // SinStub sin = new SinStub();
        // CosStub cos = new CosStub();
        // TanStub tan = new TanStub();
        // CotStub cot = new CotStub();
        // SecStub sec = new SecStub();
        // CscStub csc = new CscStub();
        // LnStub ln = new LnStub();
        // LogNBaseStub log3 = new LogNBaseStub();
        // LogNBaseStub log5 = new LogNBaseStub();


        EquationSystem system = new EquationSystem(
                sin, cos, tan, cot, sec, csc,
                ln, log3, log5
        );

        for (BigDecimal x = new BigDecimal("-2");
             x.compareTo(new BigDecimal("2")) <= 0;
             x = x.add(new BigDecimal("0.5"))) {

            System.out.println("x=" + x + " -> " +
                    system.calculate(x, eps));
        }

        CsvWriter.write(
            "results.csv",
            new BigDecimal("-2"),
            new BigDecimal("2"),
            new BigDecimal("0.1"),
            x -> system.calculate(x, eps)
        );
    }
}