package org.example;



import org.example.log.*;
import org.example.trig.*;
import org.example.util.CsvWriter;

import java.math.BigDecimal;

public class Main {

    public static void main(String[] args) {

        BigDecimal eps = new BigDecimal("0.00001");

        Sin sin = new Sin();
        Cos cos = new Cos(sin);
        Tan tan = new Tan(sin, cos);
        Cot cot = new Cot(tan);
        Sec sec = new Sec(cos);
        Csc csc = new Csc(sin);

        Ln ln = new Ln();
        LogNBase log3 = new LogNBase(ln, 3);
        LogNBase log5 = new LogNBase(ln, 5);


        EquationSystem system = new EquationSystem(
                sin, cos, tan, cot, sec, csc,
                ln, log3, log5
        );

        String filename = "results.csv";

        CsvWriter.write(
            filename,
            new BigDecimal("-35"),
            new BigDecimal("35"),
            new BigDecimal("0.001"),
            x -> system.calculate(x, eps)
        );
    }
}