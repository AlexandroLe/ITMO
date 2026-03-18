package ru.itmo.tpo.lab2.util;

import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.function.Function;

public class CsvWriter {

    public static void write(String filename,
                             BigDecimal start,
                             BigDecimal end,
                             BigDecimal step,
                             Function<BigDecimal, BigDecimal> func) {

        try (FileWriter writer = new FileWriter(filename)) {

            writer.write("x,result\n");

            for (BigDecimal x = start;
                 x.compareTo(end) <= 0;
                 x = x.add(step)) {

                writer.write(x + "," + func.apply(x) + "\n");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}