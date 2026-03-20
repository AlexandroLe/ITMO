package org.example.util;

import org.example.function.AbstractMathFunction;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.function.Function;

public class CsvWriter {

    private static final String BASE_DIR = "src/results";

    public static void write(String filename,
                             BigDecimal start,
                             BigDecimal end,
                             BigDecimal step,
                             Function<BigDecimal, BigDecimal> func) {

        File dir = new File(BASE_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        File file = new File(dir, filename);

        try (FileWriter writer = new FileWriter(file)) {

            writer.write("x,result\n");

            for (BigDecimal x = start;
                 x.compareTo(end) <= 0;
                 x = x.add(step)) {

                try {
                    BigDecimal y = func.apply(x);
                    writer.write(x + "," + y + "\n");
                } catch (Exception e) {
                    writer.write(x + ",\n");
                }
            }

        } catch (IOException e) {
            throw new RuntimeException("Ошибка при записи CSV файла", e);
        }
    }

    public static void write(String filename,
                             BigDecimal start,
                             BigDecimal end,
                             BigDecimal step,
                             AbstractMathFunction func,
                             BigDecimal eps) {

        write(filename, start, end, step, x -> func.calculate(x, eps));
    }
}