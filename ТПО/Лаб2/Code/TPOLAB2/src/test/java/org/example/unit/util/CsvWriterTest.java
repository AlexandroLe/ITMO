package org.example.unit.util;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.example.util.CsvWriter;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.util.List;
import java.util.function.Function;

import static java.math.BigDecimal.*;
import static org.junit.jupiter.api.Assertions.*;

class CsvWriterTest {

    private static final String BASE_DIR = "src/results";
    private static final String FILE_NAME = "test.csv";

    private final File file = new File(BASE_DIR, FILE_NAME);

    @Test
    void shouldCreateFile() {
        CsvWriter.write(
                FILE_NAME,
                ZERO,
                ONE,
                ONE,
                x -> ONE
        );

        assertTrue(file.exists());
    }

    @Test
    void shouldWriteHeader() throws IOException {
        CsvWriter.write(
                FILE_NAME,
                ZERO,
                ZERO,
                ONE,
                x -> ONE
        );

        List<String> lines = Files.readAllLines(file.toPath());

        assertEquals("x,result", lines.get(0));
    }

    @Test
    void shouldWriteCorrectValues() throws IOException {
        CsvWriter.write(
                FILE_NAME,
                ZERO,
                ONE,
                ONE,
                x -> x.add(ONE)
        );

        List<String> lines = Files.readAllLines(file.toPath());

        assertEquals("0,1", lines.get(1));
        assertEquals("1,2", lines.get(2));
    }

    @Test
    void shouldHandleExceptionAndWriteEmptyValue() throws IOException {
        Function<BigDecimal, BigDecimal> func = x -> {
            throw new ArithmeticException("error");
        };

        CsvWriter.write(
                FILE_NAME,
                ZERO,
                ONE,
                ONE,
                func
        );

        List<String> lines = Files.readAllLines(file.toPath());

        assertEquals("0,", lines.get(1));
        assertEquals("1,", lines.get(2));
    }

    @Test
    void shouldWorkWithAbstractMathFunction() throws IOException {
        BigDecimal TWO = BigDecimal.valueOf(2);
        
        CsvWriter.write(
                FILE_NAME,
                ZERO,
                ONE,
                ONE,
                x -> x.multiply(TWO)
        );

        List<String> lines = Files.readAllLines(file.toPath());

        assertEquals("0,0", lines.get(1));
        assertEquals("1,2", lines.get(2));
    }

    @Test
    void shouldOverwriteFile() throws IOException {
        CsvWriter.write(FILE_NAME, ZERO, ZERO, ONE, x -> ONE);
        CsvWriter.write(FILE_NAME, ZERO, ZERO, ONE, x -> TEN);

        List<String> lines = Files.readAllLines(file.toPath());

        assertEquals("0,10", lines.get(1));
    }

    @AfterEach
    void cleanup() {
        if (file.exists()) {
            file.delete();
        }
    }
}