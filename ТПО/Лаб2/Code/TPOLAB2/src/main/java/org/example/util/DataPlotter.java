package org.example.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class DataPlotter extends JFrame {
    
    private static final String CSV_FILE_PATH = "src/results/results.csv";

    public DataPlotter(String title) {
        super(title);
        
        XYSeries series = parseCSV(CSV_FILE_PATH);
        
        if (series == null || series.isEmpty()) {
            JOptionPane.showMessageDialog(this, 
                "Не удалось загрузить данные из файла: " + CSV_FILE_PATH, 
                "Ошибка", 
                JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection(series);
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            "График функции f(x)", 
            "x",                 
            "f(x)",              
            dataset,              
            PlotOrientation.VERTICAL,
            true,                  
            true,                
            false                
        );
        
        XYPlot plot = chart.getXYPlot();
        
        NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
        xAxis.setRange(-35.1, 35.1);
        xAxis.setAutoRangeIncludesZero(false);
        
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setAutoRangeIncludesZero(false);
        yAxis.setRange(-20.0, 20.0); // Устанавливаем диапазон от -10 до 10
        
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, true);  
        renderer.setSeriesShapesVisible(0, true); 
        renderer.setSeriesShape(0, new java.awt.geom.Ellipse2D.Double(-3, -3, 6, 6));
        renderer.setSeriesPaint(0, Color.BLUE);
        plot.setRenderer(renderer);
        
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));
        chartPanel.setMouseWheelEnabled(true); 
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        setContentPane(chartPanel);
    }
    
    private XYSeries parseCSV(String filePath) {
        XYSeries series = new XYSeries("f(x)");
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isFirstLine = true;
            int lineNumber = 0;
            int skippedLines = 0;
            
            while ((line = br.readLine()) != null) {
                lineNumber++;
                
                if (isFirstLine) {
                    isFirstLine = false;
                    System.out.println("Заголовок CSV: " + line);
                    continue;
                }
                
                if (line.trim().isEmpty()) {
                    continue;
                }
                
                String[] parts = line.split(",");
                if (parts.length >= 2) {
                    try {
                        double x = Double.parseDouble(parts[0].trim());
                        
                        if (parts[1] != null && !parts[1].trim().isEmpty()) {
                            double y = Double.parseDouble(parts[1].trim());
                            series.add(x, y);
                            System.out.println("Добавлена точка: x=" + x + ", y=" + y);
                        } else {
                            System.out.println("Пропущена точка x=" + x + " (нет значения)");
                            skippedLines++;
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Ошибка парсинга строки " + lineNumber + ": " + line);
                        System.err.println("Причина: " + e.getMessage());
                        skippedLines++;
                    }
                } else {
                    System.err.println("Некорректный формат строки " + lineNumber + ": " + line);
                    skippedLines++;
                }
            }
            
            System.out.println("Всего обработано строк: " + lineNumber);
            System.out.println("Пропущено строк: " + skippedLines);
            System.out.println("Загружено точек: " + series.getItemCount());
            
        } catch (IOException e) {
            System.err.println("Ошибка чтения файла: " + filePath);
            System.err.println("Причина: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
        
        return series;
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                e.printStackTrace();
            }
            
            DataPlotter example = new DataPlotter("График функции из CSV");
            example.setSize(900, 700);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setLocationRelativeTo(null); 
            example.setVisible(true);
        });
    }
}