package com.example;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class TC03_EmptyLoginFieldsTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testEmptyLoginFields() {
        driver.get("https://pikabu.ru/");


        WebElement loginInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Логин')]")
        ));
        WebElement passwordInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Пароль')]")
        ));

        loginInput.clear();
        passwordInput.clear();

        WebElement submitButton = driver.findElement(
                By.xpath("//form[@id='signin-form']//button[@type='submit']")
        );
        submitButton.click();

        WebElement validationMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//span[@class='input__validation-message']")
        ));

        System.out.println("Validation message displayed: " + validationMessage.getText());

        Assertions.assertEquals("Обязательное поле", validationMessage.getText());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}