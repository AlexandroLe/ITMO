package com.example;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class TC02_LoginInvalidPasswordTest {

    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("user-data-dir=C:/Users/AsusAspire 3/AppData/Local/Google/Chrome/User Data/Profile 4");
        driver = new ChromeDriver(options);
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testLoginWithInvalidPassword() {
        driver.get("https://pikabu.ru/");

        WebElement loginInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Логин')]")
        ));
        WebElement passwordInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Пароль')]")
        ));
        loginInput.click();
        loginInput.sendKeys("antero3112");

        passwordInput.click();
        passwordInput.sendKeys("wrong_password");

        WebElement submitButton = driver.findElement(
                By.xpath("//form[@id='signin-form']//button[@type='submit']")
        );
        submitButton.click();

        WebElement errorMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//form[@id='signin-form']//*[contains(text(),'невер')]")
        ));

        String errorText = errorMessage.getText().trim();
        System.out.println("Ошибка авторизации: " + errorText);

        Assertions.assertTrue(errorText.contains("невер"),
                "Ожидалось сообщение об ошибке");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}