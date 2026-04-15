package com.example;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class TC17_RegisterEmptyTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testExistingEmailRegistration() {
        driver.get("https://pikabu.ru/");

        WebElement signupButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[@data-to='signup']")
        ));
        signupButton.click();

        WebElement emailInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'E-mail')]")
        ));

        WebElement loginInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Никнейм на Пикабу *')]")
        ));

        WebElement passwordInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Пароль *')]")
        ));

        emailInput.clear();
        loginInput.clear();
        passwordInput.clear();

        WebElement submitButton = driver.findElement(
                By.xpath("//div[@class='tabs__tab auth tabs__tab_visible']//button[@type='submit']")
        );
        submitButton.click();

        WebElement validationMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//span[contains(text(),'Обязательное поле')]")
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