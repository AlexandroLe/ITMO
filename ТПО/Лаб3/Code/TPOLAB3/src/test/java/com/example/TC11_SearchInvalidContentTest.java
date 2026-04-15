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

public class TC11_SearchInvalidContentTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(15));
    }

    @Test
    public void testSearchExistingPost() {
        driver.get("https://pikabu.ru/");

        WebElement searchButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//div[@class='header-right-menu__item header-right-menu__search']")
        ));
        searchButton.click();

        WebElement searchInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[contains(@placeholder, 'Искать на Пикабу')]")
        ));
        searchInput.sendKeys("qwiuyrqwknfkdsbvlwkfp;oskjclhzlxkcj;welkjckjzdbdhkjfhaklsdf");

        WebElement findButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[@class='Btn__btn--AXJA8qM4 Btn__btn_primary--Zfmrd9NN Btn__btn_m--tmrj7cPZ']//span[@class='Btn__content__label--Iyob42xe']")
        ));
        findButton.click();

        WebElement resultMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//p[contains(text(),'Ничего не найдено по запросу')]")
        ));
        System.out.println("Found post: " + resultMessage.getText());

        Assertions.assertEquals("Ничего не найдено по запросу", resultMessage.getText(),
                "First search result title does not match expected");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}
