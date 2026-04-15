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

public class TC13_CreateEmptyTitlePostTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("user-data-dir=C:/Users/AsusAspire 3/AppData/Local/Google/Chrome/User Data/Profile 4");
        driver = new ChromeDriver(options);
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(15));
    }

    @Test
    public void testCreatePostWithoutTitle() {
        driver.get("https://pikabu.ru/");

        WebElement addPostButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[@class='header-right-menu__item header-right-menu__add button_add']")
        ));
        addPostButton.click();

        WebElement bodyInput = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//div[contains(@class,'story-editor-view__content')]")
        ));
        bodyInput.click();
        driver.switchTo().activeElement().sendKeys("TESTTESTTESTTEST.");

        WebElement submitButton = driver.findElement(
                By.xpath("//button[@class='pkb-btn__host--n2MGeea1 pkb-btn__host_wide--pv56KZ78']\n")
        );
        submitButton.click();

        WebElement validation = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[@class='toast toast_danger']")
        ));

        System.out.println("Validation message: " + validation.getText());

        Assertions.assertTrue(validation.getText().contains("Укажите заголовок"));
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}