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

public class TC07_LikeWithoutAuthTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testLikeWithoutAuth() {
        driver.get("https://pikabu.ru/");

        WebElement ratingBlock = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[contains(@aria-label,'плюсов')]")
        ));

        String likesText = ratingBlock.getAttribute("aria-label");
        System.out.println("Likes block: " + likesText);

        WebElement addIconButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[contains(@class,'story__rating-up')]")
        ));
        addIconButton.click();

        WebElement authNotice = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[@class='auth__notice']")
        ));

        System.out.println("Auth notice displayed: " + authNotice.getText());

        Assertions.assertTrue(authNotice.isDisplayed());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}