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

public class TC06_LikeWithAuthTest {
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
    public void testLikeWithAuth() {
        driver.get("https://pikabu.ru/");

        WebElement ratingBlock = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[contains(@aria-label,'минус')]")
        ));

        String beforeText = ratingBlock.getAttribute("aria-label");
        int beforeLikes = Integer.parseInt(beforeText.split(" ")[0]);

        System.out.println("Likes before: " + beforeLikes);

        WebElement likeButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[contains(@class,'story__rating-up')]")
        ));
        likeButton.click();

        wait.until(driver -> {
            String updated = ratingBlock.getAttribute("aria-label");
            int newLikes = Integer.parseInt(updated.split(" ")[0]);
            return newLikes == beforeLikes + 1;
        });

        String afterText = ratingBlock.getAttribute("aria-label");
        int afterLikes = Integer.parseInt(afterText.split(" ")[0]);

        System.out.println("Likes after: " + afterLikes);

        Assertions.assertEquals(beforeLikes + 1, afterLikes, "Like count did not increase by 1");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}