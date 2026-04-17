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

public class TC06a_LikeAndRemoveLikeTest {
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
    public void testLikeAndRemoveLike() {
        driver.get("https://pikabu.ru/");

        WebElement ratingBlock = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[contains(@aria-label,'минус')]")
        ));

        String initialText = ratingBlock.getAttribute("aria-label");
        int initialLikes = Integer.parseInt(initialText.split(" ")[0]);

        System.out.println("Initial likes: " + initialLikes);

        WebElement likeButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[contains(@class,'story__rating-up')]")
        ));

        likeButton.click();

        wait.until(driver -> {
            int current = Integer.parseInt(ratingBlock.getAttribute("aria-label").split(" ")[0]);
            return current == initialLikes + 1;
        });

        System.out.println("Like added");

        likeButton.click();

        wait.until(driver -> {
            int current = Integer.parseInt(ratingBlock.getAttribute("aria-label").split(" ")[0]);
            return current == initialLikes;
        });

        int finalLikes = Integer.parseInt(ratingBlock.getAttribute("aria-label").split(" ")[0]);

        System.out.println("Final likes: " + finalLikes);

        Assertions.assertEquals(initialLikes, finalLikes, "Likes count did not return to initial value");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}
