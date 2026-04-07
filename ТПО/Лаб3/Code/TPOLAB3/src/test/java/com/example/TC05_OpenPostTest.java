package com.example;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.Set;

public class TC05_OpenPostTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(20));
    }

    @Test
    public void testOpenPost() {
        driver.get("https://pikabu.ru/");

        WebElement postLink = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//article[contains(@class,'story')][1]//a[contains(@href,'/story/')]")
        ));

        ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", postLink);

        String titleInFeed = postLink.getText();

        String originalWindow = driver.getWindowHandle();
        Set<String> oldWindows = driver.getWindowHandles();

        postLink.click();

        wait.until(driver -> driver.getWindowHandles().size() > oldWindows.size());

        for (String window : driver.getWindowHandles()) {
            if (!window.equals(originalWindow)) {
                driver.switchTo().window(window);
                break;
            }
        }

        WebElement titleOnPage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//h1")
        ));

        String titlePageText = titleOnPage.getText();

        boolean hasText = !driver.findElements(By.xpath("//div[contains(@class,'story__content')]//p")).isEmpty();
        boolean hasImage = !driver.findElements(By.xpath("//div[contains(@class,'story__content')]//img")).isEmpty();

        System.out.println("Title in feed: " + titleInFeed);
        System.out.println("Title on page: " + titlePageText);
        System.out.println("Has text: " + hasText);
        System.out.println("Has image: " + hasImage);

        Assertions.assertFalse(titlePageText.isEmpty(), "Title is empty");
        Assertions.assertTrue(hasText || hasImage, "Post has no content");
        Assertions.assertTrue(driver.getCurrentUrl().contains("/story/"));
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}