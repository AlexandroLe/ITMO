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
import java.util.List;

public class TC04_FeedPostsTest {
    private static WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(15));
    }

    @Test
    public void testFeedPostsVisible() {
        driver.get("https://pikabu.ru/");
        boolean postFound = false;

        for (int i = 0; i < 10; i++) {
            List<WebElement> posts = wait.until(
                    ExpectedConditions.visibilityOfAllElementsLocatedBy(
                            By.xpath("//article[contains(@class,'story')]")
                    )
            );

            for (WebElement post : posts) {
                boolean hasTitle = post.findElements(By.xpath("//header[contains(@class,'story__header')]")).size() > 0;
                boolean hasAuthor = post.findElements(By.xpath("//a[contains(@class,'user__nick')]")).size() > 0;
                boolean hasDate = post.findElements(By.xpath("//time[contains(@class,'story__datetime')]")).size() > 0;
                boolean hasOptionsAndComments = post.findElements(By.xpath("//article[1]//div[2]//div[1]//div[4]")).size() > 0;

                if (hasTitle && hasAuthor && hasDate && hasOptionsAndComments) {
                    postFound = true;
                    System.out.println("Post verified: Title, Author, Date, Likes/Comments exist");
                    break;
                }
            }

            if (postFound) break;

            ((JavascriptExecutor) driver).executeScript("window.scrollBy(0, window.innerHeight);");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignored) {
            }
        }

        Assertions.assertTrue(postFound, "No posts with required content found");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}