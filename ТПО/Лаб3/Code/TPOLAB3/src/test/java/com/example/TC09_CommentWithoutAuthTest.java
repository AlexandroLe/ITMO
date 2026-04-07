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
import java.util.Set;

public class TC09_CommentWithoutAuthTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(15));
    }

    @Test
    public void testCommentWithoutAuth() {
        driver.get("https://pikabu.ru/");

        WebElement firstPostHeader = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//header[@class='story__header']")
        ));
        firstPostHeader.click();

        String originalWindow = driver.getWindowHandle();
        Set<String> allWindows = driver.getWindowHandles();
        for (String window : allWindows) {
            if (!window.equals(originalWindow)) {
                driver.switchTo().window(window);
                break;
            }
        }

        boolean blockFound = false;
        for (int i = 0; i < 10; i++) {
            List<WebElement> authBlocks = driver.findElements(
                    By.xpath("//section[@class='section_gray']")
            );

            if (!authBlocks.isEmpty()) {
                WebElement authBlock = authBlocks.get(0);
                if (authBlock.isDisplayed() && authBlock.getText().contains(
                        "Чтобы оставить комментарий, необходимо зарегистрироваться")) {
                    System.out.println("Auth comment block displayed: " + authBlock.getText());
                    blockFound = true;
                    break;
                }
            }

            ((JavascriptExecutor) driver).executeScript("window.scrollBy(0, window.innerHeight);");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignored) {
            }
        }

        Assertions.assertTrue(blockFound, "Auth comment block not found on post page");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}