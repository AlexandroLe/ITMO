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
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class TC08_AddCommentTest {
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
    public void testCommentWithoutAuth() {
        driver.get("https://pikabu.ru/");

        WebElement postHeader = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//header[contains(@class,'story__header')]")
        ));
        postHeader.click();

        for (String handle : driver.getWindowHandles()) {
            driver.switchTo().window(handle);
        }

        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//div[contains(@class,'story')]")
        ));

        ((JavascriptExecutor) driver).executeScript("window.scrollTo(0, document.body.scrollHeight);");

        WebElement commentInput = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//div[contains(@class,'comment')]//div[@contenteditable='true']")
        ));
        commentInput.click();
        driver.switchTo().activeElement().sendKeys("Ну, не знаю, не знаю.");

        WebElement submitButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[.//span[contains(normalize-space(), 'Отправить')]]")
        ));
        submitButton.click();

        WebElement commentText = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//p[contains(text(),'Ну, не знаю, не знаю')]")
        ));

        System.out.println("Comment found: " + commentText.getText());

        Assertions.assertTrue(commentText.isDisplayed(), "Comment was not added");

        WebElement commentOur = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//div[contains(@class,'comment')]//div[contains(@data-author-id, '11692895')]")
        ));

        Actions actions = new Actions(driver);
        actions.moveToElement(commentOur).perform();
        System.out.println("Создал");

        WebElement removeButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[@data-test='comment-remove']")
        ));
        removeButton.click();
        System.out.println("Удалил коммент");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}