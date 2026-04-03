package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.*;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC08_AddCommentTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        String browser = System.getProperty("browser", "chrome");
        if (browser.equalsIgnoreCase("firefox")) {
            FirefoxProfile profile = new FirefoxProfile(
                    new java.io.File("C:/Users/AsusAspire 3/AppData/Roaming/Mozilla/Firefox/Profiles/bpa8o2jm.TestTPO3")
            );
            FirefoxOptions options = new FirefoxOptions();
            options.setProfile(profile);
            driver = new FirefoxDriver(options);
        } else {
            ChromeOptions options = new ChromeOptions();
            options.addArguments("user-data-dir=C:/Users/AsusAspire 3/AppData/Local/Google/Chrome/User Data/Profile 4");
            driver = new ChromeDriver(options);
        }
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

        WebElement commentInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[@editor='[object Object]']//p")
        ));
        commentInput.click();
        commentInput.sendKeys("Test: successful submission.");

        WebElement submitButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//span[@class='pkb-btn__text--FO_UA9yM']")
        ));
        submitButton.click();

        WebElement commentText = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//p[contains(text(),'Test')]")
        ));

        System.out.println("Comment found: " + commentText.getText());

        Assertions.assertTrue(commentText.isDisplayed(), "Comment was not added");

    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}