package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC07_LikeWithoutAuthTest {
    private WebDriver driver;
    private WebDriverWait wait;

    @BeforeEach
    public void setUp() {
        String browser = System.getProperty("browser", "chrome");
        if (browser.equalsIgnoreCase("firefox")) {
            driver = new FirefoxDriver();
        } else {
            driver = new ChromeDriver();
        }
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testLikeWithoutAuth() {
        driver.get("https://pikabu.ru/");

        WebElement addIconButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//article//div//div//div//div//button")
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