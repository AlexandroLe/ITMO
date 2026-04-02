package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC15_RegisterExistingLoginTest {
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
    public void testExistingEmailRegistration() {
        driver.get("https://pikabu.ru/");

        WebElement signupButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//button[@data-to='signup']")
        ));
        signupButton.click();

        WebElement loginInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[@placeholder='Никнейм на Пикабу *']")
        ));
        loginInput.sendKeys("alexandroesiano");

        WebElement submitButton = driver.findElement(
                By.xpath("//div[@class='app']")
        );
        submitButton.click();

        WebElement validationMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//span[contains(text(),'Логин занят')]")
        ));

        System.out.println("Validation message displayed: " + validationMessage.getText());

        Assertions.assertEquals("Логин занят", validationMessage.getText());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}