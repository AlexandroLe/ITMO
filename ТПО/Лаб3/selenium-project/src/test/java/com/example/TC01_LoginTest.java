package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC01_LoginTest {
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
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @Test
    public void testEmptyLoginFields() {
        driver.get("https://pikabu.ru/");


        WebElement loginInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[@placeholder='Логин']")
        ));
        WebElement passwordInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath("//input[@placeholder='Пароль']")
        ));
        loginInput.click();
        loginInput.sendKeys("antero3112");

        passwordInput.click();
        passwordInput.sendKeys("fKyF.m!RP.86n3F");

        WebElement submitButton = driver.findElement(
                By.xpath("//form[@id='signin-form']//button[@type='submit']")
        );
        submitButton.click();

        WebElement userName = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//a[@title='antero3112']")
        ));

        System.out.println("Success login: " + userName.getText());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}