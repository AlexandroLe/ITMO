package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC18_RegisterEmptyTest {
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

        WebElement emailInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[@placeholder='E-mail']")
        ));

        WebElement loginInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[@placeholder='Никнейм на Пикабу *']")
        ));

        WebElement passwordInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//input[@placeholder='Пароль *']")
        ));

        emailInput.clear();
        loginInput.clear();
        passwordInput.clear();

        WebElement submitButton = driver.findElement(
                By.xpath("//div[@class='tabs__tab auth tabs__tab_visible']//button[@type='submit']")
        );
        submitButton.click();

        WebElement validationMessage = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//span[contains(text(),'Обязательное поле')]")
        ));

        System.out.println("Validation message displayed: " + validationMessage.getText());

        Assertions.assertEquals("Обязательное поле", validationMessage.getText());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}