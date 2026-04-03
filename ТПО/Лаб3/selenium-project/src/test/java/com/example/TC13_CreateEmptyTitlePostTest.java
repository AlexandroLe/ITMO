package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.*;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class TC13_CreateEmptyTitlePostTest {
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
    public void testCreatePostWithoutTitle() {
        driver.get("https://pikabu.ru/");

        WebElement addPostButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[@class='header-right-menu__item header-right-menu__add button_add']")
        ));
        addPostButton.click();

        WebElement bodyInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//p[@class='node-paragraph-view__content--hRxcjf1f']")
        ));
        bodyInput.sendKeys("Привет, это тестовый пост.");

        WebElement submitButton = driver.findElement(
                By.xpath("//button[contains(@class,'pkb-btn__host_wide')]//span[@class='pkb-btn__text--FO_UA9yM']")
        );
        submitButton.click();

        WebElement validation = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//div[@class='toast__content']")
        ));

        System.out.println("Validation message: " + validation.getText());

        Assertions.assertTrue(validation.getText().contains("Укажите заголовок"));
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}