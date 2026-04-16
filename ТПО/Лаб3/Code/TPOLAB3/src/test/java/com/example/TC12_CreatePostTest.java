package com.example;

import org.junit.jupiter.api.*;
import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.List;
@Disabled("Лимиты(((((")
public class TC12_CreatePostTest {
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
    public void testCreatePostSuccess() {
        driver.get("https://pikabu.ru/");

        WebElement addPostButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[@class='header-right-menu__item header-right-menu__add button_add']")
        ));
        addPostButton.click();

        WebElement titleInput = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//textarea[@placeholder='Заголовок']")
        ));
        titleInput.sendKeys("Важный вопрос");

        WebElement bodyInput = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//div[contains(@class,'story-editor-view__content')]")
        ));
        bodyInput.click();
        driver.switchTo().activeElement().sendKeys("Привет! Как вы считаете, какой самый нужный и полезный пост на сайте?");

        WebElement tagsInput = driver.findElement(
                By.xpath("//input[@class='pkb-input-tag__input--U63z_jON']")
        );
        tagsInput.click();
        tagsInput.sendKeys("вопрос, Обсуждение,");
        tagsInput.sendKeys(Keys.ENTER);

        List<WebElement> recommendedTags = driver.findElements(
                By.xpath("//div[@class='editor-tags__recommended--_Kv7LhI5']//svg")
        );

        if (recommendedTags.size() >= 2) {
            recommendedTags.get(0).click();
            recommendedTags.get(1).click();
        }

        WebElement submitButton = driver.findElement(
                By.xpath("//button[@class='pkb-btn__host--n2MGeea1 pkb-btn__host_wide--pv56KZ78']\n")
        );
        submitButton.click();

        WebElement postTitle = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//span[@class='story__title-link']")
        ));

        WebElement postBody = wait.until(ExpectedConditions.visibilityOfElementLocated(
                By.xpath("//p[contains(text(),'Привет, это тестовый пост.')]")
        ));

        System.out.println("Post created: " + postTitle.getText());

        Assertions.assertEquals("Мой первый автотест", postTitle.getText());
        Assertions.assertTrue(postBody.isDisplayed());
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}