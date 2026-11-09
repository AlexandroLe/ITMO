#include "led.h"
#include "my_gpio.h"
#include "stm32f4xx_hal.h"

#define LED_PIN_GREEN   GPIO_PIN_13
#define LED_PIN_YEL     GPIO_PIN_14
#define LED_PIN_RED     GPIO_PIN_15
#define LED_GPIO_PORT   GPIOD

/* Структура для управления одиночной вспышкой */
typedef struct {
    bool     active;
    LedId    led;
    uint32_t end_tick;
} FlashEffect;

/* Структура для циклического мигания */
typedef struct {
    bool     active;
    LedId    led;
    uint32_t toggle_period_ms;
    uint32_t next_toggle_tick;
    uint32_t end_tick;
    bool     current_state;
} BlinkEffect;

static FlashEffect s_flash = {0};
static BlinkEffect s_blink = {0};

/* Низкоуровневая запись пинов (с поддержкой выбора: my_gpio или HAL) */
static void hw_write_leds(bool green, bool yel, bool red)
{
#if USE_CUSTOM_GPIO
    /* Запись зеленого (PD13) */
    my_gpio_write_pin(LED_GPIO_PORT, LED_PIN_GREEN, green ? 1 : 0);

    /* Запись двухцветного (PD14 / PD15) */
    uint16_t mask = LED_PIN_YEL | LED_PIN_RED;
    uint16_t val = 0;
    if (yel) {
        val = LED_PIN_YEL; /* PD14 = 1, PD15 = 0 */
    } else if (red) {
        val = LED_PIN_RED; /* PD14 = 0, PD15 = 1 */
    }
    my_gpio_write_port(LED_GPIO_PORT, mask, val);
#else
    HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_GREEN, green ? GPIO_PIN_SET : GPIO_PIN_RESET);
    if (yel) {
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_YEL, GPIO_PIN_SET);
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_RED, GPIO_PIN_RESET);
    } else if (red) {
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_YEL, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_RED, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_YEL, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN_RED, GPIO_PIN_RESET);
    }
#endif
}

static bool s_cur_green = false;
static bool s_cur_yel   = false;
static bool s_cur_red   = false;

void led_init(void)
{
#if USE_CUSTOM_GPIO
    __HAL_RCC_GPIOD_CLK_ENABLE();
    MyGPIOConfig cfg = {
        .pin_mask   = LED_PIN_GREEN | LED_PIN_YEL | LED_PIN_RED,
        .mode       = MY_GPIO_MODE_OUTPUT_PP,
        .pull       = MY_GPIO_PULL_NONE,
        .speed      = MY_GPIO_SPEED_LOW,
        .init_state = 0
    };
    my_gpio_init(LED_GPIO_PORT, &cfg);
#endif
    led_set(LED_ALL, LED_STATE_OFF);
}

void led_set(LedId led, LedState state)
{
    bool on = (state == LED_STATE_ON);
    switch (led) {
        case LED_GREEN:
            s_cur_green = on;
            break;
        case LED_YELLOW:
            s_cur_yel = on;
            if (on) s_cur_red = false; /* Взаимоисключение на двухцветном */
            break;
        case LED_RED:
            s_cur_red = on;
            if (on) s_cur_yel = false;
            break;
        case LED_ALL:
            s_cur_green = on;
            s_cur_yel   = false;
            s_cur_red   = false;
            break;
    }
    hw_write_leds(s_cur_green, s_cur_yel, s_cur_red);
}

void led_trigger_flash(LedId led, uint32_t duration_ms)
{
    led_stop_effects();
    led_set(led, LED_STATE_ON);
    s_flash.active   = true;
    s_flash.led      = led;
    s_flash.end_tick = HAL_GetTick() + duration_ms;
}

void led_start_blink(LedId led, uint32_t toggle_period_ms, uint32_t total_duration_ms)
{
    led_stop_effects();
    s_blink.active             = true;
    s_blink.led                = led;
    s_blink.toggle_period_ms   = toggle_period_ms;
    s_blink.current_state      = true;
    s_blink.next_toggle_tick   = HAL_GetTick() + toggle_period_ms;
    s_blink.end_tick           = HAL_GetTick() + total_duration_ms;

    led_set(led, LED_STATE_ON);
}

bool led_is_blinking(void)
{
    return s_blink.active;
}

void led_stop_effects(void)
{
    if (s_flash.active) {
        led_set(s_flash.led, LED_STATE_OFF);
        s_flash.active = false;
    }
    if (s_blink.active) {
        led_set(s_blink.led, LED_STATE_OFF);
        s_blink.active = false;
    }
}

void led_update(uint32_t current_tick)
{
    /* Обновление циклического мигания */
    if (s_blink.active) {
        if ((int32_t)(current_tick - s_blink.end_tick) >= 0) {
            /* Время мигания истекло */
            led_set(s_blink.led, LED_STATE_OFF);
            s_blink.active = false;
        } else if ((int32_t)(current_tick - s_blink.next_toggle_tick) >= 0) {
            s_blink.current_state = !s_blink.current_state;
            led_set(s_blink.led, s_blink.current_state ? LED_STATE_ON : LED_STATE_OFF);
            s_blink.next_toggle_tick = current_tick + s_blink.toggle_period_ms;
        }
        return;
    }

    /* Обновление одиночной вспышки */
    if (s_flash.active) {
        if ((int32_t)(current_tick - s_flash.end_tick) >= 0) {
            led_set(s_flash.led, LED_STATE_OFF);
            s_flash.active = false;
        }
    }
}
