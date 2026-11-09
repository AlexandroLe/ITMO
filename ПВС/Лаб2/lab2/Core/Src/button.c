/**
  ******************************************************************************
  * @file           : button.c
  * @brief          : Реализация драйвера кнопки PC15 с подавлением дребезга
  *                   и распознаванием коротких/длинных нажатий.
  ******************************************************************************
  */

#include "button.h"
#include "my_gpio.h"
#include "stm32f4xx_hal.h"

#define BTN_PIN             GPIO_PIN_15
#define BTN_PORT            GPIOC

#define DEBOUNCE_TIME_MS    35U   /* Время фильтрации дребезга контактов */
#define MIN_PRESS_TIME_MS   50U   /* Минимальная длительность валидного нажатия */
#define LONG_PRESS_TIME_MS  450U  /* Порог разделения: < 450 мс - короткое, >= 450 мс - длинное */

typedef enum {
    FSM_BTN_IDLE = 0,
    FSM_BTN_DEBOUNCE_DOWN,
    FSM_BTN_DOWN,
    FSM_BTN_DEBOUNCE_UP
} BtnFsmState;

static BtnFsmState s_state = FSM_BTN_IDLE;
static uint32_t s_state_start_tick = 0;
static uint32_t s_press_start_tick = 0;

static bool hw_read_button_pressed(void)
{
#if USE_CUSTOM_GPIO
    /* Активный уровень 0 (при нажатии замыкается на землю) */
    return (my_gpio_read_pin(BTN_PORT, BTN_PIN) == 0U);
#else
    return (HAL_GPIO_ReadPin(BTN_PORT, BTN_PIN) == GPIO_PIN_RESET);
#endif
}

void button_init(void)
{
#if USE_CUSTOM_GPIO
    __HAL_RCC_GPIOC_CLK_ENABLE();
    MyGPIOConfig cfg = {
        .pin_mask   = BTN_PIN,
        .mode       = MY_GPIO_MODE_INPUT,
        .pull       = MY_GPIO_PULL_UP,
        .speed      = MY_GPIO_SPEED_LOW,
        .init_state = 0
    };
    my_gpio_init(BTN_PORT, &cfg);
#endif
    s_state = FSM_BTN_IDLE;
}

ButtonEvent button_update(uint32_t current_tick)
{
    ButtonEvent event = BTN_EVENT_NONE;
    bool pressed = hw_read_button_pressed();

    switch (s_state) {
        case FSM_BTN_IDLE:
            if (pressed) {
                s_state = FSM_BTN_DEBOUNCE_DOWN;
                s_state_start_tick = current_tick;
            }
            break;

        case FSM_BTN_DEBOUNCE_DOWN:
            if ((current_tick - s_state_start_tick) >= DEBOUNCE_TIME_MS) {
                if (pressed) {
                    /* Дребезг прошел, кнопка гарантированно нажата */
                    s_state = FSM_BTN_DOWN;
                    s_press_start_tick = current_tick;
                } else {
                    s_state = FSM_BTN_IDLE;
                }
            }
            break;

        case FSM_BTN_DOWN:
            if (!pressed) {
                /* Пользователь начал отпускать кнопку */
                s_state = FSM_BTN_DEBOUNCE_UP;
                s_state_start_tick = current_tick;
            }
            break;

        case FSM_BTN_DEBOUNCE_UP:
            if ((current_tick - s_state_start_tick) >= DEBOUNCE_TIME_MS) {
                if (!pressed) {
                    /* Кнопка гарантированно отпущена, считаем длительность нажатия */
                    uint32_t duration = current_tick - s_press_start_tick;
                    if (duration >= LONG_PRESS_TIME_MS) {
                        event = BTN_EVENT_LONG;
                    } else if (duration >= MIN_PRESS_TIME_MS) {
                        event = BTN_EVENT_SHORT;
                    }
                    s_state = FSM_BTN_IDLE;
                } else {
                    /* Ложное размыкание, возвращаемся в состояние удержания */
                    s_state = FSM_BTN_DOWN;
                }
            }
            break;
    }

    return event;
}
