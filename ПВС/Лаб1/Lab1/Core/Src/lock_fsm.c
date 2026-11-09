/**
  ******************************************************************************
  * @file           : lock_fsm.c
  * @brief          : Реализация конечного автомата кодового замка (Вариант 5).
  ******************************************************************************
  */

#include "lock_fsm.h"
#include "led.h"
#include <stdbool.h>

#define CODE_LENGTH         8U
#define INPUT_TIMEOUT_MS    5000U   /* Таймаут между нажатиями (5 секунд) */
#define UNLOCKED_TIME_MS    3000U   /* Время горения зеленого при открытии (3 секунды) */
#define ALARM_TIME_MS       5500U   /* Время мигания красного при 3 ошибках (не менее 5 сек) */
#define TIMEOUT_PULSE_MS    1500U   /* Длинный импульс желтого при таймауте */
#define FLASH_YELLOW_MS     200U    /* Короткая вспышка желтого при правильном элементе */
#define FLASH_RED_MS        250U    /* Вспышка красного при ошибке */

/* Секретная комбинация из 8 нажатий:
 * К, К, Д, К, Д, Д, К, К
 * (К = Короткое / точка, Д = Длинное / тире) */
static const ButtonEvent SECRET_CODE[CODE_LENGTH] = {
    BTN_EVENT_SHORT,
    BTN_EVENT_SHORT,
    BTN_EVENT_LONG,
    BTN_EVENT_SHORT,
    BTN_EVENT_LONG,
    BTN_EVENT_LONG,
    BTN_EVENT_SHORT,
    BTN_EVENT_SHORT
};

typedef enum {
    LOCK_STATE_INPUT = 0,
    LOCK_STATE_UNLOCKED,
    LOCK_STATE_ALARM_BLOCKED,
    LOCK_STATE_TIMEOUT_PULSE
} LockState;

static LockState s_lock_state = LOCK_STATE_INPUT;
static uint8_t   s_input_index = 0;
static uint8_t   s_error_count = 0;
static uint32_t  s_last_press_tick = 0;
static uint32_t  s_state_end_tick = 0;

void lock_fsm_init(void)
{
    s_lock_state      = LOCK_STATE_INPUT;
    s_input_index     = 0;
    s_error_count     = 0;
    s_last_press_tick = 0;
    s_state_end_tick  = 0;
    led_set(LED_ALL, LED_STATE_OFF);
}

void lock_fsm_update(ButtonEvent btn_event, uint32_t current_tick)
{
    switch (s_lock_state) {
        case LOCK_STATE_INPUT:
            /* 1. Проверка таймаута незавершенного ввода */
            if (s_input_index > 0) {
                if ((current_tick - s_last_press_tick) >= INPUT_TIMEOUT_MS) {
                    /* Таймаут истек: длинный импульс желтого и сброс */
                    s_input_index = 0;
                    led_trigger_flash(LED_YELLOW, TIMEOUT_PULSE_MS);
                    s_lock_state = LOCK_STATE_TIMEOUT_PULSE;
                    s_state_end_tick = current_tick + TIMEOUT_PULSE_MS;
                    return;
                }
            }

            /* 2. Обработка поступившего нажатия */
            if (btn_event != BTN_EVENT_NONE) {
                s_last_press_tick = current_tick;

                if (btn_event == SECRET_CODE[s_input_index]) {
                    /* Правильный элемент */
                    s_input_index++;

                    if (s_input_index >= CODE_LENGTH) {
                        /* Вся последовательность введена верно! Замок открыт! */
                        s_input_index = 0;
                        s_error_count = 0;
                        led_stop_effects();
                        led_set(LED_GREEN, LED_STATE_ON);
                        s_lock_state = LOCK_STATE_UNLOCKED;
                        s_state_end_tick = current_tick + UNLOCKED_TIME_MS;
                    } else {
                        /* Одиночное мигание желтого светодиода */
                        led_trigger_flash(LED_YELLOW, FLASH_YELLOW_MS);
                    }
                } else {
                    /* Неправильный элемент! */
                    s_input_index = 0;
                    s_error_count++;

                    if (s_error_count >= 3) {
                        /* 3 неверных попытки подряд: тревога не менее 5 секунд */
                        s_error_count = 0;
                        led_start_blink(LED_RED, 200, ALARM_TIME_MS);
                        s_lock_state = LOCK_STATE_ALARM_BLOCKED;
                        s_state_end_tick = current_tick + ALARM_TIME_MS;
                    } else {
                        /* Одиночное мигание красного светодиода */
                        led_trigger_flash(LED_RED, FLASH_RED_MS);
                    }
                }
            }
            break;

        case LOCK_STATE_UNLOCKED:
            /* Замок открыт: горит зеленый светодиод, нажатия игнорируются */
            if ((int32_t)(current_tick - s_state_end_tick) >= 0) {
                led_set(LED_GREEN, LED_STATE_OFF);
                s_lock_state = LOCK_STATE_INPUT;
            }
            break;

        case LOCK_STATE_ALARM_BLOCKED:
            /* Блокировка после 3 ошибок: красный мигает, нажатия игнорируются */
            if ((int32_t)(current_tick - s_state_end_tick) >= 0 && !led_is_blinking()) {
                s_lock_state = LOCK_STATE_INPUT;
            }
            break;

        case LOCK_STATE_TIMEOUT_PULSE:
            /* Длинный импульс желтого при сбросе по таймауту */
            if ((int32_t)(current_tick - s_state_end_tick) >= 0) {
                s_lock_state = LOCK_STATE_INPUT;
            }
            break;
    }
}
