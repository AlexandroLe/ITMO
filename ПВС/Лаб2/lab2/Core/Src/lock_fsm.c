#include "lock_fsm.h"
#include "led.h"
#include "uart_driver.h"
#include <ctype.h>
#include <string.h>
#include <stdio.h>

#define CODE_MAX_LEN        8U
#define INPUT_TIMEOUT_MS    5000U   /* Таймаут между вводами символов (5 секунд) */
#define UNLOCKED_TIME_MS    3000U   /* Зеленый горит 3 сек при открытии */
#define ALARM_TIME_MS       5000U   /* Красный мигает 5 сек при 3 ошибках */
#define FLASH_YELLOW_MS     200U    /* Желтый мигает при правильном символе */
#define FLASH_RED_MS        300U    /* Красный мигает при ошибке */

typedef enum {
    STATE_INPUT_CODE = 0,       /* Обычный ввод кода */
    STATE_UNLOCKED,             /* Замок открыт (зеленый горит) */
    STATE_BLOCKED_ALARM,        /* Заблокирован после 3 ошибок (красный мигает) */
    STATE_NEW_CODE_INPUT,       /* Ввод нового пароля (после нажатия '+') */
    STATE_NEW_CODE_CONFIRM      /* Запрос подтверждения нового пароля (y/n) */
} LockState;

static LockState s_state = STATE_INPUT_CODE;

/* Активный пароль */
static char s_active_code[CODE_MAX_LEN + 1] = "12345678";

/* Буфер для ввода нового пароля при смене */
static char s_temp_code[CODE_MAX_LEN + 1];
static uint8_t s_temp_code_len = 0;

/* Индекс текущего вводимого символа */
static uint8_t s_input_idx = 0;

/* Счётчик неверных попыток */
static uint8_t s_error_count = 0;

/* Временные метки */
static uint32_t s_last_input_tick = 0;
static uint32_t s_state_end_tick = 0;

/* Вспомогательная функция сравнения символов без учёта регистра */
static bool char_match(char a, char b)
{
    return (tolower((unsigned char)a) == tolower((unsigned char)b));
}

void lock_fsm_init(void)
{
    s_state = STATE_INPUT_CODE;
    s_input_idx = 0;
    s_error_count = 0;
    s_last_input_tick = 0;
    s_state_end_tick = 0;
    led_set(LED_ALL, LED_STATE_OFF);
    uart_driver_write_string("\r\n[Lock initialized. Enter 8-char code or '+' to change]\r\n> ");
}

void lock_fsm_reset(void)
{
    s_state = STATE_INPUT_CODE;
    s_input_idx = 0;
    s_error_count = 0;
    s_last_input_tick = 0;
    led_stop_effects();
    led_set(LED_ALL, LED_STATE_OFF);
    uart_driver_write_string("\r\n[Reset to code input]\r\n> ");
}

void lock_fsm_process_char(char c)
{
    /* Игнорируем ввод, если замок сейчас открыт или заблокирован тревогой */
    if (s_state == STATE_UNLOCKED || s_state == STATE_BLOCKED_ALARM) {
        return;
    }

    /* Эхо-вывод принятого символа в консоль */
    uart_driver_write_char(c);

    uint32_t now = HAL_GetTick();

    switch (s_state) {
        case STATE_INPUT_CODE:
            /* Cмены пароля по символу '+' */
            if (c == '+') {
                s_state = STATE_NEW_CODE_INPUT;
                s_temp_code_len = 0;
                memset(s_temp_code, 0, sizeof(s_temp_code));
                uart_driver_write_string("\r\n[Enter new code (up to 8 chars, press Enter to finish)]:\r\n> ");
                return;
            }

            if (c == '\r' || c == '\n') {
                return;
            }

            s_last_input_tick = now;

            /* Проверяем соответствие текущему символу активного пароля */
            if (char_match(c, s_active_code[s_input_idx])) {
                s_input_idx++;
                /* Каждый правильный символ сопровождается миганием жёлтого */
                led_trigger_flash(LED_YELLOW, FLASH_YELLOW_MS);

                if (s_input_idx >= CODE_MAX_LEN) {
                    /* Замок открыт */
                    s_input_idx = 0;
                    s_error_count = 0;
                    led_stop_effects();
                    led_set(LED_GREEN, LED_STATE_ON);
                    s_state = STATE_UNLOCKED;
                    s_state_end_tick = now + UNLOCKED_TIME_MS;
                    uart_driver_write_string("\r\n*** ACCESS GRANTED (LOCK OPEN) ***\r\n");
                }
            } else {
                /* Ошибка ввода символа */
                s_input_idx = 0;
                s_error_count++;
                /* Каждый неправильно введённый элемент сопровождается миганием красного и сбросом в начало */
                led_trigger_flash(LED_RED, FLASH_RED_MS);
                uart_driver_write_string("\r\n[Wrong character! Reset to start]\r\n> ");

                if (s_error_count >= 3) {
                    /* Cистема блокируется */
                    s_error_count = 0;
                    s_state = STATE_BLOCKED_ALARM;
                    s_state_end_tick = now + ALARM_TIME_MS;
                    led_start_blink(LED_RED, 200, ALARM_TIME_MS);
                    uart_driver_write_string("\r\n!!! 3 WRONG ATTEMPTS! SYSTEM LOCKED !!!\r\n");
                }
            }
            break;

        case STATE_NEW_CODE_INPUT:
            /* Окончание ввода */
            if (c == '\r' || c == '\n') {
                if (s_temp_code_len > 0) {
                    s_temp_code[s_temp_code_len] = '\0';
                    s_state = STATE_NEW_CODE_CONFIRM;
                    uart_driver_write_string("\r\nMake this code active? (y/n): ");
                } else {
                    /* Пустой пароль низя */
                    uart_driver_write_string("\r\n[Empty code. Cancelled]\r\n> ");
                    s_state = STATE_INPUT_CODE;
                }
            } else {
                if (s_temp_code_len < CODE_MAX_LEN) {
                    s_temp_code[s_temp_code_len++] = c;
                    if (s_temp_code_len == CODE_MAX_LEN) {
                        s_temp_code[s_temp_code_len] = '\0';
                        s_state = STATE_NEW_CODE_CONFIRM;
                        uart_driver_write_string("\r\nMake this code active? (y/n): ");
                    }
                }
            }
            break;

        case STATE_NEW_CODE_CONFIRM:
            if (c == 'y' || c == 'Y') {
                /* Подтверждение: новый пароль становится активным */
                strncpy(s_active_code, s_temp_code, CODE_MAX_LEN + 1);
                s_state = STATE_INPUT_CODE;
                s_input_idx = 0;
                s_error_count = 0;
                uart_driver_write_string("\r\n[New code successfully activated!]\r\n> ");
            } else {
                s_state = STATE_INPUT_CODE;
                s_input_idx = 0;
                uart_driver_write_string("\r\n[Code change cancelled]\r\n> ");
            }
            break;

        default:
            break;
    }
}

void lock_fsm_update(uint32_t current_tick)
{
    switch (s_state) {
        case STATE_INPUT_CODE:
            /* Таймаут незавершённого ввода кода */
            if (s_input_idx > 0) {
                if ((current_tick - s_last_input_tick) >= INPUT_TIMEOUT_MS) {
                    s_input_idx = 0;
                    led_trigger_flash(LED_RED, FLASH_RED_MS);
                    uart_driver_write_string("\r\n[Input timeout! Reset to start]\r\n> ");
                }
            }
            break;

        case STATE_UNLOCKED:
            /* Зеленый горит некоторое время, затем гаснет и возвращаемся в режим ввода */
            if ((int32_t)(current_tick - s_state_end_tick) >= 0) {
                led_set(LED_GREEN, LED_STATE_OFF);
                s_state = STATE_INPUT_CODE;
                s_input_idx = 0;
                uart_driver_write_string("[Lock closed. Enter code]\r\n> ");
            }
            break;

        case STATE_BLOCKED_ALARM:
            /* Блокировка на 5 сек после 3 ошибок завершилась */
            if ((int32_t)(current_tick - s_state_end_tick) >= 0) {
                led_stop_effects();
                led_set(LED_RED, LED_STATE_OFF);
                s_state = STATE_INPUT_CODE;
                s_input_idx = 0;
                uart_driver_write_string("[Alarm finished. Enter code]\r\n> ");
            }
            break;

        default:
            break;
    }
}
