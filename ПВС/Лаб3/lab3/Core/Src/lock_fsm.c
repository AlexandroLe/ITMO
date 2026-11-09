#include "lock_fsm.h"
#include "oled.h"
#include "eeprom.h"
#include "led.h"
#include "uart_driver.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define DEFAULT_PASSWORD        "1234"
#define MAX_PASSWORD_LEN        16
#define INPUT_TIMEOUT_MS        10000   /* 10 секунд на ввод */
#define OPEN_DURATION_MS        3000    /* 3 секунды замок открыт */
#define BLOCK_DURATION_MS       5000    /* 5 секунд блокировки после 3 ошибок */
#define MAX_WRONG_ATTEMPTS      3

typedef enum {
    STATE_INPUT,            /* Ожидание ввода пароля */
    STATE_UNLOCKED,         /* Доступ разрешен */
    STATE_WRONG_FLASH,      /* Одиночная ошибка (мигание красным) */
    STATE_BLOCKED,          /* 3 ошибки: блокировка с миганием */
    STATE_SET_NEW_PASSWORD, /* Ввод нового пароля */
    STATE_CONFIRM_PASSWORD  /* Подтверждение (y/n) */
} LockState;

static LockState s_state = STATE_INPUT;
static char s_master_password[MAX_PASSWORD_LEN + 1] = {0};
static char s_input_buffer[MAX_PASSWORD_LEN + 1] = {0};
static uint8_t s_input_len = 0;
static char s_new_pass_buffer[MAX_PASSWORD_LEN + 1] = {0};
static uint8_t s_new_pass_len = 0;

static uint8_t s_wrong_count = 0;
static uint32_t s_state_start_tick = 0;
static uint32_t s_last_char_tick = 0;
static bool s_screen_dirty = true;

/* Буфер для построчной отправки команд по UART */
static char s_uart_cmd_buf[64];
static uint8_t s_uart_cmd_len = 0;

static void update_oled_display(void)
{
    oled_Fill(Black);
    oled_DrawSquare(0, 127, 0, 63, White);

    oled_SetCursor(10, 5);
    oled_WriteString("CODE LOCK", Font_11x18, White);

    switch (s_state) {
        case STATE_INPUT: {
            oled_SetCursor(10, 26);
            oled_WriteString("Status: LOCKED", Font_7x10, White);

            char mask[MAX_PASSWORD_LEN + 1] = {0};
            for (uint8_t i = 0; i < s_input_len; i++) {
                mask[i] = '*';
            }
            if (s_input_len == 0) {
                uint8_t target_len = (uint8_t)strlen(s_master_password);
                if (target_len < 1) target_len = 4;
                for (uint8_t i = 0; i < target_len; i++) {
                    mask[i] = '_';
                }
            }
            oled_SetCursor(20, 44);
            oled_WriteString(mask, Font_11x18, White);
            break;
        }

        case STATE_UNLOCKED:
            oled_SetCursor(10, 26);
            oled_WriteString(">> OPENED! <<", Font_7x10, White);
            oled_SetCursor(15, 44);
            oled_WriteString("ACCESS OK", Font_11x18, White);
            break;

        case STATE_WRONG_FLASH:
            oled_SetCursor(10, 26);
            oled_WriteString(">> ERROR! <<", Font_7x10, White);
            oled_SetCursor(15, 44);
            oled_WriteString("WRONG CODE", Font_11x18, White);
            break;

        case STATE_BLOCKED: {
            oled_SetCursor(10, 26);
            oled_WriteString("!! BLOCKED !!", Font_7x10, White);
            char b_buf[16];
            uint32_t rem = 0;
            uint32_t el = HAL_GetTick() - s_state_start_tick;
            if (el < BLOCK_DURATION_MS) rem = (BLOCK_DURATION_MS - el) / 1000 + 1;
            snprintf(b_buf, sizeof(b_buf), "Wait: %lu s", rem);
            oled_SetCursor(25, 44);
            oled_WriteString(b_buf, Font_7x10, White);
            break;
        }

        case STATE_SET_NEW_PASSWORD: {
            oled_SetCursor(10, 26);
            oled_WriteString("NEW PASS:", Font_7x10, White);
            char np[MAX_PASSWORD_LEN + 1] = {0};
            strncpy(np, s_new_pass_buffer, sizeof(np) - 1);
            if (s_new_pass_len == 0) {
                strcpy(np, "____");
            }
            oled_SetCursor(20, 44);
            oled_WriteString(np, Font_11x18, White);
            break;
        }

        case STATE_CONFIRM_PASSWORD:
            oled_SetCursor(10, 26);
            oled_WriteString("Save to EEPROM?", Font_7x10, White);
            oled_SetCursor(20, 44);
            oled_WriteString("Press 'y' / 'n'", Font_7x10, White);
            break;
    }

    oled_UpdateScreen();
    s_screen_dirty = false;
}

void lock_fsm_init(void)
{
    /* Загружаем мастер-пароль из EEPROM (по умолчанию "1234") */
    eeprom_load_password(s_master_password, DEFAULT_PASSWORD);

    s_state = STATE_INPUT;
    s_input_len = 0;
    memset(s_input_buffer, 0, sizeof(s_input_buffer));
    s_wrong_count = 0;
    s_state_start_tick = HAL_GetTick();
    s_last_char_tick = HAL_GetTick();

    led_init();
    update_oled_display();

    uart_driver_write_string("STATUS:LOCKED\r\n");
    uart_driver_write_string("DISPLAY:____\r\n");
}

void lock_fsm_reset(void)
{
    s_state = STATE_INPUT;
    s_input_len = 0;
    memset(s_input_buffer, 0, sizeof(s_input_buffer));
    led_set(LED_ALL, LED_STATE_OFF);
    s_screen_dirty = true;
    uart_driver_write_string("STATUS:LOCKED\r\n");
    uart_driver_write_string("DISPLAY:____\r\n");
}

/* Обработка строки протокола от клиентской программы */
static void handle_protocol_command(const char *cmd)
{
    if (strcmp(cmd, "PING") == 0) {
        uart_driver_write_string("PONG\r\n");
        return;
    }

    if (strncmp(cmd, "AUTH:", 5) == 0) {
        const char *pass = cmd + 5;
        if (s_state == STATE_BLOCKED) {
            uart_driver_write_string("STATUS:BLOCKED\r\n");
            return;
        }

        /* Приводим к верхнему регистру для сравнения без учёта регистра */
        char check_pass[MAX_PASSWORD_LEN + 1] = {0};
        char master_upper[MAX_PASSWORD_LEN + 1] = {0};
        uint8_t pass_len = (uint8_t)strlen(pass);
        uint8_t target_len = (uint8_t)strlen(s_master_password);

        for (int i = 0; i < pass_len; i++) {
            check_pass[i] = (char)toupper((unsigned char)pass[i]);
        }
        for (int i = 0; i < target_len; i++) {
            master_upper[i] = (char)toupper((unsigned char)s_master_password[i]);
        }

        if (strcmp(check_pass, master_upper) == 0) {
            s_state = STATE_UNLOCKED;
            s_state_start_tick = HAL_GetTick();
            s_wrong_count = 0;
            s_input_len = 0;
            led_stop_effects();
            led_set(LED_GREEN, LED_STATE_ON);
            s_screen_dirty = true;
            uart_driver_write_string("STATUS:OK\r\n");
            uart_driver_write_string("DISPLAY:OPENED\r\n");
        } else {
            s_wrong_count++;
            s_input_len = 0;
            memset(s_input_buffer, 0, sizeof(s_input_buffer));
            if (s_wrong_count >= MAX_WRONG_ATTEMPTS) {
                s_state = STATE_BLOCKED;
                s_state_start_tick = HAL_GetTick();
                led_start_blink(LED_RED, 200, BLOCK_DURATION_MS);
                uart_driver_write_string("STATUS:BLOCKED\r\n");
            } else {
                s_state = STATE_WRONG_FLASH;
                s_state_start_tick = HAL_GetTick();
                led_trigger_flash(LED_RED, 500);
                uart_driver_write_string("STATUS:ERR\r\n");
            }
            s_screen_dirty = true;
            uart_driver_write_string("DISPLAY:ERROR\r\n");
        }
        return;
    }

    if (strncmp(cmd, "SET:", 4) == 0) {
        const char *new_p = cmd + 4;
        uint8_t nlen = (uint8_t)strlen(new_p);
        if (nlen >= 1 && nlen <= MAX_PASSWORD_LEN) {
            strncpy(s_master_password, new_p, MAX_PASSWORD_LEN);
            s_master_password[MAX_PASSWORD_LEN] = '\0';
            eeprom_save_password(s_master_password);

            uart_driver_write_string("STATUS:SAVED\r\n");
            s_state = STATE_INPUT;
            s_input_len = 0;
            s_screen_dirty = true;
            update_oled_display();
        } else {
            uart_driver_write_string("STATUS:SET_ERR\r\n");
        }
        return;
    }

    if (strcmp(cmd, "GET_STATUS") == 0) {
        if (s_state == STATE_UNLOCKED) uart_driver_write_string("STATUS:OK\r\n");
        else if (s_state == STATE_BLOCKED) uart_driver_write_string("STATUS:BLOCKED\r\n");
        else uart_driver_write_string("STATUS:LOCKED\r\n");
        return;
    }
}

void lock_fsm_process_char(char c)
{
    /* Накопление строки протокола по UART */
    if (c == '\r' || c == '\n') {
        if (s_uart_cmd_len > 0) {
            s_uart_cmd_buf[s_uart_cmd_len] = '\0';
            handle_protocol_command(s_uart_cmd_buf);
            s_uart_cmd_len = 0;
        } else if (s_state == STATE_SET_NEW_PASSWORD && s_new_pass_len >= 1) {
            s_state = STATE_CONFIRM_PASSWORD;
            s_screen_dirty = true;
        }
        return;
    }

    if (s_uart_cmd_len < sizeof(s_uart_cmd_buf) - 1) {
        s_uart_cmd_buf[s_uart_cmd_len++] = c;
    }

    if (s_state == STATE_BLOCKED) return;

    if (c == '+') {
        s_state = STATE_SET_NEW_PASSWORD;
        s_new_pass_len = 0;
        memset(s_new_pass_buffer, 0, sizeof(s_new_pass_buffer));
        s_screen_dirty = true;
        return;
    }

    if (s_state == STATE_CONFIRM_PASSWORD) {
        if (c == 'y' || c == 'Y') {
            strncpy(s_master_password, s_new_pass_buffer, MAX_PASSWORD_LEN);
            s_master_password[MAX_PASSWORD_LEN] = '\0';
            eeprom_save_password(s_master_password);
            uart_driver_write_string("STATUS:SAVED\r\n");
            s_state = STATE_INPUT;
            s_input_len = 0;
            s_screen_dirty = true;
        } else if (c == 'n' || c == 'N') {
            s_state = STATE_INPUT;
            s_input_len = 0;
            s_screen_dirty = true;
        }
        return;
    }

    if (s_state == STATE_SET_NEW_PASSWORD) {
        if (isalnum((unsigned char)c) && s_new_pass_len < MAX_PASSWORD_LEN) {
            s_new_pass_buffer[s_new_pass_len++] = c;
            s_screen_dirty = true;
            if (s_new_pass_len >= 8) {
                s_state = STATE_CONFIRM_PASSWORD;
            }
        }
        return;
    }

    if (s_state == STATE_INPUT) {
        if (isalnum((unsigned char)c)) {
            s_last_char_tick = HAL_GetTick();

            uint8_t target_len = (uint8_t)strlen(s_master_password);
            char expected = (char)toupper((unsigned char)s_master_password[s_input_len]);
            char input_up = (char)toupper((unsigned char)c);

            if (input_up == expected) {
                s_input_buffer[s_input_len++] = c;
                led_trigger_flash(LED_YELLOW, 200);
                s_screen_dirty = true;

                char disp_msg[32];
                snprintf(disp_msg, sizeof(disp_msg), "DISPLAY:%.*s\r\n", s_input_len, s_input_buffer);
                uart_driver_write_string(disp_msg);

                if (s_input_len >= target_len) {
                    /* Пароль полностью введён верно */
                    s_state = STATE_UNLOCKED;
                    s_state_start_tick = HAL_GetTick();
                    s_wrong_count = 0;
                    s_input_len = 0;
                    led_set(LED_GREEN, LED_STATE_ON);
                    s_screen_dirty = true;
                    uart_driver_write_string("STATUS:OK\r\n");
                }
            } else {
                /* Ошибка */
                s_wrong_count++;
                s_input_len = 0;
                memset(s_input_buffer, 0, sizeof(s_input_buffer));
                if (s_wrong_count >= MAX_WRONG_ATTEMPTS) {
                    s_state = STATE_BLOCKED;
                    s_state_start_tick = HAL_GetTick();
                    led_start_blink(LED_RED, 200, BLOCK_DURATION_MS);
                    uart_driver_write_string("STATUS:BLOCKED\r\n");
                } else {
                    s_state = STATE_WRONG_FLASH;
                    s_state_start_tick = HAL_GetTick();
                    led_trigger_flash(LED_RED, 500);
                    uart_driver_write_string("STATUS:ERR\r\n");
                }
                s_screen_dirty = true;
                uart_driver_write_string("DISPLAY:ERROR\r\n");
            }
        }
    }
}

void lock_fsm_update(uint32_t current_tick)
{
    led_update(current_tick);

    switch (s_state) {
        case STATE_INPUT:
            if (s_input_len > 0 && (current_tick - s_last_char_tick >= INPUT_TIMEOUT_MS)) {
                s_input_len = 0;
                memset(s_input_buffer, 0, sizeof(s_input_buffer));
                s_screen_dirty = true;
                uart_driver_write_string("STATUS:TIMEOUT\r\n");
                uart_driver_write_string("DISPLAY:____\r\n");
            }
            break;

        case STATE_UNLOCKED:
            if (current_tick - s_state_start_tick >= OPEN_DURATION_MS) {
                led_set(LED_GREEN, LED_STATE_OFF);
                s_state = STATE_INPUT;
                s_input_len = 0;
                s_screen_dirty = true;
                uart_driver_write_string("STATUS:LOCKED\r\n");
                uart_driver_write_string("DISPLAY:____\r\n");
            }
            break;

        case STATE_WRONG_FLASH:
            if (current_tick - s_state_start_tick >= 1000) {
                s_state = STATE_INPUT;
                s_screen_dirty = true;
                uart_driver_write_string("STATUS:LOCKED\r\n");
                uart_driver_write_string("DISPLAY:____\r\n");
            }
            break;

        case STATE_BLOCKED:
            if (current_tick - s_state_start_tick >= BLOCK_DURATION_MS) {
                s_wrong_count = 0;
                led_stop_effects();
                led_set(LED_RED, LED_STATE_OFF);
                s_state = STATE_INPUT;
                s_input_len = 0;
                s_screen_dirty = true;
                uart_driver_write_string("STATUS:LOCKED\r\n");
                uart_driver_write_string("DISPLAY:____\r\n");
            } else {
                static uint32_t last_block_sec = 0;
                uint32_t cur_sec = (current_tick - s_state_start_tick) / 1000;
                if (cur_sec != last_block_sec) {
                    last_block_sec = cur_sec;
                    s_screen_dirty = true;
                }
            }
            break;

        default:
            break;
    }

    if (s_screen_dirty) {
        update_oled_display();
    }
}
