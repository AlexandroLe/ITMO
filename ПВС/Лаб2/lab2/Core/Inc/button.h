/**
  ******************************************************************************
  * @file           : button.h
  * @brief          : Неблокирующий драйвер кнопки с защитой от дребезга
  *                   и распознаванием короткого/длинного нажатия (PC15).
  ******************************************************************************
  */

#ifndef INC_BUTTON_H_
#define INC_BUTTON_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BTN_EVENT_NONE = 0,
    BTN_EVENT_SHORT,    /* Короткое нажатие (точка) */
    BTN_EVENT_LONG      /* Длинное нажатие (тире) */
} ButtonEvent;

/* Инициализация кнопки */
void button_init(void);

/* Опрос состояния и генерация событий (неблокирующий, вызывается в цикле main) */
ButtonEvent button_update(uint32_t current_tick);

#ifdef __cplusplus
}
#endif

#endif /* INC_BUTTON_H_ */
