#ifndef INC_UART_DRIVER_H_
#define INC_UART_DRIVER_H_

#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/* Режимы работы драйвера UART */
typedef enum {
    UART_MODE_POLLING = 0,   /* Без прерываний (опрос регистров, неблокирующий) */
    UART_MODE_INTERRUPT      /* С использованием прерываний (FIFO буферизация) */
} UartDriverMode;

/* Инициализация драйвера UART */
void uart_driver_init(UART_HandleTypeDef *huart);

/* Переключение режима работы (прерывания вкл/выкл) */
void uart_driver_set_mode(UartDriverMode mode);

/* Получить текущий режим */
UartDriverMode uart_driver_get_mode(void);

/* Неблокирующее чтение символа.
 * Возвращает true, если символ получен (записывает его в *out_char).
 * Возвращает false, если данных нет (не зависает!). */
bool uart_driver_read_char(char *out_char);

/* Отправка одного символа */
void uart_driver_write_char(char c);

/* Отправка строки */
void uart_driver_write_string(const char *str);

/* Обработчик прерывания UART (вызывается из USART6_IRQHandler) */
void uart_driver_irq_handler(void);

#endif /* INC_UART_DRIVER_H_ */
