#ifndef INC_UART_DRIVER_H_
#define INC_UART_DRIVER_H_

#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/* Инициализация драйвера UART */
void uart_driver_init(UART_HandleTypeDef *huart);

/* Неблокирующее чтение символа из буфера/порта */
bool uart_driver_read_char(char *out_char);

/* Вывод одиночного символа */
void uart_driver_write_char(char c);

/* Вывод нуль-терминированной строки */
void uart_driver_write_string(const char *str);

/* Обработчик прерывания (вызывается из USART6_IRQHandler) */
void uart_driver_irq_handler(void);

#endif /* INC_UART_DRIVER_H_ */
