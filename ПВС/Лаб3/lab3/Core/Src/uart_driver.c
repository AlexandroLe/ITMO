#include "uart_driver.h"
#include "ring_buffer.h"

static UART_HandleTypeDef *s_huart = NULL;
static RingBuffer s_rx_ring;

void uart_driver_init(UART_HandleTypeDef *huart)
{
    s_huart = huart;
    ring_buffer_init(&s_rx_ring);

    if (s_huart != NULL) {
        /* Сбрасываем флаг RXNE чтением регистра DR */
        __HAL_UART_FLUSH_DRREGISTER(s_huart);

        /* Включаем прерывание по приёму байта (RXNE) */
        __HAL_UART_ENABLE_IT(s_huart, UART_IT_RXNE);
        NVIC_EnableIRQ(USART6_IRQn);
    }
}

bool uart_driver_read_char(char *out_char)
{
    if (s_huart == NULL || out_char == NULL) {
        return false;
    }

    uint8_t byte;
    if (ring_buffer_get(&s_rx_ring, &byte)) {
        *out_char = (char)byte;
        return true;
    }
    return false;
}

void uart_driver_write_char(char c)
{
    if (s_huart == NULL) return;

    /* Ожидание готовности передатчика (TXE) */
    while (__HAL_UART_GET_FLAG(s_huart, UART_FLAG_TXE) == RESET) {
    }
    s_huart->Instance->DR = (uint16_t)c;
}

void uart_driver_write_string(const char *str)
{
    if (str == NULL) return;
    while (*str) {
        uart_driver_write_char(*str++);
    }
}

void uart_driver_irq_handler(void)
{
    if (s_huart == NULL) return;

    /* Проверяем флаг поступления байта RXNE */
    if (__HAL_UART_GET_FLAG(s_huart, UART_FLAG_RXNE) != RESET &&
        __HAL_UART_GET_IT_SOURCE(s_huart, UART_IT_RXNE) != RESET)
    {
        uint8_t received_byte = (uint8_t)(s_huart->Instance->DR & 0xFF);
        ring_buffer_put(&s_rx_ring, received_byte);
    }

    /* Очистка флагов ошибок (Overrun, Noise, Framing), если возникли */
    if (__HAL_UART_GET_FLAG(s_huart, UART_FLAG_ORE) != RESET) {
        __HAL_UART_CLEAR_OREFLAG(s_huart);
    }
    if (__HAL_UART_GET_FLAG(s_huart, UART_FLAG_NE) != RESET) {
        __HAL_UART_CLEAR_NEFLAG(s_huart);
    }
    if (__HAL_UART_GET_FLAG(s_huart, UART_FLAG_FE) != RESET) {
        __HAL_UART_CLEAR_FEFLAG(s_huart);
    }
}
