/**
  ******************************************************************************
  * @file           : my_gpio.h
  * @brief          : Собственный драйвер GPIO на уровне регистров (Часть 2)
  *                   для стенда SDK-1.1M (STM32F427VI).
  ******************************************************************************
  */

#ifndef INC_MY_GPIO_H_
#define INC_MY_GPIO_H_

#include <stdint.h>
#include "stm32f4xx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Переключатель: 1 - использовать собственный регистровый драйвер, 0 - стандартный HAL */
#define USE_CUSTOM_GPIO 1

/* Режимы работы пина */
typedef enum {
    MY_GPIO_MODE_INPUT      = 0x00, /* Вход (по умолчанию) */
    MY_GPIO_MODE_OUTPUT_PP  = 0x01, /* Выход Push-Pull */
    MY_GPIO_MODE_OUTPUT_OD  = 0x02, /* Выход Open-Drain */
    MY_GPIO_MODE_ANALOG     = 0x03  /* Аналоговый режим */
} MyGPIOMode;

/* Режимы подтяжки */
typedef enum {
    MY_GPIO_PULL_NONE       = 0x00, /* Без подтяжки */
    MY_GPIO_PULL_UP         = 0x01, /* Подтяжка к питанию (Pull-Up) */
    MY_GPIO_PULL_DOWN       = 0x02  /* Подтяжка к земле (Pull-Down) */
} MyGPIOPull;

/* Скорость работы порта */
typedef enum {
    MY_GPIO_SPEED_LOW       = 0x00, /* Low speed (2 MHz) */
    MY_GPIO_SPEED_MEDIUM    = 0x01, /* Medium speed (25 MHz) */
    MY_GPIO_SPEED_FAST      = 0x02, /* Fast speed (50 MHz) */
    MY_GPIO_SPEED_HIGH      = 0x03  /* High speed (100 MHz) */
} MyGPIOSpeed;

/* Структура конфигурации одного или нескольких пинов порта */
typedef struct {
    uint16_t    pin_mask;    /* Битовая маска пинов (например, GPIO_PIN_13 | GPIO_PIN_14) */
    MyGPIOMode  mode;        /* Режим работы */
    MyGPIOPull  pull;        /* Подтяжка */
    MyGPIOSpeed speed;       /* Скорость */
    uint8_t     init_state;  /* Начальное состояние для выходов (0 или 1), исключает выбросы */
} MyGPIOConfig;

/* Функция инициализации (поддерживает безопасную повторную инициализацию без выбросов) */
void my_gpio_init(GPIO_TypeDef *GPIOx, const MyGPIOConfig *config);

/* Получение текущих настроек конкретного пина */
void my_gpio_get_config(GPIO_TypeDef *GPIOx, uint16_t pin, MyGPIOConfig *config);

/* Запись одиночного пина (через атомарный регистр BSRR) */
void my_gpio_write_pin(GPIO_TypeDef *GPIOx, uint16_t pin, uint8_t state);

/* Чтение одиночного пина */
uint8_t my_gpio_read_pin(GPIO_TypeDef *GPIOx, uint16_t pin);

/* Инвертирование одиночного пина */
void my_gpio_toggle_pin(GPIO_TypeDef *GPIOx, uint16_t pin);

/* Одновременная запись нескольких сигналов порта по маске */
void my_gpio_write_port(GPIO_TypeDef *GPIOx, uint16_t mask, uint16_t value);

/* Одновременное чтение нескольких сигналов порта по маске */
uint16_t my_gpio_read_port(GPIO_TypeDef *GPIOx, uint16_t mask);

#ifdef __cplusplus
}
#endif

#endif /* INC_MY_GPIO_H_ */
