/**
  ******************************************************************************
  * @file           : my_gpio.c
  * @brief          : Собственный регистровый драйвер GPIO (Часть 2)
  *                   для стенда SDK-1.1M (STM32F427VI).
  ******************************************************************************
  */

#include "my_gpio.h"

void my_gpio_init(GPIO_TypeDef *GPIOx, const MyGPIOConfig *config)
{
    if (GPIOx == NULL || config == NULL) {
        return;
    }

    for (uint32_t pos = 0; pos < 16; pos++) {
        uint16_t pin = (1U << pos);
        if (config->pin_mask & pin) {
            /* 1. Если настраивается выход, предварительно выставляем начальный уровень через BSRR.
             * Это предотвращает кратковременные импульсные помехи ("выбросы") при переключении режима */
            if (config->mode == MY_GPIO_MODE_OUTPUT_PP || config->mode == MY_GPIO_MODE_OUTPUT_OD) {
                if (config->init_state) {
                    GPIOx->BSRR = pin;           /* BS: установка 1 */
                } else {
                    GPIOx->BSRR = (pin << 16U);  /* BR: сброс в 0 */
                }
            }

            /* 2. Настройка типа выхода (Push-Pull / Open-Drain) */
            if (config->mode == MY_GPIO_MODE_OUTPUT_OD) {
                GPIOx->OTYPER |= (1U << pos);
            } else {
                GPIOx->OTYPER &= ~(1U << pos);
            }

            /* 3. Настройка скорости порта (OSPEEDR, 2 бита на пин) */
            GPIOx->OSPEEDR &= ~(0x3U << (pos * 2U));
            GPIOx->OSPEEDR |= ((uint32_t)config->speed << (pos * 2U));

            /* 4. Настройка подтяжки (PUPDR, 2 бита на пин) */
            GPIOx->PUPDR &= ~(0x3U << (pos * 2U));
            GPIOx->PUPDR |= ((uint32_t)config->pull << (pos * 2U));

            /* 5. Настройка режима работы (MODER, 2 бита на пин) */
            uint32_t mode_val = 0;
            switch (config->mode) {
                case MY_GPIO_MODE_INPUT:     mode_val = 0x0U; break;
                case MY_GPIO_MODE_OUTPUT_PP:
                case MY_GPIO_MODE_OUTPUT_OD: mode_val = 0x1U; break;
                case MY_GPIO_MODE_ANALOG:    mode_val = 0x3U; break;
                default:                     mode_val = 0x0U; break;
            }
            GPIOx->MODER &= ~(0x3U << (pos * 2U));
            GPIOx->MODER |= (mode_val << (pos * 2U));
        }
    }
}

void my_gpio_get_config(GPIO_TypeDef *GPIOx, uint16_t pin, MyGPIOConfig *config)
{
    if (GPIOx == NULL || config == NULL || pin == 0) {
        return;
    }

    uint32_t pos = 0;
    while ((pin & (1U << pos)) == 0 && pos < 16) {
        pos++;
    }
    if (pos >= 16) return;

    config->pin_mask = (1U << pos);

    /* Считывание MODER */
    uint32_t moder = (GPIOx->MODER >> (pos * 2U)) & 0x3U;
    uint32_t otyper = (GPIOx->OTYPER >> pos) & 0x1U;
    if (moder == 0x0U) {
        config->mode = MY_GPIO_MODE_INPUT;
    } else if (moder == 0x1U) {
        config->mode = (otyper == 1U) ? MY_GPIO_MODE_OUTPUT_OD : MY_GPIO_MODE_OUTPUT_PP;
    } else if (moder == 0x3U) {
        config->mode = MY_GPIO_MODE_ANALOG;
    }

    /* Считывание PUPDR */
    config->pull = (MyGPIOPull)((GPIOx->PUPDR >> (pos * 2U)) & 0x3U);

    /* Считывание OSPEEDR */
    config->speed = (MyGPIOSpeed)((GPIOx->OSPEEDR >> (pos * 2U)) & 0x3U);

    /* Текущее состояние выхода */
    config->init_state = (uint8_t)((GPIOx->ODR >> pos) & 0x1U);
}

void my_gpio_write_pin(GPIO_TypeDef *GPIOx, uint16_t pin, uint8_t state)
{
    if (GPIOx == NULL) return;

    if (state) {
        GPIOx->BSRR = (uint32_t)pin;            /* Биты BS0..BS15 */
    } else {
        GPIOx->BSRR = ((uint32_t)pin << 16U);   /* Биты BR0..BR15 */
    }
}

uint8_t my_gpio_read_pin(GPIO_TypeDef *GPIOx, uint16_t pin)
{
    if (GPIOx == NULL) return 0;
    return ((GPIOx->IDR & pin) != 0U) ? 1U : 0U;
}

void my_gpio_toggle_pin(GPIO_TypeDef *GPIOx, uint16_t pin)
{
    if (GPIOx == NULL) return;
    uint32_t odr = GPIOx->ODR;
    GPIOx->BSRR = ((odr & pin) << 16U) | (~odr & pin);
}

void my_gpio_write_port(GPIO_TypeDef *GPIOx, uint16_t mask, uint16_t value)
{
    if (GPIOx == NULL) return;
    /* Атомарная одновременная запись нескольких битов порта без гонок */
    uint32_t set_bits = (uint32_t)(mask & value);
    uint32_t reset_bits = (uint32_t)(mask & (~value));
    GPIOx->BSRR = set_bits | (reset_bits << 16U);
}

uint16_t my_gpio_read_port(GPIO_TypeDef *GPIOx, uint16_t mask)
{
    if (GPIOx == NULL) return 0;
    return (uint16_t)(GPIOx->IDR & mask);
}
