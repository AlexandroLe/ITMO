#ifndef INC_EEPROM_H_
#define INC_EEPROM_H_

#include "stm32f4xx_hal.h"
#include <stdint.h>
#include <stdbool.h>

/*
 * MCP79411 I2C EEPROM:
 * 7-битный адрес: 0x57 (8-битный адрес на шине: 0xAE для записи, 0xAF для чтения)
 * Объем EEPROM: 128 байт (адреса 0x00 .. 0x7F)
 */
#define EEPROM_I2C_ADDR       (0x57 << 1)  /* 0xAE */
#define EEPROM_PASS_ADDR      0x00         /* Адрес хранения пароля в EEPROM */
#define EEPROM_PASS_MAX_LEN   16           /* Максимальная длина пароля */

/* Инициализация модуля EEPROM */
void eeprom_init(I2C_HandleTypeDef *hi2c);

/* Запись блока данных в EEPROM */
bool eeprom_write(uint8_t mem_addr, const uint8_t *data, uint16_t len);

/* Чтение блока данных из EEPROM */
bool eeprom_read(uint8_t mem_addr, uint8_t *data, uint16_t len);

/* Загрузка мастер-пароля (если EEPROM не инициализирована, записывает default_pass) */
void eeprom_load_password(char *out_pass, const char *default_pass);

/* Сохранение нового мастер-пароля в EEPROM */
bool eeprom_save_password(const char *new_pass);

#endif /* INC_EEPROM_H_ */
