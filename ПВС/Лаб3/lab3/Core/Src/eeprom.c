#include "eeprom.h"
#include <string.h>

static I2C_HandleTypeDef *s_hi2c = NULL;

void eeprom_init(I2C_HandleTypeDef *hi2c)
{
    s_hi2c = hi2c;
}

bool eeprom_write(uint8_t mem_addr, const uint8_t *data, uint16_t len)
{
    if (s_hi2c == NULL || data == NULL || len == 0) {
        return false;
    }

    for (uint16_t i = 0; i < len; i++) {
        HAL_StatusTypeDef status = HAL_I2C_Mem_Write(
            s_hi2c,
            EEPROM_I2C_ADDR,
            mem_addr + i,
            I2C_MEMADD_SIZE_8BIT,
            (uint8_t *)&data[i],
            1,
            50
        );

        if (status != HAL_OK) {
            return false;
        }

        /* Пауза на внутренний цикл записи EEPROM */
        HAL_Delay(6);
    }

    return true;
}

bool eeprom_read(uint8_t mem_addr, uint8_t *data, uint16_t len)
{
    if (s_hi2c == NULL || data == NULL || len == 0) {
        return false;
    }

    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(
        s_hi2c,
        EEPROM_I2C_ADDR,
        mem_addr,
        I2C_MEMADD_SIZE_8BIT,
        data,
        len,
        100
    );

    return (status == HAL_OK);
}

void eeprom_load_password(char *out_pass, const char *default_pass)
{
    if (out_pass == NULL) return;

    /* Считываем маркер длины и строку */
    uint8_t buf[EEPROM_PASS_MAX_LEN + 1] = {0};
    bool ok = eeprom_read(EEPROM_PASS_ADDR, buf, EEPROM_PASS_MAX_LEN);

    /* Ищем нулевой терминатор или проверяем валидность */
    bool valid = ok && (buf[0] >= 0x20 && buf[0] <= 0x7E);
    uint8_t len = 0;

    if (valid) {
        for (int i = 0; i < EEPROM_PASS_MAX_LEN; i++) {
            if (buf[i] == '\0') {
                break;
            }
            if (buf[i] < 0x20 || buf[i] > 0x7E) {
                valid = false;
                break;
            }
            len++;
        }
        if (len < 1) valid = false;
    }

    if (valid) {
        memcpy(out_pass, buf, len);
        out_pass[len] = '\0';
    } else {
        strncpy(out_pass, default_pass, EEPROM_PASS_MAX_LEN);
        out_pass[EEPROM_PASS_MAX_LEN] = '\0';
        eeprom_save_password(out_pass);
    }
}

bool eeprom_save_password(const char *new_pass)
{
    if (new_pass == NULL) return false;
    uint8_t len = (uint8_t)strlen(new_pass);
    if (len > EEPROM_PASS_MAX_LEN) len = EEPROM_PASS_MAX_LEN;

    /* Записываем строку вместе с нулевым символом конца строки */
    return eeprom_write(EEPROM_PASS_ADDR, (const uint8_t *)new_pass, len + 1);
}
