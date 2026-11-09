#ifndef INC_LOCK_FSM_H_
#define INC_LOCK_FSM_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Инициализация конечного автомата замка, чтение пароля из EEPROM, отрисовка начального экрана */
void lock_fsm_init(void);

/* Обработка входящего символа / строки команд по UART */
void lock_fsm_process_char(char c);

/* Обновление логики (таймауты, анимации, перерисовка OLED) */
void lock_fsm_update(uint32_t current_tick);

/* Сброс состояния замка */
void lock_fsm_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* INC_LOCK_FSM_H_ */
