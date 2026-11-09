#ifndef INC_LOCK_FSM_H_
#define INC_LOCK_FSM_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Инициализация конечного автомата замка */
void lock_fsm_init(void);

/* Обработка входящего символа по UART */
void lock_fsm_process_char(char c);

/* Обновление логики (таймауты, анимации) */
void lock_fsm_update(uint32_t current_tick);

/* Сброс состояния замка (например, при смене режима прерываний) */
void lock_fsm_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* INC_LOCK_FSM_H_ */
