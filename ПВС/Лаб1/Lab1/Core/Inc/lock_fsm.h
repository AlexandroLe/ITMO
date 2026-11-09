/**
  ******************************************************************************
  * @file           : lock_fsm.h
  * @brief          : Логика прикладной задачи «Кодовый замок» (Вариант 5)
  ******************************************************************************
  */

#ifndef INC_LOCK_FSM_H_
#define INC_LOCK_FSM_H_

#include <stdint.h>
#include "button.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Инициализация конечного автомата замка */
void lock_fsm_init(void);

/* Неблокирующий шаг обновления логики замка */
void lock_fsm_update(ButtonEvent btn_event, uint32_t current_tick);

#ifdef __cplusplus
}
#endif

#endif /* INC_LOCK_FSM_H_ */
