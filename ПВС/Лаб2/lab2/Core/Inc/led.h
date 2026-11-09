
#ifndef INC_LED_H_
#define INC_LED_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LED_GREEN,
    LED_YELLOW,
    LED_RED,
    LED_ALL
} LedId;

typedef enum {
    LED_STATE_OFF = 0,
    LED_STATE_ON  = 1
} LedState;

void led_init(void);

void led_set(LedId led, LedState state);

void led_trigger_flash(LedId led, uint32_t duration_ms);

void led_start_blink(LedId led, uint32_t toggle_period_ms, uint32_t total_duration_ms);

bool led_is_blinking(void);

void led_stop_effects(void);

void led_update(uint32_t current_tick);

#ifdef __cplusplus
}
#endif

#endif /* INC_LED_H_ */
