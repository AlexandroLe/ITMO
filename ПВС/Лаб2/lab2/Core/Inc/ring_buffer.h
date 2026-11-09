#ifndef INC_RING_BUFFER_H_
#define INC_RING_BUFFER_H_

#include <stdint.h>
#include <stdbool.h>

#define RING_BUFFER_SIZE 128

typedef struct {
    uint8_t buffer[RING_BUFFER_SIZE];
    volatile uint16_t head;
    volatile uint16_t tail;
} RingBuffer;

static inline void ring_buffer_init(RingBuffer *rb) {
    rb->head = 0;
    rb->tail = 0;
}

static inline bool ring_buffer_is_empty(const RingBuffer *rb) {
    return (rb->head == rb->tail);
}

static inline bool ring_buffer_is_full(const RingBuffer *rb) {
    return ((uint16_t)((rb->head + 1) % RING_BUFFER_SIZE) == rb->tail);
}

static inline bool ring_buffer_put(RingBuffer *rb, uint8_t byte) {
    uint16_t next = (uint16_t)((rb->head + 1) % RING_BUFFER_SIZE);
    if (next == rb->tail) {
        return false; /* Буфер переполнен */
    }
    rb->buffer[rb->head] = byte;
    rb->head = next;
    return true;
}

static inline bool ring_buffer_get(RingBuffer *rb, uint8_t *byte) {
    if (rb->head == rb->tail) {
        return false; /* Буфер пуст */
    }
    *byte = rb->buffer[rb->tail];
    rb->tail = (uint16_t)((rb->tail + 1) % RING_BUFFER_SIZE);
    return true;
}

static inline uint16_t ring_buffer_count(const RingBuffer *rb) {
    if (rb->head >= rb->tail) {
        return (rb->head - rb->tail);
    }
    return (RING_BUFFER_SIZE - rb->tail + rb->head);
}

#endif /* INC_RING_BUFFER_H_ */
