    .data
ONE:             .word  1
RESULT:          .word  0
AND_RES:         .word  0
INTEGER:         .word  0
INPUT_ADDR:      .word  0x80
OUTPUT_ADDR:     .word  0x84
ITER_NUM:        .word  32

    .text
_start:
    load_ind     INPUT_ADDR
    store_addr   INTEGER

loop:
    load_addr    ITER_NUM
    beqz         printResult

    load_addr    INTEGER
    and          ONE
    add          RESULT
    store_addr   RESULT

    load_addr    INTEGER
    shiftr       ONE
    store_addr   INTEGER

    load_addr    ITER_NUM
    sub          ONE
    store_addr   ITER_NUM

    jmp          loop

printResult:
    load_addr    RESULT
    store_ind    OUTPUT_ADDR
    halt