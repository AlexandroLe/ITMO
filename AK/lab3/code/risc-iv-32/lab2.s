    .data

INPUT_ADDR:     .word  0x80
OUTPUT_ADDR:    .word  0x84
ERROR:          .word  0xCCCCCCCC
ONE:            .word  1
MINUS_ONE:      .word  -1

    .text
.org             0x90

_start:
    lui      sp, %hi(0x1000)                
    addi     sp, sp, %lo(0x1000)
    lui      t1, %hi(ONE)
    addi     t1, t1, %lo(ONE)
    lw       t1, 0(t1)
    jal      ra, real_start
    jal      ra, print
    halt

real_start:
    addi	 sp, sp, -4
	sw		 ra, 0(sp)	
    lui      t0, %hi(INPUT_ADDR)
    addi     t0, t0, %lo(INPUT_ADDR)
    lw       t0, 0(t0)
    lw       a0, 0(t0)
    jal      ra, getting_results
    lw		 ra, 0(sp)
	addi     sp, sp, 4
	jr 		 ra 

print:
    lui      t0, %hi(OUTPUT_ADDR)
    addi     t0, t0, %lo(OUTPUT_ADDR)
    lw       t0, 0(t0)
    sw       a0, 0(t0)
    jr       ra


getting_results:
    ble      a0, zero, neg
    and      a1, a0, t1
    beqz     a1, even_number
    add      a0, a0, t1

even_number:
    srl      a1, a0, t1
    mul      a0, a1, a1
    bgt      a0, zero, return_from_getting_results
    lui      a0, %hi(ERROR)
    addi     a0, a0, %lo(ERROR)
    lw       a0, 0(a0)

return_from_getting_results:
    jr       ra

neg:
    lui      a0, %hi(MINUS_ONE)
    addi     a0, a0, %lo(MINUS_ONE)            
    lw       a0, 0(a0)
    jr       ra
