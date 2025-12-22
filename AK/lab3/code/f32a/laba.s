    .data
len:             .byte  0
buf:             .byte  '_______________________________'
input_addr:      .word  0x80
output_addr:     .word  0x84
mask:            .byte  0, '___'
error:           .word  0xCCCCCCCC

    .text

    .org 0x90
_start:
    @p input_addr b!
    lit 1 a!
    @b
    dup
    lit -90 + -if bigFirstL
    @p mask +
    !+ len_inc

stop:
    loop
    halt

loop:
    a
    lit -33 + if err            \ размер
    @b
    dup
    lit -10 + if end            \ перенос строки
    dup
    lit -32 + if afterSpace     \ пробел
    dup
    lit -90 + -if stayL         \ проверяем что оно большое
    dup
    lit -65 + -if littL         \ проверяем число ли это вообще
    @p mask +
    !+ len_inc
    loop ;

littL:
    lit 32 +
    @p mask +
    !+ len_inc
    loop ;

afterSpace:
    @p mask +
    !+ len_inc
    @b
    dup
    lit -90 + -if bigL      \ проверяем маленькая ли буква
    @p mask +
    !+ len_inc
    loop ;

end:
    drop
    @p output_addr
    b!
    lit len a!
    @+ 
    lit 255 and
    print_cstr ;

end_print:
    drop
    ;

bigL:
    lit -32 +
    @p mask +
    !+ len_inc
    loop ;

bigFirstL:
    lit -32 +
    @p mask +
    !+ len_inc
    stop ;

stayL:
    @p mask +
    !+ len_inc
    loop ;

print_cstr:
    dup
    if end_print 
    lit -1 
    +
    @+ 
    lit 255 
    and
    !p 0x84
    print_cstr ;

err:
    @p error
    !p 0x84
    halt

len_inc:
    @p len 
    lit 1 
    +
    !p len
    ;