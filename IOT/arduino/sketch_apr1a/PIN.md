NodeMCU V3 Lolin

| Модуль   | Пин модуля | Куда на NodeMCU |   GPIO |
| -------------- | :-----------------: | --------------------: | -----: |
| PCF8574        |         VCC         |                  3.3V |      - |
| PCF8574        |         GND         |                   GND |      - |
| PCF8574        |         SDA         |                    D2 |  GPIO4 |
| PCF8574        |         SCL         |                    D3 |  GPIO0 |
| Зуммер   |         VCC         |                  3.3V |      - |
| Зуммер   |         GND         |                   GND |      - |
| Зуммер   |         I/O         |                    D1 |  GPIO5 |
| TM1638 LED&KEY |         VCC         |                  3.3V |      - |
| TM1638 LED&KEY |         GND         |                   GND |      - |
| TM1638 LED&KEY |         STB         |                    D5 | GPIO14 |
| TM1638 LED&KEY |         CLK         |                    D6 | GPIO12 |
| TM1638 LED&KEY |         DIO         |                    D7 | GPIO13 |

Клавиатура 3x4 к PCF8574T
Должна быть включена, то есть все ползунки на 1, 2, 3 (не на ON)

| Клавиатура | Куда на PCF8574 |
| -------------------- | --------------------: |
| R1 / Row 1           |                    P0 |
| R2 / Row 2           |                    P1 |
| R3 / Row 3           |                    P2 |
| R4 / Row 4           |                    P3 |
| C1 / Col 1           |                    P4 |
| C2 / Col 2           |                    P5 |
| C3 / Col 3           |                    P6 |
