import os
kicad_path = r"C:\Program Files\KiCad\10.0\share\kicad\symbols"
os.environ["KICAD9_SYMBOL_DIR"] = kicad_path
os.environ["KICAD_SYMBOL_DIR"] = kicad_path
from skidl import *

print("🔧 Creating DIP-14 components...")
modules = [Part("Interface_LineDriver", "DS8830", ref=f"U{i}") for i in range(1, 20)]  # 19 модулей

print(f"✅ Created {len(modules)} modules")

print("🔌 Creating PinSocket connector (1x12, pins 1-10 for signals, 11=VCC, 12=GND)...")
connector = Part(
    "Connector_Generic",
    "Conn_01x12",
    ref="P1",
    footprint="Connector_PinSocket_2.54mm:PinSocket_1x12_P2.54mm_Vertical",
)

# ------------------- НОВЫЕ ДАННЫЕ (40 цепей) -------------------
netlist_data = {
    1:  [[15,6], [18,9]],
    2:  [[5,4], [8,9], [9,4], [7,1], [2,11]],
    3:  [[16,12], [11,1], [16,8], [19,9], [2,3]],
    4:  [[15,3], [11,6], [8,13], [4,2], [3,1]],
    5:  [[14,3], [12,9], [17,12], [19,10]],
    6:  [[10,12], [12,4], [19,5], [5,5]],
    7:  [[16,5], [16,4], [7,12]],
    8:  [[18,10], [15,12]],
    9:  [[3,13], [1,4], [15,13], [19,1], [1,12]],
    10: [[6,5], [3,6], [16,9], [19,4], [3,9]],
    11: [[14,12], [16,10], [12,8], [19,11]],
    12: [[12,12], [4,1], [14,9]],
    13: [[19,8], [14,13], [5,8], [16,1], [12,3]],
    14: [[7,5], [19,2], [19,13], [11,10], [19,12]],
    15: [[19,6], [19,3], [18,11]],
    16: [[4,3], [3,2], [3,3], [16,2], [8,11]],
    17: [[6,3], [13,1], [17,4], [11,12], [1,2]],
    18: [[11,11], [18,8], [13,4], [12,1], [4,10]],
    19: [[13,8], [16,3], [15,5], [6,9], [18,3]],
    20: [[16,11], [7,6], [17,11]],
    21: [[10,5], [7,4], [14,2], [18,13], [10,2]],
    22: [[12,13], [2,8], [7,11]],
    23: [[17,1], [13,6], [14,5], [18,2]],
    24: [[13,11], [8,5]],
    25: [[14,8], [16,6], [3,12], [4,9]],
    26: [[17,10], [15,4], [5,2], [12,2], [17,6]],
    27: [[18,6], [3,5], [14,10], [8,2], [13,9]],
    28: [[6,2], [10,3]],
    29: [[14,6], [5,6]],
    30: [[4,12], [18,12], [6,12]],
    31: [[18,1], [18,5], [2,12], [2,1], [15,9]],
    32: [[4,4], [10,6], [5,11], [4,5]],
    33: [[1,13], [13,3], [7,3], [4,11], [16,13]],
    34: [[9,1], [4,13], [7,10], [18,4], [14,1]],
    35: [[1,3], [3,10], [4,6], [8,6], [11,4]],
    36: [[3,4], [14,4], [17,5]],
    37: [[6,1], [5,13]],
    38: [[10,9], [12,11], [8,1], [15,1], [9,5]],
    39: [[6,4], [17,9]],
    40: [[17,3], [3,8], [1,1]],
}
# ----------------------------------------------------------------

# Цепи, подключаемые к разъёму (в указанном порядке)
jst_connections = [18, 40, 7, 39, 19, 29, 25, 22, 27, 35]

print("⚡ Creating power nets (VCC on pin14, GND on pin7)...")

vcc = Net("VCC")
gnd = Net("GND")

for mod in modules:
    vcc += mod[14]   # вывод питания
    gnd += mod[7]    # вывод земли

vcc += connector[11]  # VCC на 11-й контакт разъёма
gnd += connector[12]  # GND на 12-й контакт разъёма

print(f"🕸️ Creating all {len(netlist_data)} signal nets...")

nets = {}
for net_num, connections in netlist_data.items():
    net = Net(f"net{net_num}")
    for mod, pin in connections:
        if 1 <= mod <= len(modules):
            net += modules[mod - 1][pin]
        else:
            print(f"⚠️ Warning: module {mod} does not exist (max {len(modules)})")
    nets[net_num] = net
    print(f"✅ net{net_num}: {len(net.pins)} connections")

print(f"\n🔌 Connecting signal nets to PinSocket (pins 1..10)...")
for i, net_num in enumerate(jst_connections):
    if net_num in nets:
        nets[net_num] += connector[i + 1]  # i+1 соответствует контакту разъёма
        print(f"✅ net{net_num} → PinSocket pin {i+1}")
    else:
        print(f"⚠️ net{net_num} not found in nets")

print(f"\n💾 Generating netlist...")
generate_netlist(file_="complete.net")

print(f"\n🎉 Done!")
print(f"  Modules: {len(modules)}")
print(f"  Signal nets: {len(nets)}")
print(f"  PinSocket connections: {len(jst_connections)} (pins 1-10)")
print(f"  Power: VCC on pin11, GND on pin12")
print(f"  File: complete.net")
print(f"\n💡 Import complete.net in Pcbnew: Tools → Load Netlist…")