"""
SDK-1.1M Smart Lock Control & Telemetry Station
Лабораторная работа №3 (ПВС) — Вариант 5: «Кодовый замок с EEPROM и OLED-дисплеем»
Авторы: Ястребов-Амирханов Алекси, Исупов Никита Александрович
"""

import sys
import os
import time
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False


# =========================================================================
# ЦВЕТОВАЯ ПАЛИТРА ТЁМНОЙ ТЕМЫ (HARDWARE DASHBOARD STYLE)
# =========================================================================
THEME = {
    "bg_main":       "#131519",  # Главный тёмный фон
    "bg_card":       "#1c1f26",  # Фон карточек и панелей
    "bg_card_inner": "#242831",  # Внутренний фон полей ввода
    "border":        "#2e3440",  # Границы элементов
    "text_primary":  "#eceff4",  # Основной белый текст
    "text_muted":    "#8892b0",  # Приглушённый серый текст
    "accent_blue":   "#3b82f6",  # Основной акцентный синий
    "accent_cyan":   "#00d2ff",  # Неоновый голубой (стиль OLED)
    "state_locked":  "#dc2626",  # Красный (ЗАКРЫТО)
    "state_open":    "#10b981",  # Зелёный (ОТКРЫТО)
    "state_blocked": "#f59e0b",  # Янтарный (БЛОКИРОВКА)
    "btn_bg":        "#2a303c",  # Фон кнопок
    "btn_hover":     "#374151",  # Фон кнопок при наведении
    "term_bg":       "#0b0c0e",  # Фон терминала
    "term_fg":       "#cbd5e1",  # Текст терминала
}


class ModernButton(tk.Button):
    """Стилизованная кнопка с эффектом наведения (Hover)"""
    def __init__(self, master=None, bg_color=None, hover_color=None, fg_color="#ffffff", **kwargs):
        self.normal_bg = bg_color or THEME["btn_bg"]
        self.hover_bg = hover_color or THEME["btn_hover"]
        self.fg_color = fg_color

        super().__init__(
            master,
            bg=self.normal_bg,
            fg=self.fg_color,
            activebackground=self.hover_bg,
            activeforeground=self.fg_color,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            **kwargs
        )
        self.bind("<Enter>", lambda e: self.configure(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.configure(bg=self.normal_bg))


class SDKLockMonitorApp:
    def __init__(self, root, default_com=None):
        self.root = root
        self.root.title("SDK-1.1M | Станция управления кодовым замком (ЛР №3)")
        self.root.geometry("820x720")
        self.root.minsize(740, 640)
        self.root.configure(bg=THEME["bg_main"])

        self.serial_inst = None
        self.is_connected = False
        self.worker_thread = None
        self.default_com = default_com

        # Формирование имени лог-файла сессии
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filepath = f"lock_session_{timestamp_str}.log"

        self._build_interface()

        # Автоматическое подключение, если порт передан в аргументах
        if self.default_com:
            self.com_selector.set(self.default_com)
            self.connect_hardware()

    # =====================================================================
    # ПОСТРОЕНИЕ ПОЛЬЗОВАТЕЛЬСКОГО ИНТЕРФЕЙСА
    # =====================================================================
    def _build_interface(self):
        # 1. Верхняя панель: Статус соединения и выбор COM-порта
        top_frame = tk.Frame(self.root, bg=THEME["bg_card"], bd=0, padx=14, pady=10)
        top_frame.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            top_frame,
            text="Стенд SDK-1.1M (STM32F427):",
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))

        # Поиск доступных COM-портов
        port_list = []
        if HAS_PYSERIAL:
            port_list = [p.device for p in serial.tools.list_ports.comports()]

        self.com_selector = ttk.Combobox(top_frame, values=port_list, width=10, font=("Segoe UI", 9))
        if "COM8" in port_list:
            self.com_selector.set("COM8")
        elif port_list:
            self.com_selector.set(port_list[0])
        elif self.default_com:
            self.com_selector.set(self.default_com)
        else:
            self.com_selector.set("COM8")
        self.com_selector.pack(side=tk.LEFT, padx=4)

        self.btn_toggle_link = ModernButton(
            top_frame,
            text="Подключить",
            bg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self.toggle_connection
        )
        self.btn_toggle_link.pack(side=tk.LEFT, padx=8)

        self.btn_refresh_ports = ModernButton(
            top_frame,
            text="↻ Порты",
            bg_color=THEME["btn_bg"],
            hover_color=THEME["btn_hover"],
            command=self.refresh_available_ports
        )
        self.btn_refresh_ports.pack(side=tk.LEFT, padx=2)

        self.lbl_link_indicator = tk.Label(
            top_frame,
            text="● ОФЛАЙН",
            bg=THEME["bg_card"],
            fg="#ef4444",
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_link_indicator.pack(side=tk.RIGHT, padx=4)

        # 2. Центральная информационная панель (Табло + Виртуальный OLED)
        center_card = tk.Frame(self.root, bg=THEME["bg_card"], padx=16, pady=14)
        center_card.pack(fill=tk.X, padx=14, pady=6)

        # Главный статусный баннер
        self.lbl_lock_banner = tk.Label(
            center_card,
            text="🔒 ЗАМОК ЗАБЛОКИРОВАН (LOCKED)",
            bg=THEME["state_locked"],
            fg="#ffffff",
            font=("Segoe UI", 16, "bold"),
            pady=12
        )
        self.lbl_lock_banner.pack(fill=tk.X, pady=(0, 10))

        # Виртуальный OLED-дисплей
        oled_box = tk.Frame(center_card, bg="#05070a", bd=2, relief=tk.SOLID, padx=12, pady=8)
        oled_box.pack(fill=tk.X, pady=4)

        tk.Label(
            oled_box,
            text="── ТЕЛЕМЕТРИЯ OLED SSD1306 (128x64 I2C) ──",
            bg="#05070a",
            fg="#475569",
            font=("Consolas", 8, "bold")
        ).pack(anchor="center")

        self.lbl_virtual_oled = tk.Label(
            oled_box,
            text="DISPLAY: [ ОЖИДАНИЕ СВЯЗИ ]",
            bg="#05070a",
            fg=THEME["accent_cyan"],
            font=("Consolas", 14, "bold"),
            pady=6
        )
        self.lbl_virtual_oled.pack(anchor="center")

        # 3. Интерактивная зона управления (Две колонки: Ввод пароля и Смена в EEPROM)
        ctrl_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        ctrl_frame.pack(fill=tk.X, padx=14, pady=6)

        # Левая колонка: Авторизация (AUTH)
        auth_card = tk.Frame(ctrl_frame, bg=THEME["bg_card"], padx=14, pady=12)
        auth_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(
            auth_card,
            text="ВВОД КОДА ДОСТУПА (AUTH)",
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 8))

        entry_auth_row = tk.Frame(auth_card, bg=THEME["bg_card"])
        entry_auth_row.pack(fill=tk.X, pady=2)

        self.input_auth = tk.Entry(
            entry_auth_row,
            show="●",
            bg=THEME["bg_card_inner"],
            fg=THEME["text_primary"],
            insertbackground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 12),
            width=14
        )
        self.input_auth.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 6))
        self.input_auth.bind("<Return>", lambda e: self.submit_auth())

        self.btn_auth = ModernButton(
            entry_auth_row,
            text="Открыть ➔",
            bg_color="#059669",
            hover_color="#047857",
            command=self.submit_auth
        )
        self.btn_auth.pack(side=tk.RIGHT)

        # Виртуальный мини-кейпад для быстрого клика
        numpad_box = tk.Frame(auth_card, bg=THEME["bg_card"])
        numpad_box.pack(fill=tk.X, pady=(10, 0))

        btn_grid = [
            ("1", "2", "3"),
            ("4", "5", "6"),
            ("7", "8", "9"),
            ("C", "0", "OK")
        ]
        for r_idx, row_vals in enumerate(btn_grid):
            r_frame = tk.Frame(numpad_box, bg=THEME["bg_card"])
            r_frame.pack(fill=tk.X, pady=2)
            for val in row_vals:
                if val == "C":
                    cmd = self.clear_input
                    bg_c = "#475569"
                elif val == "OK":
                    cmd = self.submit_auth
                    bg_c = "#059669"
                else:
                    cmd = lambda v=val: self.append_digit(v)
                    bg_c = THEME["btn_bg"]

                b = ModernButton(r_frame, text=val, bg_color=bg_c, command=cmd)
                b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Правая колонка: Энергонезависимая память EEPROM (SET)
        eeprom_card = tk.Frame(ctrl_frame, bg=THEME["bg_card"], padx=14, pady=12)
        eeprom_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(
            eeprom_card,
            text="ПАМЯТЬ EEPROM (MCP79411 I2C)",
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            eeprom_card,
            text="Новый мастер-код сохраняется в чип\nи не сбрасывается при отключении питания:",
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            font=("Segoe UI", 9),
            justify=tk.LEFT
        ).pack(anchor="w", pady=(0, 8))

        self.input_new_pass = tk.Entry(
            eeprom_card,
            bg=THEME["bg_card_inner"],
            fg=THEME["text_primary"],
            insertbackground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 12),
            width=16
        )
        self.input_new_pass.pack(fill=tk.X, ipady=5, pady=(0, 8))

        self.btn_save_eeprom = ModernButton(
            eeprom_card,
            text="💾 Записать в память EEPROM",
            bg_color="#d97706",
            hover_color="#b45309",
            command=self.submit_save_eeprom
        )
        self.btn_save_eeprom.pack(fill=tk.X, pady=4)

        self.btn_get_status = ModernButton(
            eeprom_card,
            text="🔄 Опросить статус (GET_STATUS)",
            bg_color=THEME["btn_bg"],
            hover_color=THEME["btn_hover"],
            command=lambda: self.send_uart_payload("GET_STATUS")
        )
        self.btn_get_status.pack(fill=tk.X, pady=4)

        # 4. Нижняя панель: Монитор шины UART и лог-консоль
        console_card = tk.Frame(self.root, bg=THEME["bg_card"], padx=14, pady=10)
        console_card.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 12))

        header_con = tk.Frame(console_card, bg=THEME["bg_card"])
        header_con.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header_con,
            text=f"МОНИТОР ШИНЫ UART (Лог-файл: {self.log_filepath})",
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        btn_clear = ModernButton(
            header_con,
            text="Очистить экран",
            bg_color=THEME["btn_bg"],
            hover_color=THEME["btn_hover"],
            command=self.clear_console
        )
        btn_clear.pack(side=tk.RIGHT)

        self.console_box = scrolledtext.ScrolledText(
            console_card,
            bg=THEME["term_bg"],
            fg=THEME["term_fg"],
            insertbackground="white",
            relief=tk.FLAT,
            font=("Consolas", 9),
            padx=8,
            pady=8
        )
        self.console_box.pack(fill=tk.BOTH, expand=True)

        # Настройка цветовых тегов для логов
        self.console_box.tag_config("TAG_TX", foreground="#38bdf8")     # Сине-голубой для исходящих
        self.console_box.tag_config("TAG_RX_OK", foreground="#34d399")  # Зеленый для успехов
        self.console_box.tag_config("TAG_RX_ERR", foreground="#f87171") # Красный для ошибок
        self.console_box.tag_config("TAG_RX_WARN", foreground="#fbbf24")# Желтый для предупреждений
        self.console_box.tag_config("TAG_SYS", foreground="#94a3b8")    # Серый для системных
        self.console_box.tag_config("TAG_TIME", foreground="#64748b")   # Темно-серый для меток

    # =====================================================================
    # ЛОГИКА ВВОДА С КЕЙПАДА
    # =====================================================================
    def append_digit(self, char):
        cur = self.input_auth.get()
        if len(cur) < 16:
            self.input_auth.insert(tk.END, char)
            # Также шлем этот символ напрямую на стенд
            self.send_uart_payload(char, add_newline=False)

    def clear_input(self):
        self.input_auth.delete(0, tk.END)

    def refresh_available_ports(self):
        if HAS_PYSERIAL:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            self.com_selector["values"] = ports
            if ports:
                self.com_selector.set(ports[0])

    # =====================================================================
    # ЛОГИРОВАНИЕ И РАБОТА С ТЕРМИНАЛОМ
    # =====================================================================
    def log_message(self, direction, text):
        now_ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{now_ts}] [{direction}] {text}\n"

        tag = "TAG_SYS"
        if direction == "TX":
            tag = "TAG_TX"
        elif direction == "RX":
            if "OK" in text or "PONG" in text or "SAVED" in text:
                tag = "TAG_RX_OK"
            elif "ERR" in text or "WRONG" in text:
                tag = "TAG_RX_ERR"
            elif "BLOCK" in text:
                tag = "TAG_RX_WARN"

        self.console_box.insert(tk.END, f"[{now_ts}] ", "TAG_TIME")
        self.console_box.insert(tk.END, f"[{direction}] {text}\n", tag)
        self.console_box.see(tk.END)

        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def clear_console(self):
        self.console_box.delete("1.0", tk.END)

    # =====================================================================
    # СЕТЕВОЕ ВЗАИМОДЕЙСТВИЕ (UART / PYSERIAL)
    # =====================================================================
    def toggle_connection(self):
        if self.is_connected:
            self.disconnect_hardware()
        else:
            self.connect_hardware()

    def connect_hardware(self):
        if not HAS_PYSERIAL:
            messagebox.showerror(
                "Ошибка",
                "Модуль pyserial не установлен!\nВыполните в консоли: pip install pyserial"
            )
            return

        port = self.com_selector.get().strip()
        if not port:
            messagebox.showwarning("Предупреждение", "Выберите имя COM-порта!")
            return

        try:
            self.serial_inst = serial.Serial(port=port, baudrate=9600, timeout=0.1)
            self.is_connected = True
            self.btn_toggle_link.configure(text="Отключить", bg="#dc2626")
            self.lbl_link_indicator.configure(text=f"● ОНЛАЙН ({port})", fg="#10b981")
            self.log_message("SYS", f"Канал открыт: {port} @ 9600 8N1")

            self.worker_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self.worker_thread.start()

            # Тестовый пинг стенда при установлении связи
            self.send_uart_payload("PING")
        except Exception as ex:
            messagebox.showerror("Ошибка порта", f"Не удалось подключиться к {port}:\n{ex}")
            self.disconnect_hardware()

    def disconnect_hardware(self):
        self.is_connected = False
        if self.serial_inst and self.serial_inst.is_open:
            try:
                self.serial_inst.close()
            except Exception:
                pass
        self.serial_inst = None
        self.btn_toggle_link.configure(text="Подключить", bg="#2563eb")
        self.lbl_link_indicator.configure(text="● ОФЛАЙН", fg="#ef4444")
        self.log_message("SYS", "Связь разорвана")

    def send_uart_payload(self, text, add_newline=True):
        if not self.is_connected or not self.serial_inst:
            messagebox.showwarning("Внимание", "Стенд не подключен!")
            return

        msg = (text + "\r\n") if add_newline else text
        try:
            self.serial_inst.write(msg.encode("utf-8"))
            self.log_message("TX", text.strip())
        except Exception as e:
            self.log_message("ERR", f"Сбой отправки: {e}")

    def submit_auth(self):
        entered = self.input_auth.get().strip()
        if not entered:
            return
        self.send_uart_payload(f"AUTH:{entered}")
        self.input_auth.delete(0, tk.END)

    def submit_save_eeprom(self):
        new_val = self.input_new_pass.get().strip()
        if not new_val:
            messagebox.showwarning("Ошибка ввода", "Пароль не может быть пустым!")
            return
        if len(new_val) > 16:
            messagebox.showwarning("Ошибка ввода", "Максимальная длина мастер-пароля — 16 символов!")
            return

        if messagebox.askyesno("Запись в память EEPROM", f"Записать новый пароль '{new_val}' в микросхему MCP79411?"):
            self.send_uart_payload(f"SET:{new_val}")
            self.input_new_pass.delete(0, tk.END)

    # =====================================================================
    # ПОТОК ПРИЕМА ДАННЫХ И ДИСПЕТЧЕРИЗАЦИЯ
    # =====================================================================
    def _serial_reader_loop(self):
        accumulated = ""
        while self.is_connected and self.serial_inst and self.serial_inst.is_open:
            try:
                chunk = self.serial_inst.read(64).decode("utf-8", errors="ignore")
                if chunk:
                    accumulated += chunk
                    while "\n" in accumulated:
                        single_line, accumulated = accumulated.split("\n", 1)
                        clean_line = single_line.strip("\r").strip()
                        if clean_line:
                            self.root.after(0, self._handle_incoming_packet, clean_line)
            except Exception:
                break
            time.sleep(0.01)

    def _handle_incoming_packet(self, packet):
        self.log_message("RX", packet)

        if packet == "PONG":
            self.lbl_lock_banner.configure(
                text="📡 СВЯЗЬ СО СТЕНДОМ УСТАНОВЛЕНА",
                bg="#2563eb",
                fg="#ffffff"
            )
            self.send_uart_payload("GET_STATUS")

        elif packet.startswith("STATUS:"):
            code = packet.split(":", 1)[1]
            if code == "OK":
                self.lbl_lock_banner.configure(
                    text="🔓 ДОСТУП РАЗРЕШЁН (UNLOCKED)",
                    bg=THEME["state_open"],
                    fg="#ffffff"
                )
            elif code == "ERR":
                self.lbl_lock_banner.configure(
                    text="❌ НЕВЕРНЫЙ ПАРОЛЬ (WRONG PIN)",
                    bg=THEME["state_locked"],
                    fg="#ffffff"
                )
            elif code == "BLOCKED":
                self.lbl_lock_banner.configure(
                    text="⚠️ ЗАБЛОКИРОВАНО НА 5 СЕК (3 ОШИБКИ)",
                    bg=THEME["state_blocked"],
                    fg="#000000"
                )
            elif code == "LOCKED":
                self.lbl_lock_banner.configure(
                    text="🔒 ЗАМОК ЗАКРЫТ (LOCKED)",
                    bg="#334155",
                    fg="#ffffff"
                )
            elif code == "SAVED":
                self.lbl_lock_banner.configure(
                    text="💾 ПАРОЛЬ ЗАПИСАН В EEPROM",
                    bg="#0284c7",
                    fg="#ffffff"
                )
                messagebox.showinfo("Память обновлена", "Новый пароль успешно сохранён в микросхеме EEPROM!")

        elif packet.startswith("DISPLAY:"):
            txt = packet.split(":", 1)[1]
            self.lbl_virtual_oled.configure(text=f"DISPLAY: [ {txt} ]")


def main():
    port_arg = sys.argv[1] if len(sys.argv) > 1 else None
    root_window = tk.Tk()
    app = SDKLockMonitorApp(root_window, default_com=port_arg)
    root_window.mainloop()


if __name__ == "__main__":
    main()
