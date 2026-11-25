#!/usr/bin/env python3
"""Найти порт SO-101, пропинговать моторы 1–6, напечатать калибровку и позиции.

Только чтение: ничего не пишет в моторы, момент не трогает (disconnect(disable_torque=False)).
Код возврата: 0 = 6/6 моторов ответили, 1 = порт не найден, 2 = ошибка шины или моторов.
"""

from __future__ import annotations

import argparse
import sys

from driver import DIAGNOSTIC_KEYS, find_ports, make_bus, probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="порт, например /dev/tty.usbmodem5A7A0185671; по умолчанию автопоиск")
    args = ap.parse_args()

    ports = [args.port] if args.port else find_ports()
    if not ports:
        print("Робот не найден: нет портов /dev/tty.usbmodem*, /dev/tty.usbserial*, /dev/tty.wchusbserial*.")
        print("Проверьте USB-кабель и питание платы моторов.")
        return 1
    if len(ports) > 1:
        print("Найдено несколько портов, проверяю первый (другой можно указать через --port):")
        for p in ports:
            print("  ", p)
    port = ports[0]
    print(f"Порт: {port}")

    try:
        bus = make_bus(port)
    except Exception as e:  # noqa: BLE001
        print(f"Не удалось открыть порт: {e}")
        return 2
    try:
        diag = probe(bus, port)
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка чтения шины: {e}")
        return 2
    finally:
        bus.disconnect(disable_torque=False)

    header = [title for _, title in DIAGNOSTIC_KEYS]
    rows = [["нет ответа" if k == "model" and r.get(k) is None else str(r.get(k, "")) for k, _ in DIAGNOSTIC_KEYS]
            for r in diag.rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    print()
    print(("OK: " if diag.ok else "ОШИБКА: ") + diag.message)
    print("В моторы ничего не записано, момент не менялся.")
    return 0 if diag.ok else 2


if __name__ == "__main__":
    sys.exit(main())
