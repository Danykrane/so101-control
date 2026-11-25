#!/usr/bin/env python3
"""Find the SO-101 port, ping motors 1-6, print calibration and positions.

Read-only: writes nothing to the motors, does not touch torque (disconnect(disable_torque=False)).
Exit code: 0 = 6/6 motors answered, 1 = port not found, 2 = bus or motor error.
"""

from __future__ import annotations

import argparse
import sys

from driver import DIAGNOSTIC_KEYS, find_ports, make_bus, probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. /dev/tty.usbmodemXXXX; auto-detected by default")
    args = ap.parse_args()

    ports = [args.port] if args.port else find_ports()
    if not ports:
        print("Robot not found: no /dev/tty.usbmodem*, /dev/tty.usbserial*, /dev/tty.wchusbserial* ports.")
        print("Check the USB cable and the motor board power supply.")
        return 1
    if len(ports) > 1:
        print("Several ports found, checking the first one (pick another with --port):")
        for p in ports:
            print("  ", p)
    port = ports[0]
    print(f"Port: {port}")

    try:
        bus = make_bus(port)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to open the port: {e}")
        return 2
    try:
        diag = probe(bus, port)
    except Exception as e:  # noqa: BLE001
        print(f"Bus read error: {e}")
        return 2
    finally:
        bus.disconnect(disable_torque=False)

    header = [title for _, title in DIAGNOSTIC_KEYS]
    rows = [["no answer" if k == "model" and r.get(k) is None else str(r.get(k, "")) for k, _ in DIAGNOSTIC_KEYS]
            for r in diag.rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    print()
    print(("OK: " if diag.ok else "ERROR: ") + diag.message)
    print("Nothing was written to the motors, torque unchanged.")
    return 0 if diag.ok else 2


if __name__ == "__main__":
    sys.exit(main())
