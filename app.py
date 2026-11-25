#!/usr/bin/env python3
"""SO-101: прототип ручного управления (этап 0). PyQt6 + pyqtgraph.opengl.

UI только отображает RobotState и шлет команды драйверу; кинематика и safety в своих модулях.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPalette, QPixmap, QVector3D
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cameras import Grabber, list_cameras
from driver import DIAGNOSTIC_KEYS, Diagnostics, FeetechDriver, RobotState, SimDriver, find_ports
from kinematics import JOINTS, JointMap, Robot, plan_cartesian
from remote import RemoteDriver, robotd_alive
from safety import Limits

LABELS = {
    "shoulder_pan": "1 Поворот базы",
    "shoulder_lift": "2 Плечо",
    "elbow_flex": "3 Локоть",
    "wrist_flex": "4 Кисть, наклон",
    "wrist_roll": "5 Кисть, вращение",
    "gripper": "6 Гриппер",
}
UNITS = ["°"] * 5 + ["%"]


class Bridge(QObject):
    """Колбэки драйвера (его поток) → Qt-сигналы (главный поток)."""

    state = pyqtSignal(object)
    log = pyqtSignal(str)
    diag = pyqtSignal(object)


class ArmView(gl.GLViewWidget):
    """3D-скелет: шарниры точками, звенья линиями, оси EE, полупрозрачный «призрак» цели."""

    def __init__(self, robot: Robot):
        super().__init__()
        self.robot = robot
        self.setBackgroundColor(QColor(24, 26, 30))
        self.setCameraPosition(pos=QVector3D(0.12, 0, 0.12), distance=0.7, elevation=22, azimuth=-55)
        grid = gl.GLGridItem(color=(255, 255, 255, 40))
        grid.setSize(0.8, 0.8)
        grid.setSpacing(0.05, 0.05)
        self.addItem(grid)
        base_axes = gl.GLAxisItem()
        base_axes.setSize(0.06, 0.06, 0.06)
        self.addItem(base_axes)
        opts = dict(antialias=True, glOptions="translucent")
        self.ghost_links = gl.GLLinePlotItem(width=3, color=(0.35, 0.7, 1.0, 0.35), mode="line_strip", **opts)
        self.ghost_jaw = gl.GLLinePlotItem(width=3, color=(0.35, 0.7, 1.0, 0.35), mode="lines", **opts)
        self.ghost_joints = gl.GLScatterPlotItem(size=9, color=(0.35, 0.7, 1.0, 0.35), pxMode=True)
        self.ghost_joints.setGLOptions("translucent")
        self.links = gl.GLLinePlotItem(width=5, color=(0.86, 0.87, 0.9, 1), mode="line_strip", **opts)
        self.jaw = gl.GLLinePlotItem(width=4, color=(0.95, 0.75, 0.3, 1), mode="lines", **opts)
        self.joints = gl.GLScatterPlotItem(size=13, color=(1.0, 0.55, 0.2, 1), pxMode=True)
        self.joints.setGLOptions("translucent")
        self.axes = gl.GLLinePlotItem(width=3, mode="lines", **opts)
        # Сначала рука, потом призрак: при совпадении призрак не проходит тест глубины и не закрывает руку.
        for item in (self.links, self.jaw, self.joints, self.axes, self.ghost_links, self.ghost_jaw, self.ghost_joints):
            self.addItem(item)
        self.banner = QLabel(self)  # «СИМУЛЯЦИЯ» или «момент выключен»: видно, почему робот стоит
        self.banner.move(12, 12)
        self.banner.hide()

    def set_banner(self, text: str, color: str) -> None:
        if not text:
            self.banner.hide()
            return
        if self.banner.text() != text:
            self.banner.setText(text)
            self.banner.setStyleSheet(f"background:{color}; color:#111; font-weight:bold; font-size:15px; padding:6px 10px; border-radius:4px;")
            self.banner.adjustSize()
        self.banner.show()

    def _pose(self, q):
        pts, jaw_tip = self.robot.skeleton(q)
        f = self.robot.frames(q)
        pivot = f["gripper"][:3, 3]
        return pts, np.array([pivot, jaw_tip]), f["ee"]

    def show_state(self, q: np.ndarray, ghost: np.ndarray | None) -> None:
        pts, jaw, ee = self._pose(q)
        self.links.setData(pos=pts)
        self.jaw.setData(pos=jaw)
        self.joints.setData(pos=pts[:-1])
        o, R = ee[:3, 3], ee[:3, :3]
        L = 0.04
        self.axes.setData(
            pos=np.array([o, o + R[:, 0] * L, o, o + R[:, 1] * L, o, o + R[:, 2] * L]),
            color=np.array([[1, 0.2, 0.2, 1]] * 2 + [[0.2, 1, 0.2, 1]] * 2 + [[0.3, 0.5, 1, 1]] * 2),
        )
        visible = ghost is not None
        for item in (self.ghost_links, self.ghost_jaw, self.ghost_joints):
            item.setVisible(visible)
        if visible:
            gp, gj, _ = self._pose(ghost)
            self.ghost_links.setData(pos=gp)
            self.ghost_jaw.setData(pos=gj)
            self.ghost_joints.setData(pos=gp[:-1])


class CameraPanel(QFrame):
    def __init__(self, title: str, slot: int, log, parent=None):
        super().__init__(parent)
        self.slot, self.log, self.worker = slot, log, None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(f"<b>{title}</b>")
        self.combo = QComboBox()
        self.button = QPushButton("Старт")
        self.button.clicked.connect(self.toggle)
        self.info = QLabel("—")
        self.view = QLabel("нет картинки")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(240, 135)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setStyleSheet("background:#111; color:#777;")
        top = QHBoxLayout()
        for w in (self.title, self.combo, self.button, self.info):
            top.addWidget(w)
        top.addStretch()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        self._pix: QPixmap | None = None
        self._seq = 0
        self.timer = QTimer(self, interval=33, timeout=self._poll)

    def fill(self, cams: list[dict]) -> bool:
        """Заполнить список. Встроенная FaceTime-камера видна, но выбрать ее нельзя."""
        self.combo.clear()
        external = [c for c in cams if not c["builtin"]]
        if not cams:
            for i in range(3):
                self.combo.addItem(f"индекс {i}", i)
        for c in cams:
            label = f"{c['index']}: {c['name']}" + (" (встроенная, не открываем)" if c["builtin"] else "")
            self.combo.addItem(label, c["index"])
            if c["builtin"]:
                self.combo.model().item(self.combo.count() - 1).setEnabled(False)
        if len(external) > self.slot:
            self.combo.setCurrentIndex(self.combo.findData(external[self.slot]["index"]))
            return True
        self.combo.setCurrentIndex(-1)
        return False

    def toggle(self):
        self.stop() if self.worker else self.start()

    def start(self):
        index = self.combo.currentData()
        if index is None:
            self.log("Камера: выберите устройство")
            return
        self.worker = Grabber(int(index))
        self.worker.start()
        self._seq = 0
        self.timer.start()
        self.button.setText("Стоп")
        self.info.setText("…")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self._poll()

    def _poll(self):
        """30 Гц: забрать последний кадр у потока камеры."""
        w = self.worker
        if w is None:
            return
        seq, frame = w.latest()
        if seq != self._seq and frame is not None:
            self._seq = seq
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, wd = rgb.shape[:2]
            img = QImage(rgb.data, wd, h, rgb.strides[0], QImage.Format.Format_RGB888)
            self._pix = QPixmap.fromImage(img)
            self.view.setPixmap(
                self._pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            self.info.setText(f"{w.fps:.0f} FPS, {wd}×{h}")
        if not w.is_alive():
            if w.error:
                self.log(w.error)
                self.info.setText(w.error)
            self.worker = None
            self.timer.stop()
            self.button.setText("Старт")


class MainWindow(QMainWindow):
    def __init__(self, mode: int = 0):
        super().__init__()
        self.setWindowTitle("SO-101: ручное управление (прототип)")
        self.robot = Robot()
        self.jmap = JointMap.load()
        self.urdf_limits = Limits(*self.jmap.motor_limits(self.robot))
        self.driver = None
        self.state: RobotState | None = None
        self.limits_shown = None
        self.bridge = Bridge()
        self.bridge.state.connect(self._on_state)
        self.bridge.log.connect(self.log)
        self.bridge.diag.connect(self._on_diag)

        self.view = ArmView(self.robot)
        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumBlockCount(2000)
        self.cams = [CameraPanel("Гриппер (wrist)", 0, self.log), CameraPanel("Сверху (front)", 1, self.log)]

        bottom = QSplitter(Qt.Orientation.Horizontal)
        for c in self.cams:
            bottom.addWidget(c)
        bottom.addWidget(self.log_box)
        bottom.setSizes([330, 330, 420])
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.view)
        left.addWidget(bottom)
        left.setSizes([560, 260])

        tabs = QTabWidget()
        tabs.addTab(self._control_panel(), "Управление")
        tabs.addTab(self._diag_panel(), "Диагностика")
        self.tabs = tabs
        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(tabs)
        main.setSizes([900, 470])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._toolbar())
        lay.addWidget(main, 1)
        self.setCentralWidget(central)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        for b in self.findChildren(QPushButton):  # пробел не должен «нажимать» кнопку в фокусе
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.timer = QTimer(self, interval=33, timeout=self._refresh)
        self.timer.start()
        self.refresh_ports()
        self.refresh_cameras(autostart=True)
        self.mode.setCurrentIndex(mode)
        self._update_buttons()
        self.connect_robot()  # подключение только читает, ничего не двигается

    # --- построение UI ---

    def _toolbar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItems(["Симуляция", "Робот (Feetech)", "robotd (общий с Claude)"])
        self.mode.currentIndexChanged.connect(self._update_buttons)
        self.port = QComboBox(editable=True)
        self.port.setMinimumWidth(230)
        self.port_refresh = QPushButton("⟳", clicked=self.refresh_ports)
        self.btn_connect = QPushButton("Подключить", clicked=self.connect_robot)
        self.btn_disconnect = QPushButton("Отключить", clicked=self.disconnect_robot)
        self.btn_torque_on = QPushButton("Включить момент", clicked=self.torque_on)
        self.btn_torque_off = QPushButton("Отключить момент", clicked=self.torque_off)
        self.btn_stop = QPushButton("СТОП (пробел)", clicked=self.stop)
        self.btn_stop.setStyleSheet("QPushButton{background:#b3261e;color:white;font-weight:bold;padding:4px 14px;}")
        self.speed = QDoubleSpinBox(minimum=2, maximum=60, value=20, singleStep=5, suffix=" °/с", decimals=0)
        self.speed.valueChanged.connect(self._send_speed)
        self.torque_label = QLabel()
        for w in (QLabel("Режим:"), self.mode, QLabel("Порт:"), self.port, self.port_refresh, self.btn_connect,
                  self.btn_disconnect, self._sep(), self.btn_torque_on, self.btn_torque_off, self.torque_label,
                  self._sep(), self.btn_stop, self._sep(), QLabel("Скорость:"), self.speed):
            h.addWidget(w)
        h.addStretch()
        return bar

    @staticmethod
    def _sep():
        s = QFrame()
        s.setFrameShape(QFrame.Shape.VLine)
        return s

    def _control_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        joints = QGroupBox("Суставы")
        g = QGridLayout(joints)
        steps = QHBoxLayout()
        steps.addWidget(QLabel("Шаг:"))
        self.step_group = QButtonGroup(self)
        for s in (1, 5, 10):
            rb = QRadioButton(f"{s}")
            rb.setChecked(s == 5)
            self.step_group.addButton(rb, s)
            steps.addWidget(rb)
        steps.addWidget(QLabel("° / %"))
        steps.addStretch()
        g.addLayout(steps, 0, 0, 1, 6)
        for c, t in enumerate(["", "Сейчас", "Цель", "", "", "Лимиты"]):
            g.addWidget(QLabel(f"<small>{t}</small>"), 1, c)
        self.rows = []
        for i, name in enumerate(JOINTS):
            r = 2 + 2 * i
            present, target, lim = QLabel("—"), QLabel("—"), QLabel("")
            present.setMinimumWidth(56)
            target.setStyleSheet("color:#7fb6ff")
            minus = QPushButton("−", clicked=lambda _, i=i: self.jog_joint(i, -1))
            plus = QPushButton("+", clicked=lambda _, i=i: self.jog_joint(i, +1))
            for b in (minus, plus):
                b.setFixedWidth(34)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda val, i=i: self._slider_moved(i, val))
            g.addWidget(QLabel(LABELS[name]), r, 0)
            g.addWidget(present, r, 1)
            g.addWidget(target, r, 2)
            g.addWidget(minus, r, 3)
            g.addWidget(plus, r, 4)
            g.addWidget(lim, r, 5)
            g.addWidget(slider, r + 1, 0, 1, 6)
            self.rows.append(dict(present=present, target=target, slider=slider, limits=lim))
        v.addWidget(joints)

        grip = QGroupBox("Гриппер")
        h = QHBoxLayout(grip)
        h.addWidget(QPushButton("Открыть", clicked=lambda: self.set_gripper(None, open_=True)))
        h.addWidget(QPushButton("Закрыть", clicked=lambda: self.set_gripper(None, open_=False)))
        h.addWidget(QPushButton("− шаг", clicked=lambda: self.jog_joint(5, -1)))
        h.addWidget(QPushButton("+ шаг", clicked=lambda: self.jog_joint(5, +1)))
        v.addWidget(grip)

        cart = QGroupBox("Декартово (IK по 5 суставам, оси базы)")
        g = QGridLayout(cart)
        self.cart_mm = QDoubleSpinBox(minimum=1, maximum=20, value=5, suffix=" мм", decimals=0)
        self.cart_deg = QDoubleSpinBox(minimum=1, maximum=15, value=5, suffix=" °", decimals=0)
        g.addWidget(QLabel("Шаг:"), 0, 0)
        g.addWidget(self.cart_mm, 0, 1)
        g.addWidget(self.cart_deg, 0, 2)
        for r, (axis, label) in enumerate([(0, "X"), (1, "Y"), (2, "Z"), (3, "Pitch")], start=1):
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(QPushButton(f"{label} −", clicked=lambda _, a=axis: self.jog_cartesian(a, -1)), r, 1)
            g.addWidget(QPushButton(f"{label} +", clicked=lambda _, a=axis: self.jog_cartesian(a, +1)), r, 2)
        self.ee_label = QLabel("—")
        self.ee_label.setTextFormat(Qt.TextFormat.RichText)
        g.addWidget(self.ee_label, 5, 0, 1, 3)
        v.addWidget(cart)
        v.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _diag_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.diag_msg = QLabel("Нет данных")
        self.diag_msg.setWordWrap(True)
        self.diag_table = QTableWidget(6, len(DIAGNOSTIC_KEYS))
        self.diag_table.setHorizontalHeaderLabels([c[1] for c in DIAGNOSTIC_KEYS])
        self.diag_table.verticalHeader().setVisible(False)
        self.diag_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(self.diag_msg)
        v.addWidget(self.diag_table, 1)
        cams = QPushButton("Обновить список камер", clicked=lambda: self.refresh_cameras(False))
        v.addWidget(cams)
        return w

    # --- лог и статус ---

    def log(self, msg: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {msg}")

    def refresh_ports(self):
        ports = find_ports()
        current = self.port.currentText()
        self.port.clear()
        self.port.addItems(ports)
        if current:
            self.port.setCurrentText(current)
        self.log("Порты: " + (", ".join(ports) if ports else "робот не найден (нет /dev/tty.usbmodem*)"))

    def refresh_cameras(self, autostart: bool):
        cams = list_cameras()
        names = ", ".join(f"{c['index']}={c['name']}" for c in cams) or "не удалось получить список"
        self.log(f"Камеры: {names}")
        for panel in self.cams:
            panel.stop()
            if panel.fill(cams) and autostart:
                panel.start()

    def _update_buttons(self):
        connected = bool(self.state and self.state.connected and self.driver and self.driver.running)
        busy = bool(self.driver and self.driver.running)
        real = self.mode.currentIndex() == 1
        self.mode.setEnabled(not busy)
        self.port.setEnabled(real and not busy)
        self.port_refresh.setEnabled(real and not busy)
        self.btn_connect.setEnabled(not busy)
        self.btn_disconnect.setEnabled(busy)
        torque = bool(connected and self.state.torque_on)
        self.btn_torque_on.setEnabled(connected and not torque)
        self.btn_torque_off.setEnabled(connected and bool(self.state.torque.any()))
        self.torque_label.setText(
            "<b style='color:#e5484d'>момент ВКЛ</b>" if torque else
            ("<b style='color:#f5a524'>момент частично</b>" if connected and self.state.torque.any() else "момент выкл")
        )

    # --- подключение ---

    def connect_robot(self):
        if self.driver and self.driver.running:
            return
        cb = dict(on_state=self.bridge.state.emit, on_log=self.bridge.log.emit, on_diag=self.bridge.diag.emit)
        if self.mode.currentIndex() == 0:
            self.driver = SimDriver(self.urdf_limits, **cb)
        elif self.mode.currentIndex() == 2:  # порт держит robotd, окно только клиент
            self.driver = RemoteDriver(**cb)
        else:
            port = self.port.currentText().strip()
            if not port:
                self.log("Порт не выбран: подключите робота и нажмите ⟳")
                return
            if robotd_alive():
                self.log("Порт робота держит robotd: подключаюсь через него (режим «robotd»)")
                self.mode.setCurrentIndex(2)
                self.driver = RemoteDriver(**cb)
            else:
                self.log(f"Подключение к {port}: только чтение (ping, калибровка из EEPROM, позиции)")
                self.driver = FeetechDriver(port, self.urdf_limits, **cb)
        self.limits_shown = None
        self.driver.start()
        if self.driver.mode != "remote":  # скорость robotd задает тот, кто им управляет
            self._send_speed()
        self._update_buttons()

    def _ask_hold_or_release(self) -> bool | None:
        """True = отпустить, False = удерживать, None = отмена. Спрашиваем, только если есть что решать."""
        s = self.state
        if not (self.driver and self.driver.running and s and s.connected and s.torque.any()):
            return False
        if self.driver.mode != "real":  # sim не жалко, robotd при отключении окна продолжает держать робота
            return False
        box = QMessageBox(self)
        box.setWindowTitle("Момент включен")
        box.setText("Момент включен. Что сделать при отключении?")
        box.setInformativeText("«Удерживать»: моторы продолжат держать позу. «Отпустить»: рука упадет под своим весом, поддержите ее.")
        hold = box.addButton("Удерживать", QMessageBox.ButtonRole.AcceptRole)
        release = box.addButton("Отпустить (рука упадет)", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(hold)
        box.exec()
        if box.clickedButton() is hold:
            return False
        if box.clickedButton() is release:
            return True
        return None

    def disconnect_robot(self) -> bool:
        release = self._ask_hold_or_release()
        if release is None:
            return False
        if self.driver:
            self.driver.close(release=release)
        self._update_buttons()
        return True

    # --- команды ---

    def _ready(self) -> bool:
        s = self.state
        if not (self.driver and self.driver.running and s and s.connected):
            self.log("Робот не подключен")
            return False
        if not s.torque_on:
            ok = QMessageBox.question(
                self, "Момент выключен",
                "Моторы без момента, рука не двигается. Включить момент? Рука сначала замрет в текущей позе, "
                "потом выполнит команду.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return False
            self.driver.enable_torque()  # очередь драйвера: сначала момент, потом команда
        return True

    def _send_speed(self):
        if self.driver:
            v = self.speed.value()
            self.driver.set_speed(v, v * 2)

    def torque_on(self):
        if self.driver:
            self.driver.enable_torque()

    def torque_off(self):
        if not self.driver:
            return
        if self.driver.mode in ("real", "remote"):
            ok = QMessageBox.warning(
                self, "Отключить момент",
                "Рука упадет под своим весом. Поддержите ее рукой. Отключить момент?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        self.driver.disable_torque()

    def stop(self):
        if self.driver and self.driver.running:
            self.driver.hold()

    def jog_joint(self, i: int, sign: int):
        if self._ready():
            d = np.zeros(6)
            d[i] = sign * self.step_group.checkedId()
            self.driver.jog(d)

    def set_gripper(self, value, open_: bool):
        if self._ready():
            lim = self.state.limits
            t = np.full(6, np.nan)
            t[5] = lim.upper[5] if open_ else lim.lower[5]
            self.driver.set_target(t)

    def _slider_moved(self, i: int, val: int):
        if self.rows[i]["slider"].isSliderDown() and self.state and self.state.torque_on:
            t = np.full(6, np.nan)
            t[i] = val / 10.0
            self.driver.set_target(t)

    def jog_cartesian(self, axis: int, sign: int):
        if not self._ready():
            return
        s = self.state
        goal = self.robot.task(self.jmap.to_urdf(s.target)[:5])
        if axis < 3:
            goal[axis] += sign * self.cart_mm.value() / 1000.0
        else:
            goal[3] += sign * math.radians(self.cart_deg.value())
        t, msg = plan_cartesian(self.robot, self.jmap, s.target, s.limits, goal)
        if t is None:
            self.log(msg + ", не выполняю")
            return
        self.driver.set_target(t)

    # --- состояние ---

    def _on_state(self, s: RobotState):
        self.state = s

    def _on_diag(self, d: Diagnostics):
        self.diag_msg.setText(f"<b>{d.port}</b>: {'OK' if d.ok else 'ОШИБКА'}, {d.message}")
        self.diag_table.clearContents()
        for r, row in enumerate(d.rows):
            for c, (key, _) in enumerate(DIAGNOSTIC_KEYS):
                val = row.get(key, "")
                self.diag_table.setItem(r, c, QTableWidgetItem("нет ответа" if key == "model" and val is None else str(val)))
        self.log(f"Диагностика {d.port}: {d.message}")
        if not d.ok:
            self.tabs.setCurrentIndex(1)

    def _refresh(self):
        s = self.state
        self._update_buttons()
        if s is None:
            return
        if self.limits_shown is not s.limits:
            self.limits_shown = s.limits
            for i, row in enumerate(self.rows):
                sl = row["slider"]
                sl.blockSignals(True)
                sl.setRange(int(math.floor(s.limits.lower[i] * 10)), int(math.ceil(s.limits.upper[i] * 10)))
                sl.blockSignals(False)
                row["limits"].setText(f"<small>[{s.limits.lower[i]:.0f}, {s.limits.upper[i]:.0f}]</small>")
        for i, row in enumerate(self.rows):
            row["present"].setText(f"{s.present[i]:.1f}{UNITS[i]}")
            row["target"].setText(f"{s.target[i]:.1f}{UNITS[i]}")
            sl = row["slider"]
            if not sl.isSliderDown():
                sl.blockSignals(True)
                sl.setValue(int(round(s.target[i] * 10)))
                sl.blockSignals(False)
        q = self.jmap.to_urdf(s.present)
        qt = self.jmap.to_urdf(s.target)
        ghost = qt if np.max(np.abs(s.target - s.present)) > 0.5 else None
        self.view.show_state(q, ghost)
        p, pt = self.robot.task(q[:5]), self.robot.task(qt[:5])
        self.ee_label.setText(
            "<small>EE сейчас: X {:.0f} Y {:.0f} Z {:.0f} мм, pitch {:.0f}°, roll {:.0f}°<br>"
            "<span style='color:#7fb6ff'>EE цель: X {:.0f} Y {:.0f} Z {:.0f} мм, pitch {:.0f}°</span></small>".format(
                *(p[:3] * 1000), math.degrees(p[3]), math.degrees(p[4]), *(pt[:3] * 1000), math.degrees(pt[3])
            )
        )
        if s.mode == "sim":
            self.view.set_banner("СИМУЛЯЦИЯ: настоящий робот не двигается. Режим выбирается вверху слева", "#f5a524")
        elif not s.connected:
            self.view.set_banner("Робот не подключен: см. лог", "#e5484d")
        elif not s.torque_on:
            self.view.set_banner("Момент выключен: нажмите «Включить момент», иначе рука не двигается", "#f5a524")
        else:
            self.view.set_banner("", "")
        mode = {"sim": "СИМУЛЯЦИЯ", "real": "РОБОТ", "remote": "ROBOTD"}.get(s.mode, s.mode)
        port = getattr(self.driver, "port", "") if self.driver else ""
        conn = "подключен" if s.connected else "отключен"
        temp = np.nanmax(s.temperature) if np.isfinite(s.temperature).any() else float("nan")
        load = np.nanmax(np.abs(s.load)) if np.isfinite(s.load).any() else float("nan")
        self.status.setText(
            f"{mode} {port}  |  {conn}  |  {s.loop_hz:.0f} Гц  |  {'момент ВКЛ' if s.torque_on else 'момент выкл'}"
            f"  |  {s.voltage:.1f} В, макс {temp:.0f} °C, нагрузка {load:.0f}"
            + (f"  |  {s.error}" if s.error else "")
        )

    # --- клавиши и закрытие ---

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and ev.key() == Qt.Key.Key_Space:
            if ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat():
                self.stop()
            ev.accept()
            return True
        return super().eventFilter(obj, ev)

    def closeEvent(self, ev):
        if not self.disconnect_robot():
            ev.ignore()
            return
        for c in self.cams:
            c.stop()
        ev.accept()


def dark(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, "#1f2125"), (QPalette.ColorRole.WindowText, "#e6e6e6"),
        (QPalette.ColorRole.Base, "#16181b"), (QPalette.ColorRole.AlternateBase, "#24272c"),
        (QPalette.ColorRole.Text, "#e6e6e6"), (QPalette.ColorRole.Button, "#2b2e34"),
        (QPalette.ColorRole.ButtonText, "#e6e6e6"), (QPalette.ColorRole.Highlight, "#3d7bfd"),
        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#2b2e34"),
        (QPalette.ColorRole.ToolTipText, "#e6e6e6"),
    ]:
        p.setColor(role, QColor(color))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6b6f76"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6b6f76"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6b6f76"))
    app.setPalette(p)


def run_demo(win: MainWindow, shot: str | None):
    """Только для sim: включить момент, подвигать суставы и IK, снять окно в файл."""
    steps = [
        (300, lambda: win.torque_on()),
        (600, lambda: win.driver.jog([20, 15, -10, 0, 0, 30])),
        (2600, lambda: win.jog_cartesian(2, +1)),
        (2700, lambda: win.driver.jog([0, 0, 0, 25, 0, 0])),
    ]
    for ms, fn in steps:
        QTimer.singleShot(ms, fn)
    if shot:
        def grab():
            win.grab().save(shot)
            print(f"screenshot: {shot}", flush=True)
            QApplication.instance().quit()
        QTimer.singleShot(3200, grab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="sim: подвигать руку для проверки")
    ap.add_argument("--screenshot", help="sim-демо, сохранить снимок окна и выйти")
    ap.add_argument("--robotd", action="store_true", help="сразу подключиться к robotd (общий робот с Claude)")
    ap.add_argument("--sim", action="store_true", help="симуляция, даже если робот подключен")
    args = ap.parse_args()
    # Режим при запуске: robotd → через него, робот на USB → напрямую, иначе симуляция.
    # Подключение только читает, поэтому выбрать реальный режим автоматически безопасно.
    if args.sim or args.demo or args.screenshot:
        mode = 0
    elif args.robotd or robotd_alive():
        mode = 2
    else:
        mode = 1 if find_ports() else 0
    app = QApplication(sys.argv)
    app.setApplicationName("SO-101 Control")
    dark(app)
    win = MainWindow(mode=mode)
    app.installEventFilter(win)
    win.resize(1440, 900)
    win.show()
    if args.demo or args.screenshot:
        run_demo(win, args.screenshot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
