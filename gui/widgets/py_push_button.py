#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 30/07/2021

"""

from pathlib import Path

from qt_core import *

GUI_PATH = Path(__file__).parents[1]


class PyPushButton(QPushButton):
    def __init__(
        self,
        text="",
        height=40,
        minimum_width=50,
        text_padding=55,
        text_color="#C3CCDF",
        icon_name="",
        icon_color="#C3CCDF",
        btn_color="#44475A",
        btn_hover="#4F5368",
        btn_pressed="#282A36FF",
        is_active=False,
    ):
        super().__init__()

        self.setText(text)
        self.setMaximumHeight(height)
        self.setMinimumHeight(height)
        self.setCursor(Qt.PointingHandCursor)

        self.minimum_width = minimum_width
        self.text_padding = text_padding
        self.text_color = text_color
        self.icon_name = icon_name
        self.icon_color = icon_color
        self.btn_color = btn_color
        self.btn_hover = btn_hover
        self.btn_pressed = btn_pressed
        self.is_active = is_active

        self.set_style()

    def set_style(self):
        default_style = f"""
        QPushButton {{
            color: {self.text_color};
            background-color: {self.btn_color};
            padding-left: {self.text_padding}px;
            text-align: left;
            border: none;
        }}
        QPushButton:hover {{
            background-color: {self.btn_hover};
        }}
        """

        active_style = f"""
        QPushButton {{
            background-color: {self.btn_hover};
            border-right: 5px solid #282A36;
        }}
        """

        style = default_style

        if self.is_active:
            style += active_style

        self.setStyleSheet(style)

    def paintEvent(self, event):
        QPushButton.paintEvent(self, event)

        self.qp = QPainter()
        self.qp.begin(self)
        self.qp.setRenderHint(QPainter.Antialiasing)
        self.qp.setPen(Qt.NoPen)

        self.rect = QRect(
            0, 0, self.minimum_width, self.height()
        )  # container for the icon

        self.draw_icon()

        self.qp.end()

    def draw_icon(self):
        icon_path = GUI_PATH / "images" / "icons" / f"{self.icon_name}"

        icon = QPixmap(str(icon_path))

        painter = QPainter(icon)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(icon.rect(), self.icon_color)

        self.qp.drawPixmap(
            (self.rect.width() - icon.width()) / 2,
            (self.rect.height() - icon.height()) / 2,
            icon,
        )

        painter.end()
