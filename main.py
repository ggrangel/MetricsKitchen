#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 27/07/2021

"""

import sys

from gui.windows.main_window.ui_main_window import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.menu_width = 50

        self.setWindowTitle("'Metricks Kitchen")

        self.ui = UI_MainWindow()
        self.ui.setupUi(self, self.menu_width)

        self.ui.btnTemp1.clicked.connect(self.evt_toggle_button_clicked)

        self.show()

    def evt_toggle_button_clicked(self):
        current_menu_width = self.ui.frameMenu.width()

        if current_menu_width == self.menu_width:
            new_width = 240
        else:
            new_width = self.menu_width

        self.animation = QPropertyAnimation(self.ui.frameMenu, b"minimumWidth")
        self.animation.setStartValue(current_menu_width)
        self.animation.setEndValue(new_width)
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.InOutCirc)
        self.animation.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
