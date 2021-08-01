#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 27/07/2021

"""

import sys

from qt_core import *
from gui.windows.main_window.ui_main_window import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.menu_width = 50

        self.setWindowTitle("'Metricks Kitchen")

        self.ui = UI_MainWindow()
        self.ui.setupUi(self, self.menu_width)

        self.ui.btnToggle.clicked.connect(self.evt_btnToggle_clicked)

        self.ui.btnHome.clicked.connect(self.evt_btnHome_clicked)

        self.ui.btnPage2.clicked.connect(self.evt_btnPage2_clicked)

        self.ui.btnSettings.clicked.connect(self.evt_btnSettings_clicked)

        self.show()

    def reset_selection(self):
        for btn in self.ui.frameMenu.findChildren(QPushButton):
            try:
                btn.set_active(False)
            except AttributeError:  # if button is not a PyPushButton it will not have a set_active method
                pass

    def evt_btnHome_clicked(self):
        self.ui.stackedPages.setCurrentWidget(self.ui.ui_pages.pageHome)
        self.reset_selection()
        self.ui.btnHome.set_active(True)

    def evt_btnPage2_clicked(self):
        self.ui.stackedPages.setCurrentWidget(self.ui.ui_pages.page_2)
        self.reset_selection()
        self.ui.btnPage2.set_active(True)

    def evt_btnSettings_clicked(self):
        self.ui.stackedPages.setCurrentWidget(self.ui.ui_pages.pageSettings)
        self.reset_selection()
        self.ui.btnSettings.set_active(True)

    def evt_btnToggle_clicked(self):
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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
