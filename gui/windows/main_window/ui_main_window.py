#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 27/07/2021

"""

from gui.pages.ui_pages import Ui_AppPages
from qt_core import *
from gui.widgets.py_push_button import PyPushButton


class UI_MainWindow:
    def setupUi(self, parent, menu_width):
        if not parent.objectName():
            parent.setObjectName("MainWindow")

        parent.resize(1200, 720)
        parent.setMinimumSize(960, 540)

        # ========== ========== ========== ========== ========== ========== central frame

        self.frameCentral = QFrame()
        self.frameCentral.setStyleSheet("background-color: #282A36")

        self.lytMain = QHBoxLayout(self.frameCentral)
        self.lytMain.setContentsMargins(0, 0, 0, 0)
        self.lytMain.setSpacing(0)

        # ========== ========== ========== ========== ========== ==========  menu (left) bar

        self.frameMenu = QFrame()
        self.frameMenu.setStyleSheet("background-color: #44475A")
        self.frameMenu.setMinimumWidth(menu_width)
        self.frameMenu.setMaximumWidth(menu_width)

        self.lytMenu = QVBoxLayout(self.frameMenu)
        self.lytMenu.setContentsMargins(0, 0, 0, 0)
        self.lytMenu.setSpacing(0)

        # ---------- ---------- ---------- ---------- ---------- ---------- top menu

        self.frameMenuTop = QFrame()
        self.frameMenuTop.setMinimumHeight(40)
        self.frameMenuTop.setObjectName("frameMenuTop")

        self.btnToggle = PyPushButton(text="Ocultar Menu", icon_name="icon_menu.svg")

        self.btnHome = PyPushButton(
            text="Página Inicial", is_active=True, icon_name="icon_home.svg"
        )

        self.btnPage2 = PyPushButton(text="Página 2", icon_name="icon_widgets.svg")

        self.lytLeftMenuTop = QVBoxLayout(self.frameMenuTop)
        self.lytLeftMenuTop.setContentsMargins(0, 0, 0, 0)
        self.lytLeftMenuTop.setSpacing(0)

        self.lytLeftMenuTop.addWidget(self.btnToggle)
        self.lytLeftMenuTop.addWidget(self.btnHome)
        self.lytLeftMenuTop.addWidget(self.btnPage2)

        # ---------- ---------- ---------- ---------- ---------- ---------- spacer

        self.spacerLeftMenu = QSpacerItem(
            20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        # ---------- ---------- ---------- ---------- ---------- ---------- bottom menu

        self.frameMenuBottom = QFrame()
        self.frameMenuBottom.setMinimumHeight(40)
        self.frameMenuBottom.setObjectName("frameMenuBottom")

        self.lytLeftMenuBottom = QVBoxLayout(self.frameMenuBottom)
        self.lytLeftMenuBottom.setContentsMargins(0, 0, 0, 0)
        self.lytLeftMenuBottom.setSpacing(0)

        self.btnSettings = PyPushButton(text="Settings", icon_name="icon_settings.svg")

        self.lytLeftMenuBottom.addWidget(self.btnSettings)

        self.lblMenuBottom = QLabel("v1.0.0")
        self.lblMenuBottom.setAlignment(Qt.AlignCenter)
        self.lblMenuBottom.setMinimumHeight(30)
        self.lblMenuBottom.setMaximumHeight(30)
        self.lblMenuBottom.setStyleSheet("color: #C3CCDF")

        # ---------- ---------- ---------- ---------- ---------- ---------- add widgets to menu layout

        self.lytMenu.addWidget(self.frameMenuTop)
        self.lytMenu.addItem(self.spacerLeftMenu)
        self.lytMenu.addWidget(self.frameMenuBottom)
        self.lytMenu.addWidget(self.lblMenuBottom)

        # ========== ========== ========== ========== ========== ========== contents (middle) frame

        self.frameContent = QFrame()
        self.frameContent.setStyleSheet("background-color: #282A36")

        self.lytContent = QVBoxLayout(self.frameContent)
        self.lytContent.setContentsMargins(0, 0, 0, 0)
        self.lytContent.setSpacing(0)

        # ========== ========== ========== ========== ========== ========== top bar

        self.frameTop = QFrame()
        self.frameTop.setMinimumHeight(30)
        self.frameTop.setMaximumHeight(30)
        self.frameTop.setStyleSheet("background-color: #21232D;" "color: #6272A4")

        self.lytTopBar = QHBoxLayout(self.frameTop)
        self.lytTopBar.setContentsMargins(10, 0, 10, 0)

        self.lblLeft = QLabel("THE METRICKS KITCHEN")

        self.spacerTop = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.lblRight = QLabel("| INITIAL PAGE")
        self.lblRight.setStyleSheet("font: 700 9pt 'Segoe UI';")

        self.lytTopBar.addWidget(self.lblLeft)
        self.lytTopBar.addItem(self.spacerTop)
        self.lytTopBar.addWidget(self.lblRight)

        # ========== ========== ========== ========== ========== ========== pages

        self.stackedPages = QStackedWidget()
        self.stackedPages.setStyleSheet("font-size: 12pt;" "color: #F8F8F2")
        self.ui_pages = Ui_AppPages()
        self.ui_pages.setupUi(self.stackedPages)
        self.stackedPages.setCurrentWidget(self.ui_pages.pageHome)  # initial page

        # ========== ========== ========== ========== ========== ========== bottom bar

        self.frameBottom = QFrame()
        self.frameBottom.setMinimumHeight(30)
        self.frameBottom.setMaximumHeight(30)
        self.frameBottom.setStyleSheet("background-color: #21232D;" "color: #6272A4")

        self.lytBottomBar = QHBoxLayout(self.frameBottom)
        self.lytBottomBar.setContentsMargins(10, 0, 10, 0)

        self.lblLeft = QLabel("By Gustavo Rangel")

        self.spacerBottom = QSpacerItem(
            20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum
        )

        self.lblRight = QLabel("@ 2021")
        self.lblRight.setStyleSheet("font: 700 9pt 'Segoe UI';")

        self.lytBottomBar.addWidget(self.lblLeft)
        self.lytBottomBar.addItem(self.spacerBottom)
        self.lytBottomBar.addWidget(self.lblRight)

        # ========== ========== ========== ========== ========== ========== add widgets to contents layout

        self.lytContent.addWidget(self.frameTop)
        self.lytContent.addWidget(self.stackedPages)
        self.lytContent.addWidget(self.frameBottom)

        # ========== ========== ========== ========== ========== ========== add widgets to main laiyout

        self.lytMain.addWidget(self.frameMenu)
        self.lytMain.addWidget(self.frameContent)

        parent.setCentralWidget(self.frameCentral)
