#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 27/07/2021

"""

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import *

UI_PATH = Path(__file__).parents[1] / 'gui' / 'pages'
PYUIC_PATH = '/home/rangelgbr/.pyenv/shims/pyuic6'


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pyuic6 GUI")
        self.resize(800, 500)

        self.ledPyuic = QLineEdit(str(PYUIC_PATH))
        self.ledPyuic.setPlaceholderText("Location of pyuic6 file")
        self.ledPyuic.textChanged.connect(self.evt_change)

        self.ledUIFile = QLineEdit()
        self.ledUIFile.setPlaceholderText("Input .ui file")
        self.ledUIFile.textChanged.connect(self.evt_change)

        self.ledPyFile = QLineEdit()
        self.ledPyFile.setPlaceholderText("Output .py file")
        self.ledPyFile.textChanged.connect(self.evt_change)

        self.chkExecutable = QCheckBox("Make executable")
        self.chkExecutable.toggled.connect(self.evt_change)

        self.txtCommand = QPlainTextEdit()

        self.btnPyuic = QPushButton("...")
        self.btnPyuic.clicked.connect(self.evt_btnPyuic_clicked)

        self.btnUIFile = QPushButton("...")
        self.btnUIFile.clicked.connect(self.evt_btnUIFile_clicked)

        self.btnPyFile = QPushButton("...")
        self.btnPyFile.clicked.connect(self.evt_btnPyFile_clicked)

        self.btnExecute = QPushButton("Execute")
        self.btnExecute.clicked.connect(self.evt_btnExecute_clicked)

        self.lytMain = QVBoxLayout()
        self.lytPyuic = QHBoxLayout()
        self.lytUIFile = QHBoxLayout()
        self.lytPyFile = QHBoxLayout()
        self.lytCommand = QHBoxLayout()

        self.lytPyuic.addWidget(self.ledPyuic, 9)
        self.lytPyuic.addWidget(self.btnPyuic, 1)

        self.lytMain.addLayout(self.lytPyuic)

        self.lytUIFile.addWidget(self.ledUIFile, 9)
        self.lytUIFile.addWidget(self.btnUIFile, 1)

        self.lytMain.addLayout(self.lytUIFile)

        self.lytPyFile.addWidget(self.ledPyFile, 9)
        self.lytPyFile.addWidget(self.btnPyFile, 1)

        self.lytMain.addLayout(self.lytPyFile)
        self.lytMain.addWidget(self.chkExecutable)

        self.lytCommand.addWidget(self.txtCommand, 9)
        self.lytCommand.addWidget(self.btnExecute, 1)
        self.lytMain.addLayout(self.lytCommand)

        self.setLayout(self.lytMain)

    # ========== ========== ========== ========== ========== ========== event handlers

    def evt_btnPyuic_clicked(self):
        sFile, sExt = QFileDialog.getOpenFileName(self, 'Pyuic6', str(PYUIC_PATH), 'Executable Files (*)')

        if sFile:
            self.ledPyuic.setText(sFile)

    def evt_btnUIFile_clicked(self):
        print(str(UI_PATH))
        sFile, sExt = QFileDialog.getOpenFileName(self,
                                                  'User Interface File',
                                                  str(UI_PATH),
                                                  'User Interface Files (*.ui)',
                                                  None,
                                                  QFileDialog.DontUseNativeDialog)

        if sFile:
            self.ledUIFile.setText(sFile)
            bn = os.path.basename(sFile)
            self.ledPyFile.setText(str(UI_PATH / 'ui_pages.py'))

    def evt_btnPyFile_clicked(self):
        sFile, sExt = QFileDialog.getSaveFileName(self, "Python Module", str(UI_PATH), 'Python Files (*.py')

        if sFile:
            self.ledPyFile.setText(sFile)

    def evt_btnExecute_clicked(self):
        os.system(self.txtCommand.toPlainText())
        QMessageBox.information(self, 'Execute', f"""You have successfully 
                                                     {self.ledUIFile.text()} to {self.ledPyFile.text()}""")

    def evt_change(self):
        if self.chkExecutable.isChecked():
            chkEx = " -x"
        else:
            chkEx = ""

        self.txtCommand.setPlainText(
            f"{self.ledPyuic.text()}{chkEx} {self.ledUIFile.text()} -o {self.ledPyFile.text()}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec())
