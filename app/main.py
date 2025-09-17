import sys
from PySide6 import QtWidgets
from ui.main_window import ImageQualityMVP

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ImageQualityMVP()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
