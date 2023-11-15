from PyQt5.QtWidgets import QApplication
import pyqt_viewer
from PyQt5.QtGui import QIcon

def main():
    qapp = QApplication([])
    main_window = pyqt_viewer.MainWindow()
    qapp.setWindowIcon(QIcon('111.jpg'))
    # main_window.setWindowTitle("SMAL Model Viewer")
    main_window.setWindowTitle("Pig3D Model Viewer")

    main_window.show()
    qapp.exec_()

if __name__ == '__main__':
    main()
