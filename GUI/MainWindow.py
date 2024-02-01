# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'v0.55.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import qdarktheme
from PyQt5 import QtCore, QtGui, QtWidgets
from GUI.GameData import GameData
from urllib import request
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QSize

# create a positive/negative bar
class LineWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.green_percentage = 50  # Initial percentage of green
        self.max_height = 3

        # Set vertical size policy to Expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_green_percentage(self, percentage):
        self.green_percentage = percentage
        self.update()

    def sizeHint(self):
        return QSize(self.width(), self.max_height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width, height = self.width(), self.height()

        # Calculate the position to split green and red
        split_position = int(width * (self.green_percentage / 100))

        # Draw the green part
        painter.fillRect(0, 0, split_position, height, Qt.green)

        # Draw the red part
        painter.fillRect(split_position, 0, width - split_position, height, Qt.red)

# create a combobox composed by check items
class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super(CheckableComboBox, self).__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)

    def getSelectedItems(self):
        res = []
        for i in range(self.count()):
            if self.model().item(i, 0).checkState() == Qt.Checked:
                res.append(i)
        return res

# main window
class Ui_MainWindow(object):
    def __init__(self, searcher):
        self.searcher = searcher
        self.games = []
        self.fieldsDict = {0: 'name', 1: 'description', 2: 'developer', 3: 'publisher', 4: 'platforms', 5: 'cgt'}

    def setupUi(self, MainWindow):
        data_font = QtGui.QFont("Arial", 14)

        MainWindow.setObjectName("Steam Search Engine")
        MainWindow.resize(800, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ResultList = QtWidgets.QListView(self.centralwidget)
        self.ResultList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.ResultList.setObjectName("ResultList")
        self.ResultList.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ResultListModel = QtGui.QStandardItemModel()
        self.ResultList.setModel(self.ResultListModel)
        self.ResultList.selectionModel().selectionChanged.connect(self.handleResultSelectionChange)
        self.ResultList.setFont(data_font)
        self.horizontalLayout.addWidget(self.ResultList)

        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 400))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 561, 757))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.TitleLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.TitleLabel.setFont(QtGui.QFont("Arial", 18))
        self.TitleLabel.setObjectName("TitleLabel")
        self.TitleLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.horizontalLayout_4.addWidget(self.TitleLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.AppidLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.AppidLabel.setFont(font)
        self.AppidLabel.setObjectName("AppidLabel")
        self.AppidLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.horizontalLayout_4.addWidget(self.AppidLabel)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.Image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.Image.setObjectName("Image")
        self.Image.setScaledContents(True)
        self.horizontalLayout_5.addWidget(self.Image)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 1, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")

        self.DescriptionLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.DescriptionLabel.setObjectName("DescriptionLabel")
        self.DescriptionLabel.setWordWrap(True)
        self.DescriptionLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.DescriptionLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.DescriptionLabel)

        self.SentimentLineLabel = LineWidget()
        #self.SentimentLineLabel.setObjectName("SentimentLineLabel")
        self.verticalLayout.addWidget(self.SentimentLineLabel, stretch=1)

        self.DeveloperLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.DeveloperLabel.setObjectName("DeveloperLabel")
        self.DeveloperLabel.setWordWrap(True)
        self.DeveloperLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.DeveloperLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.DeveloperLabel)
        self.PublisherLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.PublisherLabel.setObjectName("PublisherLabel")
        self.PublisherLabel.setWordWrap(True)
        self.PublisherLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.PublisherLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.PublisherLabel)
        self.ReleaseLable = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.ReleaseLable.setObjectName("ReleaseLable")
        self.ReleaseLable.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.ReleaseLable.setFont(data_font)
        self.verticalLayout.addWidget(self.ReleaseLable)
        self.PriceLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.PriceLabel.setObjectName("PriceLabel")
        self.PriceLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.PriceLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.PriceLabel)
        self.GenresLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.GenresLabel.setObjectName("GenresLabel")
        self.GenresLabel.setWordWrap(True)
        self.GenresLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.GenresLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.GenresLabel)
        self.TagsLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.TagsLabel.setObjectName("TagsLabel")
        self.TagsLabel.setWordWrap(True)
        self.TagsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.TagsLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.TagsLabel)
        self.CategoriesLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.CategoriesLabel.setObjectName("CategoriesLabel")
        self.CategoriesLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.CategoriesLabel.setWordWrap(True)
        self.CategoriesLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.CategoriesLabel)
        self.MinReqsLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.MinReqsLabel.setObjectName("MinReqsLabel")
        self.MinReqsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.MinReqsLabel.setWordWrap(True)
        self.MinReqsLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.MinReqsLabel)
        self.RecReqsLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.RecReqsLabel.setObjectName("RecReqsLabel")
        self.RecReqsLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.RecReqsLabel.setWordWrap(True)
        self.RecReqsLabel.setFont(data_font)
        self.verticalLayout.addWidget(self.RecReqsLabel)
        self.ReviewsView = QtWidgets.QListView(self.scrollAreaWidgetContents)
        self.ReviewsView.setObjectName("ReviewsView")
        self.ReviewsView.setFont(data_font)
        self.verticalLayout.addWidget(self.ReviewsView)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(6, 1)
        self.verticalLayout.setStretch(7, 1)
        self.verticalLayout.setStretch(8, 1)
        self.verticalLayout.setStretch(9, 1)
        self.verticalLayout.setStretch(10, 1)
        self.verticalLayout.setStretch(11, 20)
        self.gridLayout_2.addLayout(self.verticalLayout, 2, 0, 1, 1)
        self.gridLayout_2.setRowMinimumHeight(0, 1)
        self.gridLayout_2.setRowMinimumHeight(1, 250)
        self.gridLayout_2.setRowMinimumHeight(2, 450)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.SearchField = QtWidgets.QLineEdit(self.centralwidget)
        self.SearchField.setObjectName("SearchField")
        self.SearchField.setFont(data_font)
        self.SearchField.returnPressed.connect(self.execSearch)
        self.horizontalLayout_2.addWidget(self.SearchField)
        self.SearchButton = QtWidgets.QPushButton(self.centralwidget)
        self.SearchButton.setObjectName("SearchButton")
        self.SearchButton.clicked.connect(self.execSearch)
        self.SearchButton.setFont(data_font)
        self.horizontalLayout_2.addWidget(self.SearchButton)

        self.ResultsLimitSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.ResultsLimitSpinBox.setMinimum(1)
        self.ResultsLimitSpinBox.setMaximum(100)
        self.ResultsLimitSpinBox.setProperty("value", 20)
        self.ResultsLimitSpinBox.setObjectName("ResultsLimitSpinBox")
        self.ResultsLimitSpinBox.setFont(data_font)
        self.horizontalLayout_2.addWidget(self.ResultsLimitSpinBox)

        self.ComboFieldsBox = CheckableComboBox()
        self.ComboFieldsBox.addItem("Title")
        self.ComboFieldsBox.model().item(0, 0).setCheckState(QtCore.Qt.Checked)
        self.ComboFieldsBox.addItem("Description")
        self.ComboFieldsBox.model().item(1, 0).setCheckState(QtCore.Qt.Unchecked)
        self.ComboFieldsBox.addItem("Developer")
        self.ComboFieldsBox.model().item(2, 0).setCheckState(QtCore.Qt.Unchecked)
        self.ComboFieldsBox.addItem("Publisher")
        self.ComboFieldsBox.model().item(3, 0).setCheckState(QtCore.Qt.Unchecked)
        self.ComboFieldsBox.addItem("Platforms")
        self.ComboFieldsBox.model().item(4, 0).setCheckState(QtCore.Qt.Unchecked)
        self.ComboFieldsBox.addItem("Categories,Genres,Tags")
        self.ComboFieldsBox.model().item(5, 0).setCheckState(QtCore.Qt.Unchecked)
        self.ComboFieldsBox.setFont(data_font)
        self.horizontalLayout_2.addWidget(self.ComboFieldsBox)

        self.horizontalLayout_2.setStretch(0, 10)
        self.horizontalLayout_2.setStretch(1, 1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.clearGameView()

    def execSearch(self):
        fields_indexes = self.ComboFieldsBox.getSelectedItems()
        if len(fields_indexes) <= 0:
            print("no selected fields")
            return
        fields = []
        for i in fields_indexes:
            fields.append(self.fieldsDict[i])
        self.games = []
        self.clearGameView()
        results = self.searcher.search(self.SearchField.text(), fields, self.ResultsLimitSpinBox.value())
        self.ResultListModel.removeRows(0, self.ResultListModel.rowCount())
        for i in results:
            self.ResultListModel.appendRow(QtGui.QStandardItem(i["name"]))
            self.games.append(GameData(i))



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.clearGameView()
        self.SearchButton.setText(_translate("MainWindow", "Search"))
        self.ResultsLimitSpinBox.setToolTip(_translate("MainWindow", "Number of searched games"))
        self.SearchField.setToolTip(_translate("MainWindow", "Insert a query"))
        self.ComboFieldsBox.setToolTip(_translate("MainWindow", "Select fields where to search in"))

    def handleResultSelectionChange(self):
        if len(self.ResultList.selectedIndexes()) > 0:
            self.updateGameView(self.games[self.ResultList.selectedIndexes()[0].row()])

    def clearGameView(self):
        self.TitleLabel.setText(' ')
        self.AppidLabel.setText(' ')
        self.Image.setText(' ')
        self.DescriptionLabel.setText(' ')
        self.SentimentLineLabel.hide()
        self.DeveloperLabel.setText(' ')
        self.PublisherLabel.setText(' ')
        self.ReleaseLable.setText(' ')
        self.PriceLabel.setText(' ')
        self.GenresLabel.setText(' ')
        self.TagsLabel.setText(' ')
        self.CategoriesLabel.setText(' ')
        self.MinReqsLabel.setText(' ')
        self.RecReqsLabel.setText(' ')


    def updateGameView(self, game):
        color = '"#94a2b3"'
        self.TitleLabel.setText(game.name)
        self.AppidLabel.setText('<font color=' + color + '><b>APPID</b></font>: ' + game.app_id)
        self.DescriptionLabel.setText(game.description)

        if game.positive_ratings + game.negative_ratings > 0:
            self.SentimentLineLabel.show()
            self.SentimentLineLabel.set_green_percentage(int((game.positive_ratings * 100) / (game.positive_ratings + game.negative_ratings)))
        else:
            self.SentimentLineLabel.hide()

        self.DeveloperLabel.setText('<font color=' + color + '><b>DEVELOPER:</b></font> ' + game.developer)
        self.PublisherLabel.setText('<font color=' + color + '><b>PUBLISHER:</b></font> ' + game.publisher)
        self.ReleaseLable.setText('<font color=' + color + '><b>RELEASE DATE:</b></font> ' + game.release_date)
        self.PriceLabel.setText('<font color=' + color + '><b>PRICE:</b></font> ' + str(game.price) + '$')
        self.GenresLabel.setText('<font color=' + color + '><b>GENRES:</b></font> ' + game.genres)
        self.TagsLabel.setText('<font color=' + color + '><b>TAGS:</b></font> ' + game.tags)
        self.CategoriesLabel.setText('<font color=' + color + '><b>CATEGORIES:</b></font> ' + game.categories)
        self.MinReqsLabel.setText('<font color=' + color + '><b>MINIMUM REQUIREMENTS:</b></font> ' + game.minimum_requirements)
        self.RecReqsLabel.setText('<font color=' + color + '><b>RECOMMENDED REQUIREMENTS:</b></font> ' + game.recommended_requirements)
        self.Image.setText(' ')
        QtCore.QCoreApplication.processEvents()
        try:
            img = QtGui.QPixmap()
            img.loadFromData(request.urlopen(game.header_img).read())
            self.Image.setPixmap(img)
        except Exception as e:
            print(f"Error accessing {game.header_img}: {e}")

def launchGui(searcher):
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme()
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(searcher)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
