import sys
import cv2
import numpy as np
import os
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QGroupBox, QGridLayout, QComboBox, QProgressBar,
    QFileDialog, QStackedWidget, QScrollArea, QMessageBox, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap


class PokemonCardUIOnly(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pokemon Card Defect Detection System - UI Only")
        self.setGeometry(100, 100, 1400, 900)

        # Camera and display state
        self.cap = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.current_frame = None
        # Screenshot area state
        self.capture_area_visible = True
        self.capture_area_w = 300
        self.capture_area_h = 200
        # Model selection / detection state
        self.model_path = None
        self.yolo = None
        self.detection_active = False
        self.infer_on_roi = True

        # Auto capture state & timers
        self.auto_capture_active = False
        self.countdown_remaining = 0
        self.interval_remaining = 0
        self.countdown_phase = None  # 'initial' or 'interval'
        self.batch_size = 10
        self.captured_count = 0
        self.overlay_text = ""
        # Session folders (per run)
        self.session_id = None
        self.session_folder_raw = None
        self.session_folder_result = None
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._on_countdown_tick)
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._on_capture_tick)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        self.tab_widget = QWidget()
        self.tab_layout = QVBoxLayout(self.tab_widget)

        self.tabs = QComboBox()
        self.tabs.addItem("Live View")
        self.tabs.addItem("History")
        self.tabs.currentIndexChanged.connect(self.switch_tab)
        self.tab_layout.addWidget(self.tabs)

        self.tab_content_area = QStackedWidget()
        self.tab_layout.addWidget(self.tab_content_area, 1)

        self.video_widget = self.create_video_widget()
        self.tab_content_area.addWidget(self.video_widget)

        self.history_widget = self.create_history_widget()
        self.tab_content_area.addWidget(self.history_widget)

        main_layout.addWidget(self.tab_widget, 3)

    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)

        title = QLabel("Pokemon Card Defect Detection")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Step 1: Screenshot area (UI only)
        step1_group = QGroupBox("步驟1：截圖範圍設定")
        step1_layout = QVBoxLayout(step1_group)
        self.camera_status_label = QLabel("攝影機狀態：UI預覽")
        step1_layout.addWidget(self.camera_status_label)
        self.camera_btn = QPushButton("啟動攝影機")
        self.camera_btn.clicked.connect(self.toggle_camera)
        step1_layout.addWidget(self.camera_btn)
        step1_layout.addWidget(QLabel("截圖範圍設定："))
        box_layout = QGridLayout()
        box_layout.addWidget(QLabel("寬度:"), 0, 0)
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(100, 500)
        self.width_spinbox.setValue(300)
        self.width_spinbox.valueChanged.connect(self.update_capture_area)
        box_layout.addWidget(self.width_spinbox, 0, 1)
        box_layout.addWidget(QLabel("高度:"), 1, 0)
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(100, 400)
        self.height_spinbox.setValue(200)
        self.height_spinbox.valueChanged.connect(self.update_capture_area)
        box_layout.addWidget(self.height_spinbox, 1, 1)
        step1_layout.addLayout(box_layout)
        self.show_capture_area_checkbox = QCheckBox("顯示截圖範圍")
        self.show_capture_area_checkbox.setChecked(True)
        self.show_capture_area_checkbox.toggled.connect(self.set_show_capture_area)
        step1_layout.addWidget(self.show_capture_area_checkbox)
        layout.addWidget(step1_group)

        # Step 2: YOLO (UI only)
        step2_group = QGroupBox("步驟2：YOLO模型設定")
        step2_layout = QVBoxLayout(step2_group)
        self.model_status_label = QLabel("模型狀態：未選擇")
        step2_layout.addWidget(self.model_status_label)
        self.select_model_btn = QPushButton("選擇YOLO模型")
        self.select_model_btn.clicked.connect(self.select_model)
        step2_layout.addWidget(self.select_model_btn)
        layout.addWidget(step2_group)

        # Step 3: Detection & Capture
        step3_group = QGroupBox("步驟3：實時辨識與自動拍攝")
        step3_layout = QVBoxLayout(step3_group)
        self.detection_status_label = QLabel("辨識狀態：未開始")
        step3_layout.addWidget(self.detection_status_label)
        mode_layout = QHBoxLayout()
        self.auto_mode_btn = QPushButton("自動拍攝模式")
        self.auto_mode_btn.clicked.connect(self.toggle_auto_capture_mode)
        mode_layout.addWidget(self.auto_mode_btn)
        self.manual_capture_btn = QPushButton("手動拍攝")
        self.manual_capture_btn.clicked.connect(self.noop)
        mode_layout.addWidget(self.manual_capture_btn)
        step3_layout.addLayout(mode_layout)
        self.start_detection_btn = QPushButton("開始實時辨識")
        self.start_detection_btn.clicked.connect(self.toggle_detection)
        step3_layout.addWidget(self.start_detection_btn)
        self.current_result_label = QLabel("當前結果：UI預覽")
        self.current_result_label.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;")
        step3_layout.addWidget(self.current_result_label)
        # 拍攝張數調整
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("拍攝張數:"))
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 100)
        self.batch_size_spinbox.setValue(self.batch_size)
        self.batch_size_spinbox.valueChanged.connect(self.on_batch_size_changed)
        size_layout.addWidget(self.batch_size_spinbox)
        step3_layout.addLayout(size_layout)
        progress_layout = QVBoxLayout()
        self.capture_info_label = QLabel("拍攝進度：0/10（UI預覽）")
        progress_layout.addWidget(self.capture_info_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        step3_layout.addLayout(progress_layout)
        layout.addWidget(step3_group)

        # Step 4: Report
        step4_group = QGroupBox("步驟4：報告生成")
        step4_layout = QVBoxLayout(step4_group)
        self.report_status_label = QLabel("報告狀態：未生成")
        step4_layout.addWidget(self.report_status_label)
        self.generate_report_btn = QPushButton("生成完整報告")
        self.generate_report_btn.clicked.connect(self.step4_generate_report)
        self.generate_report_btn.setEnabled(False)
        step4_layout.addWidget(self.generate_report_btn)
        layout.addWidget(step4_group)

        # Step 5: History
        step5_group = QGroupBox("步驟5：記錄回顧")
        step5_layout = QVBoxLayout(step5_group)
        self.history_status_label = QLabel("歷史記錄：可查看")
        step5_layout.addWidget(self.history_status_label)
        self.view_history_btn = QPushButton("查看歷史記錄")
        self.view_history_btn.clicked.connect(self.step5_view_history)
        step5_layout.addWidget(self.view_history_btn)
        layout.addWidget(step5_group)

        # Status area (UI only)
        status_group = QGroupBox("系統狀態（UI）")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("請按順序完成各步驟（UI預覽）")
        self.status_label.setStyleSheet("background-color: lightblue; padding: 5px; border-radius: 3px;")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)

        layout.addStretch()
        return panel

    def create_video_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        # 讓畫面可隨視窗擴展並填滿
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(True)
        self.video_label.setText("攝影機畫面（UI預覽）")
        layout.addWidget(self.video_label, 1)
        return widget

    def create_history_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        title = QLabel("檢測歷史記錄")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("選擇批次:"))
        self.batch_selector = QComboBox()
        self.batch_selector.currentIndexChanged.connect(self.load_batch_results)
        batch_layout.addWidget(self.batch_selector, 1)
        layout.addLayout(batch_layout)
        self.batch_info = QLabel("尚無檢測記錄")
        layout.addWidget(self.batch_info)
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_area.setStyleSheet("border: 1px solid #ccc;")
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_scroll_area.setWidget(self.results_widget)
        refresh_btn = QPushButton("刷新歷史記錄")
        refresh_btn.clicked.connect(self.update_history_page)
        layout.addWidget(refresh_btn)
        layout.addWidget(self.results_scroll_area, 1)
        return widget

    def switch_tab(self, index):
        self.tab_content_area.setCurrentIndex(index)

    def noop(self, *args, **kwargs):
        # 佔位：取消所有功能與邏輯
        return None

    # 功能：攝影機基本顯示
    def toggle_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.frame_timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_btn.setText("啟動攝影機")
            self.camera_status_label.setText("攝影機狀態：已停止")
            self.video_label.setText("攝影機畫面（UI預覽）")
            return
        # open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            self.camera_status_label.setText("攝影機狀態：開啟失敗")
            return
        self.camera_btn.setText("停止攝影機")
        self.camera_status_label.setText("攝影機狀態：運行中")
        self.frame_timer.start(33)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame
        # 畫面上疊加截圖範圍框（置中）
        disp = frame.copy()
        h, w = disp.shape[:2]
        rect_w = max(50, min(self.capture_area_w, w))
        rect_h = max(50, min(self.capture_area_h, h))
        x1 = (w - rect_w) // 2
        y1 = (h - rect_h) // 2
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        if self.capture_area_visible:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 顯示倒數/拍攝狀態文字疊圖
        if self.overlay_text:
            cv2.putText(disp, self.overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            # 顯示拍攝進度（英文）
            cv2.putText(disp, f"Progress: {self.captured_count}/{self.batch_size}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        # YOLO 即時推論與繪製結果
        if self.detection_active and self.yolo is not None:
            try:
                import time
                t0 = time.time()
                if self.infer_on_roi:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        results = self.yolo(roi, verbose=False)
                        plotted = results[0].plot()
                        ph, pw = plotted.shape[:2]
                        ph = min(ph, y2 - y1)
                        pw = min(pw, x2 - x1)
                        disp[y1:y1+ph, x1:x1+pw] = plotted[0:ph, 0:pw]
                        try:
                            num_boxes = len(results[0].boxes)
                        except Exception:
                            num_boxes = 0
                        dt = max(time.time() - t0, 1e-6)
                        fps = int(1.0 / dt)
                        self.current_result_label.setText(f"當前結果：{num_boxes} 個偵測框 | FPS={fps}")
                else:
                    results = self.yolo(frame, verbose=False)
                    disp = results[0].plot()
                    try:
                        num_boxes = len(results[0].boxes)
                    except Exception:
                        num_boxes = 0
                    dt = max(time.time() - t0, 1e-6)
                    fps = int(1.0 / dt)
                    self.current_result_label.setText(f"當前結果：{num_boxes} 個偵測框 | FPS={fps}")
            except Exception as e:
                self.current_result_label.setText(f"當前結果：推論錯誤 {str(e)}")

        # 顯示到 QLabel
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        lw = max(1, self.video_label.width())
        lh = max(1, self.video_label.height())
        pix = pix.scaled(lw, lh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    # ===== 自動拍攝模式 =====
    def toggle_auto_capture_mode(self):
        # 開/關 自動拍攝模式（10秒倒數，拍10張、每20秒）
        if self.auto_capture_active:
            # 停止模式
            self._stop_auto_capture(reset_status=True)
            return
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "請先啟動攝影機")
            return
        # 初始化狀態
        # 依使用者設定的拍攝張數
        try:
            self.batch_size = int(self.batch_size_spinbox.value())
        except Exception:
            self.batch_size = 10
        self.captured_count = 0
        self.progress_bar.setRange(0, self.batch_size)
        self.progress_bar.setValue(0)
        self.capture_info_label.setText(f"拍攝進度：{self.captured_count}/{self.batch_size}")
        self.auto_capture_active = True
        self.auto_mode_btn.setText("停止自動拍攝")
        self.overlay_text = "Countdown starting..."
        # 倒數10秒
        self.countdown_remaining = 10
        self.interval_remaining = 0
        self.countdown_phase = 'initial'
        # 建立批次資料夾（captured_images 與 Result_images）
        from datetime import datetime
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_folder_raw = os.path.join("captured_images", self.session_id)
        self.session_folder_result = os.path.join("Result_images", self.session_id)
        try:
            os.makedirs(self.session_folder_raw, exist_ok=True)
            os.makedirs(self.session_folder_result, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"建立批次資料夾失敗：{str(e)}")
        self._update_overlay_text()
        self.countdown_timer.start(1000)

    def _update_overlay_text(self):
        # 更新畫面上的英文提示，避免亂碼
        if not self.auto_capture_active:
            self.overlay_text = ""
            return
        if self.countdown_phase == 'initial' and self.countdown_timer.isActive():
            self.overlay_text = f"Countdown: {self.countdown_remaining}s"
        elif self.countdown_phase == 'interval' and self.capture_timer.isActive():
            self.overlay_text = f"Next shot in: {self.interval_remaining}s"
        else:
            self.overlay_text = ""

    def _on_countdown_tick(self):
        # 倒數計時：支援初始10秒與每張之間20秒倒數顯示
        if not self.auto_capture_active:
            self.countdown_timer.stop()
            return
        if self.countdown_phase == 'initial':
            self.countdown_remaining -= 1
            if self.countdown_remaining <= 0:
                # 初始倒數完成，立刻拍第一張
                self.countdown_timer.stop()
                self._capture_once()
                if self.captured_count < self.batch_size:
                    # 啟動20秒間隔拍攝，同時顯示20秒倒數
                    self.interval_remaining = 20
                    self.countdown_phase = 'interval'
                    self.capture_timer.start(20000)
                    self.countdown_timer.start(1000)
                else:
                    self._stop_auto_capture(reset_status=True)
        elif self.countdown_phase == 'interval':
            self.interval_remaining -= 1
            if self.interval_remaining <= 0:
                # 倒數顯示結束，等待 capture_timer 觸發實際拍攝
                self.countdown_timer.stop()
        else:
            self.countdown_timer.stop()
        self._update_overlay_text()

    def _on_capture_tick(self):
        if not self.auto_capture_active:
            self.capture_timer.stop()
            return
        self._capture_once()
        if self.captured_count >= self.batch_size:
            self.capture_timer.stop()
            self._stop_auto_capture(reset_status=True)
        else:
            # 下一張前重新計20秒倒數
            self.interval_remaining = 20
            self.countdown_phase = 'interval'
            if not self.countdown_timer.isActive():
                self.countdown_timer.start(1000)
        self._update_overlay_text()

    def _capture_once(self):
        # 使用目前影格與置中ROI擷取並存檔
        if self.current_frame is None:
            return
        frame = self.current_frame.copy()
        h, w = frame.shape[:2]
        rect_w = max(50, min(self.capture_area_w, w))
        rect_h = max(50, min(self.capture_area_h, h))
        x1 = (w - rect_w) // 2
        y1 = (h - rect_h) // 2
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        # 準備檔名與資料夾（以批次資料夾為主）
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        idx = self.captured_count + 1
        filename = f"pokemon_card_{idx:02d}_{ts}.jpg"
        # 若沒有啟動批次，Fallback 建立一次性資料夾
        if not self.session_folder_raw:
            self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_folder_raw = os.path.join("captured_images", self.session_id)
            os.makedirs(self.session_folder_raw, exist_ok=True)
        if not self.session_folder_result:
            self.session_folder_result = os.path.join("Result_images", self.session_id)
            os.makedirs(self.session_folder_result, exist_ok=True)
        save_path = os.path.join(self.session_folder_raw, filename)
        result_path = os.path.join(self.session_folder_result, filename)
        try:
            cv2.imwrite(save_path, roi)
            self.captured_count += 1
            self.progress_bar.setValue(self.captured_count)
            self.capture_info_label.setText(f"拍攝進度：{self.captured_count}/{self.batch_size}")
            # 如果有啟動即時辨識，將辨識疊圖一併存到 Result_images/<session>/
            if self.detection_active and self.yolo is not None:
                try:
                    results = self.yolo(roi)
                    overlay = results[0].plot()
                    cv2.imwrite(result_path, overlay)
                    self.current_result_label.setText(
                        f"Saved: RAW={save_path} | RESULT={result_path}")
                except Exception as e:
                    # 就算辨識失敗也不影響原始保存
                    self.current_result_label.setText(f"Saved RAW={save_path} (result save failed: {str(e)})")
            else:
                self.current_result_label.setText(f"Saved RAW={save_path}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"存檔失敗：{str(e)}")

    def _stop_auto_capture(self, reset_status=False):
        self.countdown_timer.stop()
        self.capture_timer.stop()
        self.auto_capture_active = False
        self.countdown_phase = None
        self.overlay_text = ""
        self.auto_mode_btn.setText("自動拍攝模式")
        # 啟用報告生成
        if self.captured_count > 0:
            self.generate_report_btn.setEnabled(True)
            self.report_status_label.setText("報告狀態：可生成")
            # 步驟5：有批次後可查看歷史
            self.history_status_label.setText("歷史記錄：可查看")
        if reset_status:
            self.current_result_label.setText("當前結果：UI預覽")

    def on_batch_size_changed(self, val):
        # 動態調整拍攝張數、進度條與資訊顯示
        try:
            self.batch_size = int(val)
        except Exception:
            return
        self.progress_bar.setRange(0, self.batch_size)
        self.capture_info_label.setText(f"拍攝進度：{self.captured_count}/{self.batch_size}")
        self._update_overlay_text()

    # ===== 步驟5：記錄回顧 =====
    def step5_view_history(self):
        # 切換到歷史頁並更新
        self.tabs.setCurrentIndex(1)
        self.update_history_page()
        self.status_label.setText("步驟5 - 正在查看歷史記錄")

    def update_history_page(self):
        # 列出 captured_images 與 Result_images 的批次，填入下拉選單
        try:
            self.batch_selector.clear()
            sessions = []
            base_raw = "captured_images"
            if os.path.isdir(base_raw):
                for name in sorted(os.listdir(base_raw)):
                    if os.path.isdir(os.path.join(base_raw, name)):
                        sessions.append(name)
            if sessions:
                # 最新在最上（倒序）
                for s in reversed(sessions):
                    self.batch_selector.addItem(s)
                self.batch_info.setText(f"找到 {len(sessions)} 個批次")
                # 預設載入最新
                self.batch_selector.setCurrentIndex(0)
                self.load_batch_results(0)
            else:
                self.batch_info.setText("尚無檢測記錄")
        except Exception as e:
            self.batch_info.setText(f"載入批次錯誤：{str(e)}")

    def load_batch_results(self, index):
        # 依選擇的批次載入 RAW 與 RESULT 圖片並顯示縮圖
        try:
            # 清空結果區
            while self.results_layout.count() > 0:
                item = self.results_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            session_id = self.batch_selector.currentText()
            if not session_id:
                return
            raw_dir = os.path.join("captured_images", session_id)
            res_dir = os.path.join("Result_images", session_id)
            raw_files = []
            if os.path.isdir(raw_dir):
                raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
            result_files = []
            if os.path.isdir(res_dir):
                result_files = sorted([f for f in os.listdir(res_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
            # 組對
            pairs = []
            for rf in raw_files:
                resf = rf if rf in result_files else None
                pairs.append((os.path.join(raw_dir, rf), os.path.join(res_dir, resf) if resf else None))
            if not pairs:
                self.results_layout.addWidget(QLabel("此批次無圖片"))
                return
            # 逐對顯示縮圖
            for i,(raw_path,res_path) in enumerate(pairs, start=1):
                row = QWidget()
                row_layout = QHBoxLayout(row)
                # RAW 縮圖
                try:
                    raw_img = cv2.imread(raw_path)
                    if raw_img is None:
                        raise RuntimeError("RAW 圖片讀取失敗")
                    rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg).scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    raw_label = QLabel()
                    raw_label.setPixmap(pix)
                    raw_label.setToolTip(os.path.basename(raw_path))
                    row_layout.addWidget(raw_label)
                except Exception:
                    row_layout.addWidget(QLabel("RAW讀取失敗"))
                # RESULT 縮圖
                if res_path and os.path.isfile(res_path):
                    try:
                        res_img = cv2.imread(res_path)
                        if res_img is None:
                            raise RuntimeError("RESULT 圖片讀取失敗")
                        rgb2 = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                        qimg2 = QImage(rgb2.data, rgb2.shape[1], rgb2.shape[0], rgb2.strides[0], QImage.Format_RGB888)
                        pix2 = QPixmap.fromImage(qimg2).scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        res_label = QLabel()
                        res_label.setPixmap(pix2)
                        res_label.setToolTip(os.path.basename(res_path))
                        row_layout.addWidget(res_label)
                    except Exception:
                        row_layout.addWidget(QLabel("RESULT讀取失敗"))
                else:
                    missing = QLabel("無對應結果圖")
                    missing.setStyleSheet("color:#c00;font-weight:bold")
                    row_layout.addWidget(missing)
                self.results_layout.addWidget(row)
            self.batch_info.setText(f"批次 {session_id}：共 {len(pairs)} 對")
        except Exception as e:
            self.batch_info.setText(f"載入結果錯誤：{str(e)}")

    # ===== 步驟4：報告生成 =====
    def step4_generate_report(self):
        # 從最近的批次資料夾彙整 RAW / RESULT 圖並生成 HTML 報告
        try:
            # 取得 Session ID（若無則選擇最新）
            session_id = self.session_id
            base_raw = "captured_images"
            base_res = "Result_images"
            if not session_id:
                if not os.path.isdir(base_raw):
                    QMessageBox.warning(self, "警告", "尚未拍攝，找不到 captured_images")
                    return
                sessions = [d for d in os.listdir(base_raw) if os.path.isdir(os.path.join(base_raw, d))]
                if not sessions:
                    QMessageBox.warning(self, "警告", "尚未拍攝，找不到批次資料夾")
                    return
                sessions.sort()
                session_id = sessions[-1]
            raw_dir = os.path.join(base_raw, session_id)
            res_dir = os.path.join(base_res, session_id)
            if not os.path.isdir(raw_dir):
                QMessageBox.warning(self, "警告", f"找不到原始資料夾：{raw_dir}")
                return
            raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            result_files = sorted([f for f in os.listdir(res_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(res_dir) else []
            # 建立對應清單
            pairs = []
            for rf in raw_files:
                resf = rf if rf in result_files else None
                pairs.append((os.path.join(raw_dir, rf), os.path.join(res_dir, resf) if resf else None))
            if not pairs:
                QMessageBox.information(self, "提示", "此批次尚無可彙整的圖片")
                return
            # 生成 HTML 報告
            os.makedirs("reports", exist_ok=True)
            html_path = os.path.join("reports", f"report_{session_id}.html")
            from datetime import datetime
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>")
                f.write(f"<title>Report {session_id}</title>")
                f.write("<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;} .pair{display:flex;gap:20px;margin-bottom:24px;} .pair img{max-width:48%;border:1px solid #ccc;} .meta{margin-bottom:12px;padding:10px;background:#f5f5f5;border:1px solid #ddd;} .missing{color:#c00;font-weight:bold}</style>")
                f.write("</head><body>")
                f.write(f"<h1>Pokemon Card Inspection Report</h1>")
                f.write(f"<div class='meta'><div>Session: {session_id}</div><div>Created at: {created_at}</div><div>Total images: {len(pairs)}</div></div>")
                for i,(raw_path,res_path) in enumerate(pairs, start=1):
                    # 先將檔案路徑中的反斜線轉為斜線，避免 f-string 轉譯錯誤
                    raw_web = raw_path.replace('\\', '/')
                    res_web = res_path.replace('\\', '/') if res_path else None
                    f.write("<div class='pair'>")
                    f.write(f"<div><h3>RAW #{i}</h3><img src='../{raw_web}' alt='raw'></div>")
                    if res_web and os.path.isfile(res_path):
                        f.write(f"<div><h3>RESULT #{i}</h3><img src='../{res_web}' alt='result'></div>")
                    else:
                        f.write(f"<div><h3>RESULT #{i}</h3><div class='missing'>No result image</div></div>")
                    f.write("</div>")
                f.write("</body></html>")
            self.report_status_label.setText(f"報告狀態：已生成 - {os.path.basename(html_path)}")
            QMessageBox.information(self, "成功", f"報告已生成：{html_path}")
            try:
                import webbrowser, os as _os
                webbrowser.open('file://' + _os.path.abspath(html_path))
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"報告生成失敗：{str(e)}")

    def update_capture_area(self):
        self.capture_area_w = int(self.width_spinbox.value())
        self.capture_area_h = int(self.height_spinbox.value())

    def set_show_capture_area(self, checked):
        self.capture_area_visible = bool(checked)

    # 功能：選擇模型（僅選檔顯示路徑）
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇YOLO模型", "", "Model Files (*.pt *.onnx);;All Files (*)")
        if path:
            self.model_path = path
            try:
                if YOLO is None:
                    raise RuntimeError("Ultralytics 未安裝或載入失敗")
                self.yolo = YOLO(path)
                self.model_status_label.setText(f"模型狀態：已載入 {os.path.basename(path)}")
            except Exception as e:
                self.yolo = None
                self.model_status_label.setText(f"模型狀態：已選擇（載入失敗） {path}\n錯誤：{str(e)}")

    def toggle_detection(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "請先啟動攝影機")
            return
        if self.yolo is None and self.model_path:
            try:
                if YOLO is None:
                    raise RuntimeError("Ultralytics 未安裝或載入失敗")
                self.yolo = YOLO(self.model_path)
                self.model_status_label.setText(f"模型狀態：已載入 {os.path.basename(self.model_path)}")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"模型載入失敗：{str(e)}")
                return
        if self.yolo is None:
            QMessageBox.warning(self, "警告", "尚未選擇或載入YOLO模型")
            return
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.detection_status_label.setText("辨識狀態：運行中")
            self.start_detection_btn.setText("停止實時辨識")
            self.current_result_label.setText("當前結果：推論中…")
        else:
            self.detection_status_label.setText("辨識狀態：未開始")
            self.start_detection_btn.setText("開始實時辨識")
            self.current_result_label.setText("當前結果：UI預覽")


def main():
    app = QApplication(sys.argv)
    window = PokemonCardUIOnly()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()