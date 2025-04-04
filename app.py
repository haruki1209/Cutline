import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import traceback

def find_top_alpha_in_base(base_rgba):
    """
    台座画像(base_rgba)のアルファチャンネルを走査して、
    '最上端'にあたる行（アルファ値 > 0 のピクセルが最初に出現する行）を見つけ、
    その行内でアルファ>0となるピクセルの左端・右端の列インデックスを返す。
    
    戻り値:
       ((left_col, top_row), (right_col, top_row))
       見つからなければ None を返す。
    """
    h, w = base_rgba.shape[:2]
    top_row = None
    for row in range(h):
        if np.any(base_rgba[row, :, 3] > 0):
            top_row = row
            break
    if top_row is None:
        return None
    row_alpha = base_rgba[top_row, :, 3]
    cols = np.where(row_alpha > 0)[0]
    if len(cols) == 0:
        return None
    left_col = int(np.min(cols))
    right_col = int(np.max(cols))
    return (left_col, top_row), (right_col, top_row)

def add_tab_to_character(char_rgba, binary_mask, tab_width=40, tab_height=20):
    """
    キャラクター画像(char_rgba)とその二値マスク(binary_mask)にタブを追加する。
    タブはキャラクターの下端中央に追加され、tab_width, tab_heightでサイズを指定する。
    画像サイズが足りない場合は、画像・マスクを拡張します。
    """
    # マスクはコピーして処理
    mask = binary_mask.copy()
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return char_rgba, mask
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w_cont, h_cont = cv2.boundingRect(main_contour)
    bottom_y = y + h_cont
    center_x = x + w_cont // 2
    tab_left = max(center_x - tab_width // 2, 0)
    tab_right = tab_left + tab_width
    tab_top = bottom_y
    tab_bottom = bottom_y + tab_height

    # マスクの高さが足りない場合、拡張する
    h_mask, w_mask = mask.shape[:2]
    if tab_bottom > h_mask:
        new_height = tab_bottom
        new_mask = np.zeros((new_height, w_mask), dtype=mask.dtype)
        new_mask[:h_mask, :] = mask
        mask = new_mask

    # タブ部分を白（255）で塗りつぶす
    cv2.rectangle(mask, (tab_left, tab_top), (tab_right, tab_bottom), 255, -1)

    # char_rgba も同様に高さが足りない場合、拡張する
    h_rgba, w_rgba = char_rgba.shape[:2]
    if tab_bottom > h_rgba:
        new_height = tab_bottom
        new_rgba = np.zeros((new_height, w_rgba, 4), dtype=char_rgba.dtype)
        new_rgba[:h_rgba, :, :] = char_rgba
        char_rgba = new_rgba

    # キャラクター画像の下端部分（直上の1行）のピクセル情報をタブ部分にコピー
    if bottom_y - 1 >= 0:
        bottom_row = char_rgba[bottom_y-1:bottom_y, tab_left:tab_right].copy()
    else:
        bottom_row = np.full((1, tab_right-tab_left, 4), 255, dtype=char_rgba.dtype)
    for row in range(tab_top, tab_bottom):
        char_rgba[row, tab_left:tab_right] = bottom_row
        # アルファは確実に255にする
        char_rgba[row, tab_left:tab_right, 3] = 255

    return char_rgba, mask

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("画像処理システム")
        
        # ウィンドウの最小サイズを設定
        self.root.minsize(1200, 800)
        
        # メインフレームの設定
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの設定
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # 左側：画像アップロードエリアとボタン
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # 画像アップロードエリア
        self.upload_frame = ttk.LabelFrame(self.left_frame, text="画像アップロード", padding="20")
        self.upload_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.upload_frame.grid_columnconfigure(0, weight=1)
        self.upload_frame.grid_rowconfigure(0, weight=1)
        
        # ドラッグ&ドロップエリア（シンプルなデザイン）
        self.drop_area = ttk.Label(
            self.upload_frame,
            text="⬆\n\nドラッグ&ドロップで画像をアップロード\n\nまたは",
            padding="50",
            relief="solid",
            style="Drop.TLabel"
        )
        self.drop_area.grid(row=0, column=0, padx=(20, 20), pady=(20, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ファイル選択ボタン
        self.select_btn = ttk.Button(
            self.upload_frame,
            text="ファイルを選択",
            command=self.select_file,
            style="Upload.TButton"
        )
        self.select_btn.grid(row=1, column=0, pady=(0, 20))
        
        # 操作ボタンエリア
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        self.operation_frame.grid_columnconfigure(0, weight=1)
        self.operation_frame.grid_columnconfigure(1, weight=1)
        self.operation_frame.grid_columnconfigure(2, weight=1)
        
        # ボタンのスタイル設定
        style = ttk.Style()
        style.configure("Drop.TLabel", 
                       font=("Helvetica", 11),
                       foreground='#666666')
        style.configure("Upload.TButton",
                       padding=5,
                       font=("Helvetica", 10),
                       background='white',
                       relief='solid')
        style.configure("Gray.TButton", 
                       padding=10, 
                       background='#808080')
        
        # 画像処理用の変数
        self.current_image = None  
        
        # 各種ボタン
        self.object_detect_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"
        )
        self.object_detect_btn.grid(row=0, column=0, padx=5)
        
        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座の合成",
            style="Gray.TButton",
            width=15,
            command=self.toggle_base_options,  # ダイアログ表示ではなくトグル表示
            state="disabled"
        )
        self.combine_btn.grid(row=0, column=1, padx=5)
        
        self.output_btn = ttk.Button(
            self.operation_frame,
            text="画像出力",
            style="Gray.TButton",
            width=15,
            state="disabled"
        )
        self.output_btn.grid(row=0, column=2, padx=5)
        
        # 選択された台座を表示するラベル
        self.selected_base_label = ttk.Label(
            self.operation_frame,
            text="",
            font=("Helvetica", 9),
            foreground='#666666'
        )
        self.selected_base_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))

        # 台座選択用の変数
        self.base_var = tk.StringVar(value="16mm")  
        self.base_var.trace('w', self.update_selected_base_label)

        # 台座パーツの定義
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }

        # ▼▼▼ 台座選択のトグル表示用フレーム（中央寄せ） ▼▼▼
        self.base_options_frame = ttk.Frame(self.operation_frame)
        self.base_options_inner_frame = ttk.Frame(self.base_options_frame)
        self.base_options_inner_frame.pack(expand=True)
        for size in ["16mm", "14mm", "12mm", "10mm"]:
            button = ttk.Checkbutton(
                self.base_options_inner_frame,
                text=f"台座 {size}",
                variable=self.base_var,
                onvalue=size,
                offvalue="",
                command=self.update_selected_base_label
            )
            button.pack(anchor="center", pady=5)
        self.apply_base_btn = ttk.Button(
            self.base_options_inner_frame,
            text="合成実行",
            command=self.combine_base
        )
        self.apply_base_btn.pack(anchor="center", pady=5)
        # ▲▲▲

        # 右側：処理結果エリア
        self.result_frame = ttk.LabelFrame(self.main_frame, text="処理結果", padding="20")
        self.result_frame.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=1)
        
        # 画像表示用のラベル
        self.result_label = ttk.Label(
            self.result_frame,
            text="画像を処理するとここに表示されます",
            padding="100",
            relief="groove",
            anchor="center",
            justify="center"
        )
        self.result_label.grid(
            row=0, 
            column=0, 
            sticky=(tk.W, tk.E, tk.N, tk.S),
            padx=10,
            pady=10
        )

        # ズーム関連
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.current_display_image = None

        # マウスホイールイベントをバインド
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)      
        self.result_label.bind('<Button-4>', self.on_mousewheel)         
        self.result_label.bind('<Button-5>', self.on_mousewheel)       

        # ドラッグ&ドロップの実装
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)

        # スタイル設定
        style.configure("TLabel", anchor="center", justify="center")

    # ------------------------------------------------
    # 台座オプション表示の切り替え
    # ------------------------------------------------
    def toggle_base_options(self):
        if self.base_options_frame.winfo_manager():
            self.base_options_frame.grid_remove()
        else:
            self.base_options_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))

    # ------------------------------------------------
    # 画像表示関連
    # ------------------------------------------------
    def resize_image_with_aspect_ratio(self, image, max_size):
        width, height = image.size
        ratio = min(max_size[0]/width, max_size[1]/height)
        new_size = (int(width*ratio), int(height*ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def update_image_display(self, image):
        if not image:
            return
        self.current_display_image = image
        original_size = image.size
        zoom_size = (int(original_size[0] * self.zoom_factor), 
                     int(original_size[1] * self.zoom_factor))
        zoomed_image = image.resize(zoom_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(zoomed_image)
        self.result_label.configure(image=photo, text="")
        self.result_label.image = photo

    def on_mousewheel(self, event):
        if not self.current_display_image:
            return
        if hasattr(event, 'delta'):
            if event.delta < 0:
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            else:
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        else:
            if event.num == 5:
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            elif event.num == 4:
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        self.update_image_display(self.current_display_image)

    # ------------------------------------------------
    # ファイル読み込み / ドラッグ&ドロップ
    # ------------------------------------------------
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            class DummyEvent:
                pass
            event = DummyEvent()
            event.data = file_path
            self.handle_drop(event)

    def handle_drop(self, event):
        file_path = event.data.strip('{}').strip('"')
        print(f"Processed path: {file_path}")
        try:
            image = Image.open(file_path)
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                self.current_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                self.current_image = image_array
            self.update_image_display(image)
            self.object_detect_btn.configure(state="normal")
            self.combine_btn.configure(state="normal")
            self.output_btn.configure(state="normal")
            print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")

    def handle_drag_enter(self, event):
        self.drop_area.configure(relief="sunken")

    def handle_drag_leave(self, event):
        self.drop_area.configure(relief="solid")

    # ------------------------------------------------
    # 画像処理：輪郭線作成
    # ------------------------------------------------
    def create_outline(self):
        print("Starting create_outline")
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        try:
            processed_binary = self.preprocess_image(self.current_image)
            contour = self.detect_contours(processed_binary)
            if contour is None:
                print("Error: 輪郭を検出できませんでした")
                return
            result = self.draw_outline(self.current_image, processed_binary, contour)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            print("輪郭線の作成が完了しました")
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()

    def preprocess_image(self, image):
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            _, binary = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return binary

    def detect_contours(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return None
        main_contour = max(contours, key=cv2.contourArea)
        smooth_contour = []
        for i in range(len(main_contour)):
            p1 = main_contour[i][0]
            p2 = main_contour[(i + 1) % len(main_contour)][0]
            for t in range(8):
                x = int(p1[0] + (p2[0] - p1[0]) * t / 8)
                y = int(p1[1] + (p2[1] - p1[1]) * t / 8)
                smooth_contour.append([[x, y]])
        smooth_contour = np.array(smooth_contour)
        epsilon = 0.0008 * cv2.arcLength(smooth_contour, True)
        approx = cv2.approxPolyDP(smooth_contour, epsilon, True)
        return approx

    def draw_outline(self, original_image, binary, contour):
        result = original_image.copy()
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        gap = 12
        thickness = 2
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
        kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
        outer_edge = cv2.dilate(mask, kernel_thick, iterations=1)
        inner_edge = cv2.dilate(mask, kernel_gap, iterations=1)
        outline = cv2.subtract(outer_edge, inner_edge)
        if result.shape[2] == 4:
            result[outline == 255] = (0, 0, 255, 255)
        else:
            result[outline == 255] = (0, 0, 255)
        return result

    # ------------------------------------------------
    # 画像処理：重心 & 最下点の計算
    # ------------------------------------------------
    def calculate_image_center_and_bottom(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None, None
        center_x = int(M["m10"] / M["m00"])
        bottom_y = max(point[0][1] for point in main_contour)
        return center_x, bottom_y

    # ------------------------------------------------
    # 画像処理：台座合成（キャラクターと台座の輪郭をブリッジして連結）
    # ------------------------------------------------
    def combine_base(self):
        """
        キャラクター画像と台座画像を大きなキャンバス上で合体し、
        キャラクター下端にタブ（差し込み用突起）を追加した上で、
        切り抜かれたキャラクター画像の下部を基準に台座画像を自動配置します。
        初期配置はキャラクター画像の最下端中央に台座画像を配置する仕様です。
        （※後にGUIでドラッグ＆ドロップによる微調整機能を追加予定）
        """
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        
        try:
            print("台座合成処理を開始")
            # --- 1) キャラクター画像をRGBA化 ---
            char_bgr = self.current_image
            if char_bgr.shape[2] == 3:
                char_rgba = np.dstack((char_bgr, np.full((char_bgr.shape[0], char_bgr.shape[1]), 255, dtype=np.uint8)))
            else:
                char_rgba = char_bgr.copy()
            
            # --- 1.5) キャラクターにタブを追加 ---
            # タブサイズは必要に応じて調整（例: 40x20ピクセル）
            processed_binary = self.preprocess_image(char_bgr)
            tab_width = 40
            tab_height = 20
            char_rgba, processed_binary = add_tab_to_character(char_rgba, processed_binary, tab_width, tab_height)
            
            # --- 2) 台座画像をRGBAで読み込み ---
            size = self.base_var.get()
            base_img_path = self.base_parts[size]
            base_rgba = cv2.imread(base_img_path, cv2.IMREAD_UNCHANGED)
            if base_rgba is None:
                print("Error: 台座画像の読み込みに失敗しました")
                return
            
            # --- 3) キャラクターの下端情報取得 ---
            center_x, bottom_y = self.calculate_image_center_and_bottom(processed_binary)
            if center_x is None or bottom_y is None:
                print("Error: キャラクターの輪郭を検出できませんでした")
                return
            
            # キャラクター下端の左右端を取得（margin内の点を収集）
            figure_contour = self.detect_contours(processed_binary)
            figure_bottom_points = []
            margin = 10
            for p in figure_contour:
                if abs(p[0][1] - bottom_y) <= margin:
                    figure_bottom_points.append(p[0])
            if len(figure_bottom_points) < 2:
                print("Warning: キャラクターの最下部左右端が見つかりませんでした")
                left_figure_bottom = (center_x - 20, bottom_y)
                right_figure_bottom = (center_x + 20, bottom_y)
            else:
                left_figure_bottom = min(figure_bottom_points, key=lambda p: p[0])
                right_figure_bottom = max(figure_bottom_points, key=lambda p: p[0])
            
            # --- 4) キャンバスの用意＆画像配置 ---
            char_h, char_w = char_rgba.shape[:2]
            base_h, base_w = base_rgba.shape[:2]
            # 台座配置：キャラクターの下端中央に配置する（オフセット調整可能）
            base_x = center_x - base_w // 2
            base_y = bottom_y - 10  # キャラクター下端から10ピクセル上に台座を配置（調整可能）
            new_h = max(char_h, base_y + base_h + 50)
            new_w = max(char_w, base_x + base_w + 50)
            canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            # キャラクター配置
            canvas[:char_h, :char_w] = char_rgba
            # 台座配置（アルファ>0 の部分だけコピー）
            for y in range(base_h):
                for x in range(base_w):
                    ay = base_y + y
                    ax = base_x + x
                    if 0 <= ay < new_h and 0 <= ax < new_w:
                        alpha_base = base_rgba[y, x, 3]
                        if alpha_base > 0:
                            canvas[ay, ax] = base_rgba[y, x]
            
            # --- 5) 台座画像内で実質的な最上端を検出 ---
            base_top_coords = find_top_alpha_in_base(base_rgba)
            if base_top_coords is None:
                print("Error: 台座画像内に有効なアルファ領域が見つかりません")
                return
            (left_col, top_row), (right_col, top_row) = base_top_coords
            # 台座画像の実質的な最上端の座標をキャンバス上に変換
            base_top_left = (base_x + left_col, base_y + top_row)
            base_top_right = (base_x + right_col, base_y + top_row)
            
            # --- 6) ブリッジ用ポリゴンの作成 ---
            # キャラクター下端左右と台座の実質的最上端左右を結ぶ
            bridging_pts = np.array([
                left_figure_bottom,
                base_top_left,
                base_top_right,
                right_figure_bottom
            ], dtype=np.int32)
            # 新しいマスクを作成してブリッジ部分を塗りつぶす
            bridge_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            cv2.fillPoly(bridge_mask, [bridging_pts], 255)
            # 既存のアルファチャネルとブリッジマスクの最大値をとる
            canvas[..., 3] = np.maximum(canvas[..., 3], bridge_mask)
            
            # --- 7) 合体後のシルエット作成 ---
            alpha_channel = canvas[:, :, 3]
            _, combined_binary = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if not contours:
                print("Error: 合体後の輪郭を検出できませんでした")
                return
            main_contour = max(contours, key=cv2.contourArea)
            
            # --- 8) モルフォロジー処理でアウトライン生成 ---
            gap = 12
            thickness = 2
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            temp_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            cv2.drawContours(temp_mask, [main_contour], -1, 255, -1)
            outer_edge = cv2.dilate(temp_mask, kernel_thick, iterations=1)
            inner_edge = cv2.dilate(temp_mask, kernel_gap, iterations=1)
            outline_mask = cv2.subtract(outer_edge, inner_edge)
            
            # --- 9) アウトラインを赤色(RGBA)で描画 ---
            outline_result = canvas.copy()
            red = (0, 0, 255, 255)
            outline_coords = np.where(outline_mask == 255)
            outline_result[outline_coords] = red
            
            # --- 10) 画面表示用にRGB変換して更新 ---
            final_bgr = cv2.cvtColor(outline_result, cv2.COLOR_BGRA2BGR)
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            final_pil = Image.fromarray(final_rgb)
            self.update_image_display(final_pil)
            
            print("キャラクターと台座をブリッジして1つの輪郭線を生成しました。")
        
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

    def update_selected_base_label(self, *args):
        selected = self.base_var.get()
        if selected:
            self.selected_base_label.configure(text=f"選択中の台座: {selected}")
        else:
            self.selected_base_label.configure(text="")

    def find_bottom_points(self, image, bottom_y, margin=5):
        if image.shape[2] == 4:
            mask = image[:, :, 3] > 0
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        bottom_region = mask[bottom_y-margin:bottom_y+margin, :]
        y_indices, x_indices = np.where(bottom_region > 0)
        if len(x_indices) == 0:
            return None
        left_x = np.min(x_indices)
        right_x = np.max(x_indices)
        actual_y = bottom_y - margin + y_indices[0]
        return (left_x, actual_y), (right_x, actual_y)

def create_base_image(width, height, base_width):
    """台座画像の作成（実用的なバージョン：差し込み口なし）"""
    canvas_width = base_width + 60
    canvas_height = int(base_width * 0.8)
    image = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    top_width = base_width
    bottom_width = base_width + 20
    top_height = canvas_height // 3
    top_left = (canvas_width//2 - top_width//2, 0)
    top_right = (canvas_width//2 + top_width//2, top_height)
    cv2.rectangle(image, top_left, top_right, (128, 128, 128, 255), -1)
    pts = np.array([
        [canvas_width//2 - top_width//2, top_height],
        [canvas_width//2 - bottom_width//2, canvas_height-1],
        [canvas_width//2 + bottom_width//2, canvas_height-1],
        [canvas_width//2 + top_width//2, top_height]
    ], np.int32)
    cv2.fillPoly(image, [pts], (128, 128, 128, 255))
    cv2.rectangle(image, top_left, top_right, (0, 0, 0, 255), 2)
    cv2.polylines(image, [pts], True, (0, 0, 0, 255), 2)
    return image

# 台座画像の生成
sizes = {
    "16mm": 160,
    "14mm": 140,
    "12mm": 120,
    "10mm": 100
}

for size_name, base_width in sizes.items():
    image = create_base_image(base_width + 100, base_width + 50, base_width)
    filename = f"nichidai_base_{size_name}.png"
    cv2.imwrite(filename, image)
    print(f"Created: {filename}")

def main():
    root = TkinterDnD.Tk()
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
