import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import traceback

def add_tab_to_character(char_rgba, binary_mask, tab_width=40, tab_height=20):
    """キャラクターにタブを追加する。"""
    mask = binary_mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # マスク拡張
    h_mask, w_mask = mask.shape[:2]
    if tab_bottom > h_mask:
        new_height = tab_bottom
        new_mask = np.zeros((new_height, w_mask), dtype=mask.dtype)
        new_mask[:h_mask, :] = mask
        mask = new_mask

    cv2.rectangle(mask, (tab_left, tab_top), (tab_right, tab_bottom), 255, -1)

    # RGBA拡張
    h_rgba, w_rgba = char_rgba.shape[:2]
    if tab_bottom > h_rgba:
        new_height = tab_bottom
        new_rgba = np.zeros((new_height, w_rgba, 4), dtype=char_rgba.dtype)
        new_rgba[:h_rgba, :, :] = char_rgba
        char_rgba = new_rgba

    # タブ部分の色をキャラクター下端の色で引き伸ばす
    if bottom_y - 1 >= 0:
        bottom_row = char_rgba[bottom_y-1:bottom_y, tab_left:tab_right].copy()
    else:
        bottom_row = np.full((1, tab_right-tab_left, 4), 255, dtype=char_rgba.dtype)
    for row in range(tab_top, tab_bottom):
        char_rgba[row, tab_left:tab_right] = bottom_row
        char_rgba[row, tab_left:tab_right, 3] = 255

    return char_rgba, mask

def add_supplement_region(char_rgba, extension_height=50):
    """キャラクター下端を延長し、補完領域を追加する。"""
    h, w = char_rgba.shape[:2]
    alpha = char_rgba[:, :, 3]
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return char_rgba
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w_rect, h_rect = cv2.boundingRect(main_contour)
    bottom_y = y + h_rect

    new_height = h + extension_height
    new_rgba = np.zeros((new_height, w, 4), dtype=char_rgba.dtype)
    new_rgba[:h, :w] = char_rgba

    # 下端を四角く延長
    left_bottom = (x, bottom_y)
    right_bottom = (x + w_rect, bottom_y)
    poly_pts = np.array([
        [left_bottom[0], left_bottom[1]],
        [left_bottom[0], left_bottom[1] + extension_height],
        [right_bottom[0], right_bottom[1] + extension_height],
        [right_bottom[0], right_bottom[1]]
    ], dtype=np.int32)

    supplement_mask = np.zeros((new_height, w), dtype=np.uint8)
    cv2.fillPoly(supplement_mask, [poly_pts], 255)
    new_rgba[..., 3] = np.maximum(new_rgba[..., 3], supplement_mask)

    return new_rgba

def calculate_bottom_of_alpha(rgba_image):
    """
    RGBA画像からアルファチャンネルを見て、キャラクター最下端のy座標を返す。
    戻り値: bottom_y
    """
    alpha = rgba_image[:, :, 3]
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    _, y, _, h_rect = cv2.boundingRect(main_contour)
    bottom_y = y + h_rect
    return bottom_y

def create_outline_mask(bgr_or_rgba):
    """
    キャラクター画像（BGR or RGBA）を受け取り、
    キャラクター部分の二値マスクを作成。
    """
    if bgr_or_rgba.shape[2] == 4:
        alpha = bgr_or_rgba[:, :, 3]
        _, binary = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(bgr_or_rgba, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary

def add_red_outline(bgr_image, binary_mask):
    """
    BGR画像とその二値マスクを受け取り、キャラクターの周囲に赤い輪郭線を付与したBGRを返す。
    """
    # 輪郭検出
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return bgr_image
    main_contour = max(contours, key=cv2.contourArea)

    # 輪郭を平滑化（スプライン的に補完）
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

    # アウトライン（赤）
    result = bgr_image.copy()
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, [approx], -1, 255, -1)
    gap = 12
    thickness = 2
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(gap+thickness)+1, 2*(gap+thickness)+1))
    kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap+1, 2*gap+1))
    outer_edge = cv2.dilate(mask, kernel_thick, iterations=1)
    inner_edge = cv2.dilate(mask, kernel_gap, iterations=1)
    outline = cv2.subtract(outer_edge, inner_edge)
    # 赤で塗る
    if result.shape[2] == 4:
        result[outline == 255] = (0, 0, 255, 255)
    else:
        result[outline == 255] = (0, 0, 255)

    return result

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("画像処理システム")
        self.root.minsize(1200, 800)
        
        # メインフレーム
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # 左側フレーム
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # 画像アップロード
        self.upload_frame = ttk.LabelFrame(self.left_frame, text="画像アップロード", padding="20")
        self.upload_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.upload_frame.grid_columnconfigure(0, weight=1)
        self.upload_frame.grid_rowconfigure(0, weight=1)
        
        style = ttk.Style()
        style.configure("Drop.TLabel", font=("Helvetica", 11), foreground='#666666')
        style.configure("Upload.TButton", padding=5, font=("Helvetica", 10), background='white', relief='solid')
        style.configure("Gray.TButton", padding=10, background='#808080')
        
        self.drop_area = ttk.Label(
            self.upload_frame,
            text="⬆\n\nドラッグ＆ドロップで画像をアップロード\n\nまたは",
            padding="50",
            relief="solid",
            style="Drop.TLabel"
        )
        self.drop_area.grid(row=0, column=0, padx=(20,20), pady=(20,5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.select_btn = ttk.Button(
            self.upload_frame,
            text="ファイルを選択",
            command=self.select_file,
            style="Upload.TButton"
        )
        self.select_btn.grid(row=1, column=0, pady=(0,20))
        
        # 操作ボタン
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10,0), sticky=(tk.W, tk.E))
        self.operation_frame.grid_columnconfigure(0, weight=1)
        self.operation_frame.grid_columnconfigure(1, weight=1)
        self.operation_frame.grid_columnconfigure(2, weight=1)
        
        self.current_image = None
        
        self.outline_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"
        )
        self.outline_btn.grid(row=0, column=0, padx=5)
        
        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座合成",
            style="Gray.TButton",
            width=15,
            command=self.toggle_base_options,  # ←トグル表示
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
        
        self.selected_base_label = ttk.Label(
            self.operation_frame,
            text="",
            font=("Helvetica", 9),
            foreground='#666666'
        )
        self.selected_base_label.grid(row=1, column=0, columnspan=3, pady=(5,0))
        
        # ニチダイ台座選択
        self.base_var = tk.StringVar(value="16mm")
        self.base_var.trace('w', self.update_selected_base_label)
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }
        
        # 台座選択トグルフレーム
        self.base_options_frame = ttk.Frame(self.operation_frame)
        self.base_options_inner_frame = ttk.Frame(self.base_options_frame)
        self.base_options_inner_frame.pack(expand=True)
        
        for size in ["16mm", "14mm", "12mm", "10mm"]:
            cbtn = ttk.Checkbutton(
                self.base_options_inner_frame,
                text=f"台座 {size}",
                variable=self.base_var,
                onvalue=size,
                offvalue="",
                command=self.update_selected_base_label
            )
            cbtn.pack(anchor="center", pady=5)
        
        self.apply_base_btn = ttk.Button(
            self.base_options_inner_frame,
            text="合成実行",
            command=self.combine_base
        )
        self.apply_base_btn.pack(anchor="center", pady=5)
        
        # 右側フレーム：処理結果
        self.result_frame = ttk.LabelFrame(self.main_frame, text="処理結果", padding="20")
        self.result_frame.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=1)
        
        self.result_label = ttk.Label(
            self.result_frame,
            text="画像を処理するとここに表示されます",
            padding="100",
            relief="groove",
            anchor="center",
            justify="center"
        )
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # ズーム関連
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.current_display_image = None
        
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)
        self.result_label.bind('<Button-4>', self.on_mousewheel)
        self.result_label.bind('<Button-5>', self.on_mousewheel)
        
        # ドラッグ&ドロップ
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)
        
        style.configure("TLabel", anchor="center", justify="center")

    # ▼▼▼ トグル動作 ▼▼▼
    def toggle_base_options(self):
        if self.base_options_frame.winfo_manager():
            self.base_options_frame.grid_remove()
        else:
            self.base_options_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))

    def update_selected_base_label(self, *args):
        selected = self.base_var.get()
        if selected:
            self.selected_base_label.configure(text=f"選択中の台座: {selected}")
        else:
            self.selected_base_label.configure(text="")

    # ▼▼▼ 画像読み込み / 表示 ▼▼▼
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
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
            self.outline_btn.configure(state="normal")
            self.combine_btn.configure(state="normal")
            self.output_btn.configure(state="normal")
            print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def handle_drag_enter(self, event):
        self.drop_area.configure(relief="sunken")
    
    def handle_drag_leave(self, event):
        self.drop_area.configure(relief="solid")

    def update_image_display(self, pil_image):
        if not pil_image:
            return
        self.current_display_image = pil_image
        w, h = pil_image.size
        zoom_w = int(w * self.zoom_factor)
        zoom_h = int(h * self.zoom_factor)
        zoomed_image = pil_image.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
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
            # Linux対策
            if event.num == 5:  # Down
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            elif event.num == 4:  # Up
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        self.update_image_display(self.current_display_image)

    # ▼▼▼ 輪郭線作成 ▼▼▼
    def create_outline(self):
        """キャラクターに赤い輪郭線を描画した状態を self.current_image に保存。"""
        print("Starting create_outline")
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        try:
            # 輪郭線マスク生成
            binary_mask = create_outline_mask(self.current_image)
            # 赤輪郭を重ねる
            outlined_bgr = add_red_outline(self.current_image, binary_mask)
            self.current_image = outlined_bgr  # 輪郭線が付与された状態を保持
            # 表示
            result_rgb = cv2.cvtColor(outlined_bgr, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            print("輪郭線の作成が完了しました")
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()

    # ▼▼▼ 台座合成（トグル→合成実行）▼▼▼
    def combine_base(self):
        """
        1) キャラクター画像（self.current_image）にタブ＋補完区間を付与
        2) 台座画像を読み込み（ここでは実際には黒い矩形に「台座」と描く形でもOK）
        3) キャラクター最下端に台座を配置
        4) 余計な再アウトラインは行わず、既存の赤輪郭線を保持
        """
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return

        # トグルフレームが表示されている場合は閉じる
        if self.base_options_frame.winfo_manager():
            self.base_options_frame.grid_remove()

        try:
            print("処理開始: キャラクターに補完区間を追加し、台座を最下部に配置します。")
            
            # --- キャラクターをRGBA化（既に赤輪郭線付き）---
            char_bgr = self.current_image
            if char_bgr.shape[2] == 3:
                char_rgba = np.dstack((char_bgr, np.full((char_bgr.shape[0], char_bgr.shape[1]), 255, dtype=np.uint8)))
            else:
                char_rgba = char_bgr.copy()

            # --- タブ追加用のマスク生成（もともと輪郭線付きなので少し甘めに二値化）---
            binary_mask = create_outline_mask(char_bgr)
            char_rgba, binary_mask = add_tab_to_character(char_rgba, binary_mask, tab_width=40, tab_height=20)

            # --- 補完区間を追加 ---
            extension_height = 50
            char_rgba = add_supplement_region(char_rgba, extension_height=extension_height)

            # --- キャラクターの最下端を取得 ---
            bottom_y = calculate_bottom_of_alpha(char_rgba)
            if bottom_y is None:
                print("Error: キャラクターの最下端が取得できませんでした")
                return

            # --- 台座（黒い長方形）を作成 + 「台座」文字描画 ---
            # ここでは、ニチダイ台座画像を貼る代わりに「黒い矩形に文字」の例を示します。
            # （もし本当に画像を貼るなら、下記で imread & 貼り付けすればOK）
            base_width = 200
            base_height = 40
            base_img = np.zeros((base_height, base_width, 4), dtype=np.uint8)
            # 塗りつぶし（黒）
            base_img[:, :, 0:3] = (0, 0, 0)  # BGR=黒
            base_img[:, :, 3] = 255         # アルファ=255
            # 文字「台座」を白で描画
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "台座"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (base_width - text_size[0]) // 2
            text_y = (base_height + text_size[1]) // 2
            cv2.putText(base_img, text, (text_x, text_y), font, 1, (255,255,255,255), 2, cv2.LINE_AA)

            # --- キャラクター画像と台座を合成するキャンバスを用意 ---
            char_h, char_w = char_rgba.shape[:2]
            new_h = char_h + 50  # 台座を置く余白
            new_w = max(char_w, base_width + 50)
            canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            # キャラクター配置
            canvas[:char_h, :char_w] = char_rgba

            # --- 台座をキャラ最下端に合わせて配置 ---
            base_x = (char_w - base_width)//2  # 水平中央合わせ
            base_y = bottom_y
            if base_y + base_height > new_h:
                # 足りなければキャンバス拡張
                tmp_h = base_y + base_height
                bigger = np.zeros((tmp_h, new_w, 4), dtype=np.uint8)
                bigger[:new_h, :new_w] = canvas
                canvas = bigger
                new_h = tmp_h
            
            for y in range(base_height):
                for x in range(base_width):
                    ay = base_y + y
                    ax = base_x + x
                    if 0 <= ay < new_h and 0 <= ax < new_w:
                        alpha_val = base_img[y, x, 3]
                        if alpha_val > 0:
                            canvas[ay, ax] = base_img[y, x]

            # --- 結果を self.current_image に反映 (BGRA→BGR) ---
            final_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
            self.current_image = final_bgr

            # PIL変換して表示
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            final_pil = Image.fromarray(final_rgb)
            self.update_image_display(final_pil)

            print("キャラクターの下端に台座（黒い矩形）を配置しました。")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

def main():
    from tkinterdnd2 import TkinterDnD
    root = TkinterDnD.Tk()
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
