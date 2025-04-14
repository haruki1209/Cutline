import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import traceback

def add_tab_to_character(char_rgba, binary_mask, tab_width=40, tab_height=20):
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
    alpha = rgba_image[:, :, 3]
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    _, y, _, h_rect = cv2.boundingRect(main_contour)
    bottom_y = y + h_rect
    return bottom_y

def create_outline_mask(bgr_or_rgba):
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

def calculate_filtered_foot_bbox(rgba_image, region_height=20, center_margin_ratio=0.25, min_area=10):
    bottom_y = calculate_bottom_of_alpha(rgba_image)
    if bottom_y is None:
        return None, None, None
    foot_top = max(0, bottom_y - region_height)
    alpha = rgba_image[..., 3]
    foot_region = alpha[foot_top:bottom_y, :]
    contours, _ = cv2.findContours(foot_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Detected contours:", len(contours))
    if not contours:
        return bottom_y, None, None
    image_width = rgba_image.shape[1]
    center_x = image_width / 2
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        cnt_center = x + w / 2
        print(f"Contour at x={x}, y={y}, w={w}, h={h}, area={area}, center={cnt_center}")
        if abs(cnt_center - center_x) < (image_width * center_margin_ratio):
            valid_contours.append(cnt)
    if not valid_contours:
        valid_contours = contours  # fallback
    main_contour = max(valid_contours, key=cv2.contourArea)
    xs = [pt[0][0] for pt in main_contour]
    x_min = int(np.min(xs))
    x_max = int(np.max(xs))
    return bottom_y, x_min, x_max

def draw_bottom_line_from_outline(image, outline_mask, line_thickness=2, color=(0, 0, 255)):
    bottom_y, x_min, x_max = calculate_filtered_foot_bbox(image, region_height=20)
    if bottom_y is None or x_min is None or x_max is None:
        return image
    cv2.line(image, (x_min, bottom_y), (x_max, bottom_y), color, thickness=line_thickness)
    return image

# 新規追加：最終結果の背景調整
def finalize_canvas(canvas, mode='trim'):
    if mode == 'trim':
        alpha = canvas[..., 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            return canvas[y:y+h, x:x+w]
        else:
            return canvas
    elif mode == 'transparent':
        result = canvas.copy()
        mask = (result[...,3] == 0)
        result[mask] = [0, 0, 0, 0]
        return result
    elif mode == 'white':
        result = canvas.copy()
        mask = (result[...,3] == 0)
        result[mask] = [255, 255, 255, 255]
        return result
    else:
        return canvas

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("画像処理システム")
        self.root.minsize(1200, 800)
        
        # 輪郭線のみ描画した状態の画像を保持する変数
        self.outlined_image = None
        
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
            command=self.toggle_base_options,
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
        
        # 台座選択（単体選択）
        self.base_var = tk.StringVar(value="16mm")
        self.base_var.trace('w', self.update_selected_base_label)
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }
        
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
        
        # 右側フレーム：処理結果表示
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
        
        # ドラッグ＆ドロップ設定
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)
        
        style.configure("TLabel", anchor="center", justify="center")

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
            if event.num == 5:
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            elif event.num == 4:
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        self.update_image_display(self.current_display_image)

    def create_outline(self):
        """キャラクターに赤い輪郭線を描画（キャラ本体から一定の間隔をあける）"""
        print("Starting create_outline")
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        try:
            binary_mask = create_outline_mask(self.current_image)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                print("Error: 輪郭を検出できませんでした")
                return
            main_contour = max(contours, key=cv2.contourArea)
            contour_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            #cv2.imwrite("debug_all_contours.png", contour_vis)

            gap = 12
            thickness = 2
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(gap+thickness)+1, 2*(gap+thickness)+1))
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap+1, 2*gap+1))
            outer_edge = cv2.dilate(binary_mask, kernel_thick, iterations=1)
            inner_edge = cv2.dilate(binary_mask, kernel_gap, iterations=1)
            outline = cv2.subtract(outer_edge, inner_edge)

            result = self.current_image.copy()
            if result.shape[2] == 4:
                result[outline == 255] = (0, 0, 255, 255)
            else:
                result[outline == 255] = (0, 0, 255)
            self.current_image = result
            self.outlined_image = self.current_image.copy()

            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            print("輪郭線の作成が完了しました")
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()

    def combine_base(self):
        if self.outlined_image is None:
            print("Error: 先に輪郭線を作成してください")
            return

        try:
            print("処理開始: 台座を最下部に配置し、下端付近の輪郭（補完線）を作成します。")
            # ① 輪郭線作成済み画像から RGBA 画像を生成
            char_bgr = self.outlined_image.copy()
            if char_bgr.shape[2] == 3:
                char_rgba = np.dstack((char_bgr, np.full((char_bgr.shape[0], char_bgr.shape[1]), 255, dtype=np.uint8)))
            else:
                char_rgba = char_bgr.copy()

            # ② キャラクターの下端を計算
            char_bottom = calculate_bottom_of_alpha(char_rgba)
            if char_bottom is None:
                char_bottom = char_rgba.shape[0]

            # ③ 台座サイズと台座ファイルを取得
            base_sizes = {
                "16mm": (200, 40),
                "14mm": (175, 35),
                "12mm": (150, 30),
                "10mm": (125, 25)
            }
            selected_size = self.base_var.get()
            if selected_size in base_sizes:
                base_width, base_height = base_sizes[selected_size]
            else:
                base_width, base_height = (200, 40)

            # 台座画像ファイルの読み込み（alpha付き）
            base_file = self.base_parts.get(selected_size, "")
            if base_file and os.path.exists(base_file):
                base_img = cv2.imread(base_file, cv2.IMREAD_UNCHANGED)
                base_img = cv2.resize(base_img, (base_width, base_height), interpolation=cv2.INTER_AREA)
            else:
                # 台座画像が見つからなければ、単色の台座を生成
                base_img = np.zeros((base_height, base_width, 4), dtype=np.uint8)
                base_img[..., :3] = 0
                base_img[..., 3] = 255

            # ④ 合成キャンバスの作成：幅はキャラクター画像と台座の最大値、縦はキャラ下端＋台座高さ
            composite_width = max(char_rgba.shape[1], base_width)
            composite_height = max(char_rgba.shape[0], char_bottom + base_height)
            composite = np.zeros((composite_height, composite_width, 4), dtype=np.uint8)

            # キャラクター画像を合成キャンバスの中央に配置
            x_offset = (composite_width - char_rgba.shape[1]) // 2
            composite[0:char_rgba.shape[0], x_offset:x_offset+char_rgba.shape[1]] = char_rgba

            # 台座配置：台座の上端をキャラクターの下端に合わせ、水平中央に配置
            base_x = (composite_width - base_width) // 2
            base_y = char_bottom
            for y in range(base_height):
                for x in range(base_width):
                    comp_y = base_y + y
                    comp_x = base_x + x
                    if comp_y >= composite_height or comp_x >= composite_width:
                        continue
                    base_pixel = base_img[y, x]
                    base_alpha = base_pixel[3] / 255.0
                    if base_alpha > 0:
                        composite[comp_y, comp_x] = base_pixel

            # ⑤ 下端付近のみの輪郭抽出＆描画
            # ROI設定：キャラクター下端から台座領域＋若干の余白
            roi_top = max(0, char_bottom - 10)
            roi_bottom = min(composite.shape[0], char_bottom + base_height + 10)
            roi = composite[roi_top:roi_bottom, :, :]
            # ROI の alpha チャネルで2値化
            roi_alpha = roi[..., 3]
            _, roi_mask = cv2.threshold(roi_alpha, 128, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            roi_mask_dilated = cv2.dilate(roi_mask, kernel, iterations=1)
            contours_roi, _ = cv2.findContours(roi_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_roi:
                main_contour_roi = max(contours_roi, key=cv2.contourArea)
                # ROI内の輪郭座標は (0,0) 基準なので、元の composite 座標に合わせるため y に roi_top を加算
                for pt in main_contour_roi:
                    pt[0][1] += roi_top
                outline_color = (0, 0, 255, 255)  # 赤色、完全不透明
                cv2.drawContours(composite, [main_contour_roi], -1, outline_color, 2)
            
            # ⑥ 最終的にキャンバスをトリミングまたは透明処理
            final_canvas = finalize_canvas(composite, mode='transparent')
            final_bgr = cv2.cvtColor(final_canvas, cv2.COLOR_BGRA2BGR)
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            final_pil = Image.fromarray(final_rgb)
            self.current_image = final_bgr
            self.update_image_display(final_pil)

            print("台座合成と下端領域の補完線作成が完了しました。")
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
