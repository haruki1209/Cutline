import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import traceback

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("画像処理システム")
        self.root.minsize(1200, 800)

        # 現在の画像保持用
        self.current_image = None
        self.outlined_image = None
        self.complement_image = None  # 補完線作成結果保存用
        self.current_display_image = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # メインフレームの設定
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # 左側フレーム（画像アップロードや各操作ボタン）
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # 画像アップロードエリア
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
            text="\n\nドラッグ＆ドロップで画像をアップロード\n\nまたは",
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

        # 操作用ボタンエリア
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10,0), sticky=(tk.W, tk.E))
        # ボタンを5列配置（輪郭線作成、補完線作成、輪郭デバッグ、台座合成、画像出力）
        self.operation_frame.grid_columnconfigure(0, weight=1)
        self.operation_frame.grid_columnconfigure(1, weight=1)
        self.operation_frame.grid_columnconfigure(2, weight=1)
        self.operation_frame.grid_columnconfigure(3, weight=1)
        self.operation_frame.grid_columnconfigure(4, weight=1)

        # 輪郭線作成ボタン
        self.outline_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"
        )
        self.outline_btn.grid(row=0, column=0, padx=5)
        
        # 補完線作成ボタン
        self.complement_line_btn = ttk.Button(
            self.operation_frame,
            text="補完線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_complement_line,
            state="disabled"
        )
        self.complement_line_btn.grid(row=0, column=1, padx=5)
        
        # 輪郭デバッグボタン
        self.contour_debug_btn = ttk.Button(
            self.operation_frame,
            text="輪郭デバッグ",
            style="Gray.TButton",
            width=15,
            command=self.debug_show_contours,
            state="disabled"
        )
        self.contour_debug_btn.grid(row=0, column=2, padx=5)
        
        # 台座合成ボタン（現段階ではプレースホルダ）
        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座合成",
            style="Gray.TButton",
            width=15,
            command=self.combine_base,
            state="disabled"
        )
        self.combine_btn.grid(row=0, column=3, padx=5)
        
        # 画像出力ボタン（現段階ではプレースホルダ）
        self.output_btn = ttk.Button(
            self.operation_frame,
            text="画像出力",
            style="Gray.TButton",
            width=15,
            state="disabled"
        )
        self.output_btn.grid(row=0, column=4, padx=5)
        
        self.selected_base_label = ttk.Label(
            self.operation_frame,
            text="",
            font=("Helvetica", 9),
            foreground='#666666'
        )
        self.selected_base_label.grid(row=1, column=0, columnspan=5, pady=(5,0))
        
        # 台座選択用（GUIはそのまま）
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
            text="合成実行（リセット済み）",
            command=self.combine_base
        )
        self.apply_base_btn.pack(anchor="center", pady=5)

        # 右側フレーム（処理結果表示エリア）
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
        
        # ズーム操作用バインド
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)
        self.result_label.bind('<Button-4>', self.on_mousewheel)
        self.result_label.bind('<Button-5>', self.on_mousewheel)
        
        # ドラッグ＆ドロップ設定
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)
        
        style.configure("TLabel", anchor="center", justify="center")

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
            self.current_image = image
            self.update_image_display(image)
            # 画像が読み込まれたので、各処理ボタンを有効化
            self.outline_btn.configure(state="normal")
            self.complement_line_btn.configure(state="normal")
            self.contour_debug_btn.configure(state="normal")
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

    # 輪郭線作成処理（先の実装）
    def create_outline(self):
        print("輪郭線作成開始...")
        try:
            # PIL画像をOpenCV用にBGR配列に変換
            image_cv = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            
            gap = 20
            thickness = 3
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap+1, 2*gap+1))
            dilate_gap = cv2.dilate(binary, kernel_gap, iterations=1)
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(gap+thickness)+1, 2*(gap+thickness)+1))
            dilate_thick = cv2.dilate(binary, kernel_thick, iterations=1)
            ring = cv2.subtract(dilate_thick, dilate_gap)
            result = image_cv.copy()
            result[ring == 255] = (0, 0, 255)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(result_rgb)
            self.outlined_image = pil_result
            self.update_image_display(pil_result)
            print("輪郭線作成が完了しました。")
        except Exception as e:
            print(f"Error in create_outline: {e}")
            traceback.print_exc()

    # 補完線作成処理（具体的対策に基づく実装）
    def create_complement_line(self):
        print("補完線作成開始...")
        try:
            # 補完線処理対象は、輪郭線作成結果があればそちら、なければ元画像
            source_image = self.outlined_image if self.outlined_image is not None else self.current_image
            if source_image is None:
                print("画像が読み込まれていません。")
                return
            img = np.array(source_image.convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("輪郭が検出されませんでした。")
                return
            main_contour = max(contours, key=cv2.contourArea)
            # 下端のY座標は輪郭上の最大値
            bottom_y = max(pt[0][1] for pt in main_contour)
            # 下端付近（bottom_y-5以上）の点から左右のx座標を抽出
            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y - 5]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            x_vals = [pt[0] for pt in lower_points]
            left_x = min(x_vals)
            right_x = max(x_vals)
            line_thickness = 2
            color = (0, 255, 0)  # 緑
            img_with_line = img.copy()
            cv2.line(img_with_line, (left_x, bottom_y), (right_x, bottom_y), color, thickness=line_thickness)
            result_pil = Image.fromarray(img_with_line)
            self.complement_image = result_pil
            self.update_image_display(result_pil)
            print("補完線作成が完了しました。")
        except Exception as e:
            print("Error in create_complement_line:", e)
            traceback.print_exc()

    # 輪郭デバッグ処理：主要輪郭を描画し、下端・左右端を確認する
    def debug_show_contours(self):
        print("輪郭デバッグ開始...")
        try:
            # 対象画像は輪郭線作成結果があればそれ、なければ元画像
            source_image = self.outlined_image if self.outlined_image is not None else self.current_image
            if source_image is None:
                print("画像が読み込まれていません。")
                return
            img = np.array(source_image.convert("RGB"))
            # 二値化処理
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("輪郭が検出されませんでした。")
                return
            main_contour = max(contours, key=cv2.contourArea)
            # 輪郭の極値を計算
            bottom_y = max(pt[0][1] for pt in main_contour)
            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y - 5]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            x_vals = [pt[0] for pt in lower_points]
            left_x = min(x_vals)
            right_x = max(x_vals)
            print(f"輪郭デバッグ情報: 下端 y = {bottom_y}, 左 x = {left_x}, 右 x = {right_x}")

            # 輪郭描画（オーバーレイ）
            debug_img = img.copy()
            cv2.drawContours(debug_img, [main_contour], -1, (255, 0, 0), 2)  # 輪郭を青色で描画
            # 下端の点に赤い丸印
            cv2.circle(debug_img, (left_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (right_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (int((left_x+right_x)/2), bottom_y), 5, (0, 255, 255), -1)  # 中央に黄色丸印
            result_pil = Image.fromarray(debug_img)
            self.update_image_display(result_pil)
            print("輪郭デバッグが完了しました。")
        except Exception as e:
            print("Error in debug_show_contours:", e)
            traceback.print_exc()

    # 台座合成機能（現段階ではプレースホルダ）
    def combine_base(self):
        print("台座合成機能はリセットされました。")
        self.update_image_display(self.current_image)

def main():
    root = TkinterDnD.Tk()
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
