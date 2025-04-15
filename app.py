import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
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

        # モルフォロジーのパラメータ（必要に応じて調整）
        self.gap = 20
        self.thickness = 3

        # 台座関連の設定（既存台座画像がある前提）
        self.base_var = tk.StringVar(value="16mm")
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }
        # 台座画像のリサイズサイズ（幅×高さ）
        self.base_sizes = {"16mm": (200, 40), "14mm": (175, 35), "12mm": (150, 30), "10mm": (125, 25)}

        # --- 以下、GUIの設定（画像アップロード、操作パネル、結果表示など） ---
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.left_frame.grid_columnconfigure(0, weight=1)

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

        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10,0), sticky=(tk.W, tk.E))
        # 5列配置（輪郭線作成、補完線作成、輪郭デバッグ、台座合成、画像出力）
        self.operation_frame.grid_columnconfigure(0, weight=1)
        self.operation_frame.grid_columnconfigure(1, weight=1)
        self.operation_frame.grid_columnconfigure(2, weight=1)
        self.operation_frame.grid_columnconfigure(3, weight=1)
        self.operation_frame.grid_columnconfigure(4, weight=1)

        self.outline_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"
        )
        self.outline_btn.grid(row=0, column=0, padx=5)

        self.complement_line_btn = ttk.Button(
            self.operation_frame,
            text="補完線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_complement_line,
            state="disabled"
        )
        self.complement_line_btn.grid(row=0, column=1, padx=5)

        self.contour_debug_btn = ttk.Button(
            self.operation_frame,
            text="輪郭デバッグ",
            style="Gray.TButton",
            width=15,
            command=self.debug_show_contours,
            state="disabled"
        )
        self.contour_debug_btn.grid(row=0, column=2, padx=5)

        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座合成",
            style="Gray.TButton",
            width=15,
            command=self.combine_base,
            state="disabled"
        )
        self.combine_btn.grid(row=0, column=3, padx=5)

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

        self.result_label.bind('<MouseWheel>', self.on_mousewheel)
        self.result_label.bind('<Button-4>', self.on_mousewheel)
        self.result_label.bind('<Button-5>', self.on_mousewheel)

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

    # 統一した二値化処理を行う関数
    def get_binary_mask(self, image_cv):
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = cv2.bitwise_not(mask)
        print(f"Otsu threshold: {ret}, mask mean: {np.mean(mask):.2f}")
        return mask

    # 輪郭線作成処理（統一した二値化処理を利用）
    def create_outline(self):
        print("輪郭線作成開始...")
        try:
            image_cv = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            binary = self.get_binary_mask(image_cv)
            gap = self.gap
            thickness = self.thickness
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

    # 補完線作成処理（統一した二値化処理を使用し、左右に個別余白を設定）
    def create_complement_line(self):
        print("補完線作成開始...")
        try:
            source_image = self.outlined_image if self.outlined_image is not None else self.current_image
            if source_image is None:
                print("画像が読み込まれていません。")
                return

            img = np.array(source_image.convert("RGB"))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = self.get_binary_mask(img_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("輪郭が検出されませんでした。")
                return

            main_contour = max(contours, key=cv2.contourArea)
            bottom_y = max(pt[0][1] for pt in main_contour)
            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y - 5]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            x_vals = [pt[0] for pt in lower_points]
            left_x = min(x_vals)
            right_x = max(x_vals)
            print(f"[補完線] 元の下端 y = {bottom_y}, 左 x = {left_x}, 右 x = {right_x}")

            # 個別に余白を設定（左は30、右は50ピクセル）
            left_margin = 0
            right_margin = 350
            h_img, w_img, _ = img_bgr.shape
            extended_left = max(0, left_x - left_margin)
            extended_right = min(w_img - 1, right_x + right_margin)
            
            line_thickness = 2
            color = (0, 255, 0)  # 緑 (BGR)
            img_with_line = img.copy()
            cv2.line(img_with_line, (extended_left, bottom_y), (extended_right, bottom_y), color, thickness=line_thickness)
            result_pil = Image.fromarray(img_with_line)
            self.complement_image = result_pil
            self.update_image_display(result_pil)
            print(f"[補完線] 拡張後 左 x = {extended_left}, 右 x = {extended_right}")
            print("補完線作成が完了しました。")
        except Exception as e:
            print("Error in create_complement_line:", e)
            traceback.print_exc()

    # 輪郭デバッグ処理：統一された二値化処理で得られた輪郭を描画し、下端・左右端を表示する
    def debug_show_contours(self):
        print("輪郭デバッグ開始...")
        try:
            source_image = self.outlined_image if self.outlined_image is not None else self.current_image
            if source_image is None:
                print("画像が読み込まれていません。")
                return
            img = np.array(source_image.convert("RGB"))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            binary = self.get_binary_mask(img_bgr)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("輪郭が検出されませんでした。")
                return
            main_contour = max(contours, key=cv2.contourArea)
            bottom_y = max(pt[0][1] for pt in main_contour)
            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y - 5]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            x_vals = [pt[0] for pt in lower_points]
            left_x = min(x_vals)
            right_x = max(x_vals)
            print(f"[輪郭デバッグ] 下端 y = {bottom_y}, 左 x = {left_x}, 右 x = {right_x}")
            debug_img = img.copy()
            cv2.drawContours(debug_img, [main_contour], -1, (255, 0, 0), 2)  # 青色で描画
            cv2.circle(debug_img, (left_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (right_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (int((left_x+right_x)/2), bottom_y), 5, (0, 255, 255), -1)
            result_pil = Image.fromarray(debug_img)
            self.update_image_display(result_pil)
            print("輪郭デバッグが完了しました。")
        except Exception as e:
            print("Error in debug_show_contours:", e)
            traceback.print_exc()

    # 台座合成処理：キャラクター画像と台座画像を合成する
    def combine_base(self):
        print("台座合成開始...")
        try:
            # キャラクター画像は輪郭線作成済みのものを優先
            source = self.outlined_image if self.outlined_image is not None else self.current_image
            if source is None:
                print("画像が読み込まれていません。")
                return
            
            # キャラクター画像をRGBAに変換
            source_rgba = source.convert("RGBA")
            source_np = np.array(source_rgba)
            char_h, char_w, _ = source_np.shape

            # キャラクターの下端をαチャンネルから取得
            alpha_channel = source_np[:, :, 3]
            rows = np.where(np.any(alpha_channel > 0, axis=1))[0]
            if len(rows) == 0:
                char_bottom = char_h
            else:
                char_bottom = rows[-1]
            print(f"キャラクター下端: {char_bottom}")

            # 台座画像の取得
            base_key = self.base_var.get()
            base_filename = self.base_parts.get(base_key, "")
            pedestal_size = self.base_sizes.get(base_key, (200, 40))
            if os.path.exists(base_filename):
                pedestal_cv = cv2.imread(base_filename, cv2.IMREAD_UNCHANGED)
                pedestal_cv = cv2.resize(pedestal_cv, pedestal_size, interpolation=cv2.INTER_AREA)
                pedestal_cv = cv2.cvtColor(pedestal_cv, cv2.COLOR_BGRA2RGBA)
                pedestal_img = Image.fromarray(pedestal_cv)
            else:
                print(f"台座ファイルが見つかりません: {base_filename}")
                pedestal_img = Image.new("RGBA", pedestal_size, (128,128,128,255))
            
            ped_w, ped_h = pedestal_img.size
            print(f"台座サイズ: {ped_w}x{ped_h}")

            # 合成キャンバスのサイズ
            comp_w = max(char_w, ped_w)
            comp_h = char_h + ped_h
            composite = Image.new("RGBA", (comp_w, comp_h), (0,0,0,0))

            # キャラクター画像をキャンバス上部中央に配置
            char_x = (comp_w - char_w) // 2
            composite.paste(source_rgba, (char_x, 0), source_rgba)
            
            # 補完線（赤い水平線）をキャラクター下端に描画
            draw = ImageDraw.Draw(composite)
            line_thickness = 2
            line_color = (0, 0, 255, 255)  # 赤色
            draw.line([(char_x, char_bottom), (char_x + char_w, char_bottom)], fill=line_color, width=line_thickness)
            
            # 台座画像を下部中央に配置：キャラクター下端に余白なく接するように配置
            ped_x = (comp_w - ped_w) // 2
            ped_y = char_bottom  # 台座画像をキャラクターの下端にぴったり合わせる
            composite.paste(pedestal_img, (ped_x, ped_y), pedestal_img)

            self.update_image_display(composite)
            print("台座合成が完了しました。")
        except Exception as e:
            print("Error in combine_base:", e)
            traceback.print_exc()

def main():
    root = TkinterDnD.Tk()
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
