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

        # 画像保持用
        self.current_image = None
        self.outlined_image = None
        self.complement_image = None
        self.current_display_image = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # モルフォロジーのパラメータ（調整可能）
        self.gap = 20
        self.thickness = 3

        # 台座関連の設定
        self.base_var = tk.StringVar(value="16mm")
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }
        self.base_sizes = {
            "16mm": (200, 40),
            "14mm": (175, 35),
            "12mm": (150, 30),
            "10mm": (125, 25)
        }

        # ---------- GUI の設定 ----------
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

        # 操作フレーム
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10,0), sticky=(tk.W, tk.E))
        for i in range(4):
            self.operation_frame.grid_columnconfigure(i, weight=1)

        self.outline_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"
        )
        self.outline_btn.grid(row=0, column=0, padx=5)

        self.contour_debug_btn = ttk.Button(
            self.operation_frame,
            text="輪郭デバッグ",
            style="Gray.TButton",
            width=15,
            command=self.debug_show_contours,
            state="disabled"
        )
        self.contour_debug_btn.grid(row=0, column=1, padx=5)

        # ★ 台座合成ボタンを押すと、台座選択ラジオボタンを表示
        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座合成",
            style="Gray.TButton",
            width=15,
            command=self.toggle_base_options,
            state="disabled"
        )
        self.combine_btn.grid(row=0, column=2, padx=5)

        self.output_btn = ttk.Button(
            self.operation_frame,
            text="画像出力",
            style="Gray.TButton",
            width=15,
            state="disabled"
        )
        self.output_btn.grid(row=0, column=3, padx=5)

        self.selected_base_label = ttk.Label(
            self.operation_frame,
            text="選択中の台座: 16mm",
            font=("Helvetica", 9),
            foreground='#666666'
        )
        self.selected_base_label.grid(row=1, column=0, columnspan=4, pady=(5, 0))

        # ▼ 台座の選択フレーム（はじめは非表示にする）
        self.base_options_frame = ttk.Frame(self.operation_frame)

        # ここでラジオボタンを縦方向に配置する
        for size in ["16mm", "14mm", "12mm", "10mm"]:
            rbtn = ttk.Radiobutton(
                self.base_options_frame,
                text=f"台座 {size}",
                variable=self.base_var,
                value=size,
                command=self.update_selected_base_label
            )
            rbtn.pack(side=tk.TOP, anchor='w', pady=5)

        # 合成実行ボタンも縦方向に配置
        self.apply_base_btn = ttk.Button(
            self.base_options_frame,
            text="合成実行",
            command=self.combine_base
        )
        self.apply_base_btn.pack(side=tk.TOP, pady=5)

        # 初期状態では非表示（台座合成ボタンを押して表示）
        self.base_options_frame.grid_remove()

        # 結果表示フレーム
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

    def toggle_base_options(self):
        """台座合成ボタンを押したときに、台座選択ラジオボタンと合成実行ボタンを表示/非表示にする"""
        if self.base_options_frame.winfo_ismapped():
            # すでに表示されているなら隠す
            self.base_options_frame.grid_remove()
        else:
            # 非表示なら表示する
            self.base_options_frame.grid(row=2, column=0, columnspan=4, pady=(10, 0))
            print("台座を選択し、合成実行をクリックしてください。")

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
            # 処理ボタンを有効化
            self.outline_btn.configure(state="normal")
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
            # Linuxなどdelta未対応の場合のフォールバック
            if event.num == 5:  # スクロールダウン
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            elif event.num == 4:  # スクロールアップ
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        self.update_image_display(self.current_display_image)

    # 統一した二値化処理
    def get_binary_mask(self, image_cv):
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = cv2.bitwise_not(mask)
        print(f"Otsu threshold: {ret}, mask mean: {np.mean(mask):.2f}")
        return mask

    # 輪郭線作成
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

    # 輪郭デバッグ処理
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
            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y - 20]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            x_vals = [pt[0] for pt in lower_points]
            left_x = min(x_vals)
            right_x = max(x_vals)
            print(f"[輪郭デバッグ] 下端 y = {bottom_y}, 左 x = {left_x}, 右 x = {right_x}")
            debug_img = img.copy()
            cv2.drawContours(debug_img, [main_contour], -1, (255, 0, 0), 2)
            cv2.circle(debug_img, (left_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (right_x, bottom_y), 5, (0, 0, 255), -1)
            cv2.circle(debug_img, (int((left_x+right_x)/2), bottom_y), 5, (0, 255, 255), -1)
            result_pil = Image.fromarray(debug_img)
            self.update_image_display(result_pil)
            print("輪郭デバッグが完了しました。")
        except Exception as e:
            print("Error in debug_show_contours:", e)
            traceback.print_exc()

    # 台座合成ボタンを押した後、「合成実行」で呼ばれるのがこのメソッド
    def combine_base(self):
        print("台座合成開始...")
        try:
            # 1) 輪郭抽出とバウンディングボックス取得
            source = self.outlined_image if self.outlined_image is not None else self.current_image
            if source is None:
                print("画像が読み込まれていません。")
                return

            img = np.array(source.convert("RGB"))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = self.get_binary_mask(img_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("輪郭が検出されませんでした。")
                return
            main_contour = max(contours, key=cv2.contourArea)
            x_full, y_full, w_full, h_full = cv2.boundingRect(main_contour)

            # 2) トリミング（背景除去）
            cropped_bgr = img_bgr[y_full : y_full + h_full, x_full : x_full + w_full]
            cropped_rgba = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            # 3) 下端座標の計算（ローカル座標系）
            bottom_y_global = max(pt[0][1] for pt in main_contour)
            bottom_y_local = bottom_y_global - y_full

            lower_points = [pt[0] for pt in main_contour if pt[0][1] >= bottom_y_global - 5]
            if not lower_points:
                lower_points = [pt[0] for pt in main_contour]
            foot_left_local = min([p[0] for p in lower_points]) - x_full
            foot_right_local = max([p[0] for p in lower_points]) - x_full
            foot_width_local = foot_right_local - foot_left_local

            print(f"[補完線] 下端 y(local) = {bottom_y_local}, 足元幅 = {foot_width_local}, 全体幅 = {w_full}")

            # 補完線の左右端を計算
            comp_line_left_local = foot_left_local
            comp_line_right_local = w_full

            # 4) 合成キャンバス設定
            char_w, char_h = cropped_rgba.size
            base_key = self.base_var.get()  # 選択中の台座
            base_filename = self.base_parts.get(base_key, "")
            pedestal_size = self.base_sizes.get(base_key, (200, 40))

            # 台座画像の取得・リサイズ
            if os.path.exists(base_filename):
                pedestal_cv = cv2.imread(base_filename, cv2.IMREAD_UNCHANGED)
                pedestal_cv = cv2.resize(pedestal_cv, pedestal_size, interpolation=cv2.INTER_AREA)
                pedestal_cv = cv2.cvtColor(pedestal_cv, cv2.COLOR_BGRA2RGBA)
                pedestal_img = Image.fromarray(pedestal_cv)
            else:
                print(f"台座ファイルが見つかりません: {base_filename}")
                pedestal_img = Image.new("RGBA", pedestal_size, (128,128,128,255))
            ped_w, ped_h = pedestal_img.size

            # キャンバス：横幅はキャラ画像と台座画像の最大、縦はキャラ高さ＋台座高さ
            comp_w = max(char_w, ped_w)
            comp_h = char_h + ped_h
            composite = Image.new("RGBA", (comp_w, comp_h), (0, 0, 0, 0))

            # キャラクター画像をキャンバス上部中央に配置
            char_x = (comp_w - char_w) // 2
            composite.paste(cropped_rgba, (char_x, 0), cropped_rgba)

            # 補完線を描画
            draw = ImageDraw.Draw(composite)
            line_color = (255, 0, 0, 255)
            line_thickness = 2
            line_y = bottom_y_local
            line_left = char_x + comp_line_left_local
            line_right = char_x + comp_line_right_local
            draw.line([(line_left, line_y), (line_right, line_y)],
                      fill=line_color, width=line_thickness)

            # 台座を足元に配置
            ped_x = (comp_w - ped_w) // 2
            ped_y = line_y
            composite.paste(pedestal_img, (ped_x, ped_y), pedestal_img)

            # 表示
            self.update_image_display(composite)
            print(f"台座合成が完了しました。[選択された台座: {base_key}]")

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
