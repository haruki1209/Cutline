import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD

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
        self.current_image = None  # 現在の画像（OpenCV形式）
        
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
            command=self.show_base_dialog,
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
        self.base_var = tk.StringVar(value="サンワ工場_台座17cm")
        self.base_var.trace('w', self.update_selected_base_label)

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
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)       # Windows
        self.result_label.bind('<Button-4>', self.on_mousewheel)         # Linux(上スクロール)
        self.result_label.bind('<Button-5>', self.on_mousewheel)         # Linux(下スクロール)

        # ドラッグ&ドロップの実装
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)

        # スタイル設定
        style.configure("TLabel", anchor="center", justify="center")

    # ------------------------------------------------
    # 画像表示関連
    # ------------------------------------------------
    def resize_image_with_aspect_ratio(self, image, max_size):
        """アスペクト比を保持しながら画像をリサイズ"""
        width, height = image.size
        ratio = min(max_size[0]/width, max_size[1]/height)
        new_size = (int(width*ratio), int(height*ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def update_image_display(self, image):
        """画像表示の更新"""
        if not image:
            return

        self.current_display_image = image
        
        # オリジナルサイズ
        original_size = image.size
        
        # ズーム適用
        zoom_size = (int(original_size[0] * self.zoom_factor), 
                     int(original_size[1] * self.zoom_factor))
        zoomed_image = image.resize(zoom_size, Image.Resampling.LANCZOS)
        
        # PhotoImage形式に変換
        photo = ImageTk.PhotoImage(zoomed_image)
        
        # 画像を表示
        self.result_label.configure(image=photo, text="")
        self.result_label.image = photo  # 参照を保持

    def on_mousewheel(self, event):
        """マウスホイールでのズーム処理"""
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
        file_path = event.data
        file_path = file_path.strip('{}').strip('"')
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
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            gap = 20
            thickness = 3
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            dilate_gap = cv2.dilate(binary, kernel_gap, iterations=1)
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
            dilate_thick = cv2.dilate(binary, kernel_thick, iterations=1)
            ring = cv2.subtract(dilate_thick, dilate_gap)
            result = self.current_image.copy()
            result[ring == 255] = (0, 0, 255)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            print("輪郭線の作成が完了しました")
        except Exception as e:
            print(f"Error processing image: {e}")

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
    # 画像処理：台座合成
    # ------------------------------------------------
    def combine_base(self):
        print("Starting combine_base function")
        
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        
        try:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            gap = 20
            thickness = 3
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            dilate_gap = cv2.dilate(binary, kernel_gap, iterations=1)
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
            dilate_thick = cv2.dilate(binary, kernel_thick, iterations=1)
            ring = cv2.subtract(dilate_thick, dilate_gap)
            
            center_x, bottom_y = self.calculate_image_center_and_bottom(binary)
            if center_x is None or bottom_y is None:
                print("Error: 画像の特徴点を検出できませんでした")
                return
            
            # 台座選択に応じた画像パスとオフセット設定
            base_type = self.base_var.get()
            if base_type == "サンワ工場_台座17cm":
                base_img_path = "sanwa_base_17cm.png"
                offset_y = 20
            elif base_type == "サンワ工場_台座11cm":
                base_img_path = "sanwa_base_11cm.png"
                offset_y = 15
            else:  # ニチダイ工場_台座17cm
                base_img_path = "nichidai_base_17cm.png"
                offset_y = 20
            
            if not os.path.exists(base_img_path):
                print(f"Error: 台座画像が見つかりません: {base_img_path}")
                return
            
            base_img = cv2.imread(base_img_path, cv2.IMREAD_UNCHANGED)
            if base_img is None:
                print(f"Error: 台座画像の読み込みに失敗しました: {base_img_path}")
                return
            
            result = self.current_image.copy()
            result[ring == 255] = (0, 0, 255)
            cv2.circle(result, (center_x, bottom_y), 10, (0, 255, 0), -1)
            cv2.circle(result, (center_x, bottom_y), 12, (255, 255, 255), 2)
            
            result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            bh, bw = base_img.shape[:2]
            top_left_x = center_x - bw // 2
            top_left_y = bottom_y + offset_y
            
            canvas_h = max(result_rgba.shape[0], top_left_y + bh)
            canvas_w = max(result_rgba.shape[1], top_left_x + bw)
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            canvas[:result_rgba.shape[0], :result_rgba.shape[1]] = result_rgba
            
            for y in range(bh):
                for x in range(bw):
                    alpha = base_img[y, x, 3] if base_img.shape[2] == 4 else 255
                    if alpha > 0:
                        cy = top_left_y + y
                        cx = top_left_x + x
                        if 0 <= cy < canvas_h and 0 <= cx < canvas_w:
                            canvas[cy, cx] = base_img[y, x]
            
            final_bgr = canvas[..., :3]
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            final_pil = Image.fromarray(final_rgb)
            self.update_image_display(final_pil)
            
            print(f"選択された台座: {base_type}")
            print(f"重心X: {center_x}, 最下点Y: {bottom_y}")
            print("台座合成が完了しました")
            
        except Exception as e:
            print(f"Error combining base: {e}")
            import traceback
            traceback.print_exc()

    def update_selected_base_label(self, *args):
        """選択された台座のラベルを更新"""
        selected = self.base_var.get()
        if selected:
            self.selected_base_label.configure(text=f"選択中の台座: {selected}")
        else:
            self.selected_base_label.configure(text="")

    def show_base_dialog(self):
        """台座選択ダイアログを表示"""
        dialog = tk.Toplevel(self.root)
        dialog.title("台座選択")
        dialog.geometry("300x200")  # ダイアログのサイズ
        dialog.transient(self.root)  # メインウィンドウの子ウィンドウとして設定
        dialog.grab_set()  # モーダルダイアログとして設定
        
        # ダイアログの中央配置
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
            self.root.winfo_rooty() + self.root.winfo_height()//2 - 100
        ))

        # 台座オプションフレーム
        option_frame = ttk.LabelFrame(dialog, text="台座オプション", padding="10")
        option_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 台座の選択肢
        self.base_options = [
            "サンワ工場_台座17cm",
            "サンワ工場_台座11cm",
            "ニチダイ工場_台座17cm"
        ]

        # ラジオボタンで選択肢を表示
        for option in self.base_options:
            ttk.Radiobutton(
                option_frame,
                text=option,
                variable=self.base_var,
                value=option
            ).pack(anchor="w", pady=2)

        # ボタンフレーム
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=5)

        # OKボタン
        ok_btn = ttk.Button(
            button_frame,
            text="OK",
            command=lambda: self.apply_base_selection(dialog)
        )
        ok_btn.pack(side="right", padx=5)

        # キャンセルボタン
        cancel_btn = ttk.Button(
            button_frame,
            text="キャンセル",
            command=dialog.destroy
        )
        cancel_btn.pack(side="right", padx=5)

    def apply_base_selection(self, dialog):
        """選択された台座を適用"""
        dialog.destroy()
        self.update_selected_base_label()  # ラベルを更新
        self.combine_base()  # 台座合成処理を実行

def main():
    root = TkinterDnD.Tk()
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
