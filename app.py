import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
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
        self.base_var = tk.StringVar(value="16mm")  # デフォルト値を16mmに変更
        self.base_var.trace('w', self.update_selected_base_label)

        # 台座パーツの定義
        self.base_parts = {
            "16mm": "nichidai_base_16mm.png",
            "14mm": "nichidai_base_14mm.png",
            "12mm": "nichidai_base_12mm.png",
            "10mm": "nichidai_base_10mm.png"
        }

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
        """輪郭線の作成（改善版）"""
        print("Starting create_outline")
        
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
            
        try:
            # 画像の前処理
            processed_binary = self.preprocess_image(self.current_image)
            
            # 輪郭検出
            contour = self.detect_contours(processed_binary)
            if contour is None:
                print("Error: 輪郭を検出できませんでした")
                return
            
            # 輪郭線の描画
            result = self.draw_outline(self.current_image, processed_binary, contour)
            
            # 結果を表示
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            
            print("輪郭線の作成が完了しました")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()

    def preprocess_image(self, image):
        """画像の前処理（改善版）"""
        # アルファチャンネルの確認
        if image.shape[2] == 4:
            # アルファチャンネルがある場合
            alpha = image[:, :, 3]
            _, binary = cv2.threshold(alpha, 240, 255, cv2.THRESH_BINARY)  # しきい値を上げる
        else:
            # RGBの場合
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # しきい値を上げる

        # ノイズ除去を強化
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return binary

    def detect_contours(self, binary):
        """輪郭検出の処理（精度改善版）"""
        # 輪郭検出
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS  # より精密な輪郭検出
        )
        
        if not contours:
            return None
        
        # 最大の輪郭を選択
        main_contour = max(contours, key=cv2.contourArea)
        
        # 輪郭を滑らかにする
        smooth_contour = []
        for i in range(len(main_contour)):
            p1 = main_contour[i][0]
            p2 = main_contour[(i + 1) % len(main_contour)][0]
            # より細かい補間
            for t in range(8):
                x = int(p1[0] + (p2[0] - p1[0]) * t / 8)
                y = int(p1[1] + (p2[1] - p1[1]) * t / 8)
                smooth_contour.append([[x, y]])
        
        smooth_contour = np.array(smooth_contour)
        
        # 輪郭の近似を調整
        epsilon = 0.0008 * cv2.arcLength(smooth_contour, True)
        approx = cv2.approxPolyDP(smooth_contour, epsilon, True)
        
        return approx

    def draw_outline(self, original_image, binary, contour):
        """輪郭線の描画処理（改善版）"""
        result = original_image.copy()
        
        # マスクの作成
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # パラメータ調整
        gap = 12  # 内側への距離
        thickness = 2  # 輪郭線の太さ
        
        # 輪郭線の作成
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
        kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
        
        # 外側と内側の輪郭を作成
        outer_edge = cv2.dilate(mask, kernel_thick, iterations=1)
        inner_edge = cv2.dilate(mask, kernel_gap, iterations=1)
        
        # 輪郭線の抽出
        outline = cv2.subtract(outer_edge, inner_edge)
        
        # 輪郭線を赤色で描画
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
    # 画像処理：台座合成
    # ------------------------------------------------
    def combine_base(self):
        """台座の合成処理（改善版）"""
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        
        try:
            print("台座合成処理を開始")
            
            # 画像の前処理と輪郭検出
            processed_binary = self.preprocess_image(self.current_image)
            contour = self.detect_contours(processed_binary)
            if contour is None:
                print("Error: 輪郭を検出できませんでした")
                return

            # 重心と最下点の計算
            center_x, bottom_y = self.calculate_image_center_and_bottom(processed_binary)
            print(f"重心X: {center_x}, 最下点Y: {bottom_y}")
            
            # 選択された台座の読み込み
            size = self.base_var.get()
            base_img_path = self.base_parts[size]
            print(f"選択された台座: {size}, パス: {base_img_path}")
            
            base_img = cv2.imread(base_img_path, cv2.IMREAD_UNCHANGED)
            if base_img is None:
                print(f"Error: 台座画像の読み込みに失敗: {base_img_path}")
                return
            print(f"台座画像サイズ: {base_img.shape}")

            # 台座の結合位置を計算
            connection_points = self.find_connection_points(contour, bottom_y)
            if connection_points:
                left_point, right_point = connection_points
                print(f"結合ポイント - 左: {left_point}, 右: {right_point}")
            
            # 台座の配置と結合
            result = self.merge_base_with_outline(
                self.current_image,
                base_img,
                connection_points,
                center_x,
                bottom_y
            )
            
            # 結果の表示
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            
            print("台座合成が完了しました")
            
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

    def find_connection_points(self, contour, bottom_y, margin=10):
        """輪郭線と台座の結合ポイントを検出"""
        # 最下点付近の輪郭点を探す
        bottom_points = []
        for point in contour:
            if abs(point[0][1] - bottom_y) <= margin:
                bottom_points.append(point[0])
        
        if not bottom_points:
            return None
        
        # 左右の結合ポイントを特定
        left_point = min(bottom_points, key=lambda p: p[0])
        right_point = max(bottom_points, key=lambda p: p[0])
        
        return left_point, right_point

    def merge_base_with_outline(self, original_image, base_img, connection_points, center_x, bottom_y):
        """台座と輪郭線を結合（改善版）"""
        if connection_points is None:
            print("結合ポイントが見つかりませんでした")
            return original_image

        left_point, right_point = connection_points
        result = original_image.copy()
        
        # 台座画像のサイズ調整
        base_width = right_point[0] - left_point[0] + 60  # 余裕を持たせる
        base_height = base_img.shape[0]
        aspect_ratio = base_img.shape[1] / base_img.shape[0]
        new_width = int(base_height * aspect_ratio)
        resized_base = cv2.resize(base_img, (new_width, base_height))
        
        # 台座の配置位置を計算
        top_left_x = center_x - new_width // 2
        top_left_y = bottom_y - 10  # 少し上に配置
        
        # キャンバスのサイズを調整
        canvas_height = max(result.shape[0], top_left_y + base_height)
        canvas_width = max(result.shape[1], top_left_x + new_width)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:result.shape[0], :result.shape[1]] = result
        
        # 台座の合成
        for y in range(base_height):
            for x in range(new_width):
                ty = top_left_y + y
                tx = top_left_x + x
                if (0 <= ty < canvas_height and 0 <= tx < canvas_width and 
                    resized_base[y, x, 3] > 0):  # アルファチャンネルをチェック
                    if not np.array_equal(canvas[ty, tx], [0, 0, 255]):  # 輪郭線以外
                        canvas[ty, tx] = resized_base[y, x, :3]
        
        # デバッグ表示
        cv2.circle(canvas, (center_x, bottom_y), 5, (0, 255, 0), -1)  # 中心点
        cv2.circle(canvas, tuple(left_point), 5, (255, 0, 0), -1)  # 左端
        cv2.circle(canvas, tuple(right_point), 5, (0, 0, 255), -1)  # 右端
        
        return canvas

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
        dialog.geometry("300x250")  # サイズ調整
        dialog.transient(self.root)
        dialog.grab_set()
        
        # ダイアログの中央配置
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
            self.root.winfo_rooty() + self.root.winfo_height()//2 - 125
        ))

        # 台座オプションフレーム
        option_frame = ttk.LabelFrame(dialog, text="ニチダイ台座サイズ", padding="10")
        option_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # サイズ選択のラジオボタン
        for size in ["16mm", "14mm", "12mm", "10mm"]:
            ttk.Radiobutton(
                option_frame,
                text=f"台座 {size}",
                variable=self.base_var,
                value=size
            ).pack(anchor="w", pady=5)

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

def create_base_image(width, height, base_width):
    """台座画像の作成（実用的なバージョン）"""
    # キャンバスサイズ
    canvas_width = base_width + 60  # 余白を追加
    canvas_height = int(base_width * 0.8)  # 高さは幅の80%
    
    # 透明な背景の作成
    image = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    
    # 台座の形状を定義
    top_width = base_width
    bottom_width = base_width + 20
    top_height = canvas_height // 3
    
    # 台座の上部（四角形）
    top_left = (canvas_width//2 - top_width//2, 0)
    top_right = (canvas_width//2 + top_width//2, top_height)
    cv2.rectangle(image, top_left, top_right, (128, 128, 128, 255), -1)
    
    # 台座の下部（台形）
    pts = np.array([
        [canvas_width//2 - top_width//2, top_height],  # 左上
        [canvas_width//2 - bottom_width//2, canvas_height-1],  # 左下
        [canvas_width//2 + bottom_width//2, canvas_height-1],  # 右下
        [canvas_width//2 + top_width//2, top_height]  # 右上
    ], np.int32)
    
    cv2.fillPoly(image, [pts], (128, 128, 128, 255))
    
    # 輪郭線を追加
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
