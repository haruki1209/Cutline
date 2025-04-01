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
            relief="solid",  # grooveからsolidに変更
            style="Drop.TLabel"
        )
        self.drop_area.grid(row=0, column=0, padx=(20, 20), pady=(20, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ファイル選択ボタン
        self.select_btn = ttk.Button(
            self.upload_frame,
            text="ファイルを選択",
            command=self.select_file,
            style="Upload.TButton"  # 新しいスタイルを使用
        )
        self.select_btn.grid(row=1, column=0, pady=(0, 20))
        
        # 操作ボタンエリア（左側の下部に配置）
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        self.operation_frame.grid_columnconfigure(0, weight=1)
        self.operation_frame.grid_columnconfigure(1, weight=1)
        self.operation_frame.grid_columnconfigure(2, weight=1)
        
        # ボタンのスタイル設定
        style = ttk.Style()
        style.configure("Drop.TLabel", 
                       font=("Helvetica", 11),
                       foreground='#666666')  # テキストの色を灰色に
        style.configure("Upload.TButton",     # ファイル選択ボタン用の新しいスタイル
                       padding=5,
                       font=("Helvetica", 10),
                       background='white',
                       relief='solid')
        style.configure("Gray.TButton", 
                       padding=10, 
                       background='#808080')
        
        # 画像処理用の変数を追加
        self.current_image = None  # 現在の画像（CV2形式）を保持
        
        # ボタンにコマンドを追加（初期状態は無効）
        self.object_detect_btn = ttk.Button(
            self.operation_frame,
            text="輪郭線作成",
            style="Gray.TButton",
            width=15,
            command=self.create_outline,
            state="disabled"  # 初期状態を無効に
        )
        self.object_detect_btn.grid(row=0, column=0, padx=5)
        
        self.combine_btn = ttk.Button(
            self.operation_frame,
            text="台座の合成",
            style="Gray.TButton",
            width=15,
            command=self.combine_base,
            state="disabled"  # 初期状態を無効に
        )
        self.combine_btn.grid(row=0, column=1, padx=5)
        
        self.output_btn = ttk.Button(
            self.operation_frame,
            text="画像出力",
            style="Gray.TButton",
            width=15,
            state="disabled"  # 初期状態を無効に
        )
        self.output_btn.grid(row=0, column=2, padx=5)
        
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
            anchor="center",  # テキストを中央揃え
            justify="center"  # 複数行テキストも中央揃え
        )
        self.result_label.grid(
            row=0, 
            column=0, 
            sticky=(tk.W, tk.E, tk.N, tk.S),
            padx=10,  # 左右の余白
            pady=10   # 上下の余白
        )

        # ズーム関連の変数を追加
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.current_display_image = None

        # マウスホイールイベントをバインド
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)       # Windows
        self.result_label.bind('<Button-4>', self.on_mousewheel)         # Linux上スクロール
        self.result_label.bind('<Button-5>', self.on_mousewheel)         # Linux下スクロール

        # ドラッグ&ドロップの実装
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)

        # スタイル設定を追加
        style = ttk.Style()
        style.configure(
            "TLabel",
            anchor="center",  # ラベル内のコンテンツを中央に
            justify="center"  # テキストを中央揃え
        )

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
        
        # 画像のオリジナルサイズを保持
        original_size = image.size
        
        # ズーム適用（オリジナルサイズに対して）
        zoom_size = (int(original_size[0] * self.zoom_factor), 
                    int(original_size[1] * self.zoom_factor))
        zoomed_image = image.resize(zoom_size, Image.Resampling.LANCZOS)
        
        # PhotoImage形式に変換
        photo = ImageTk.PhotoImage(zoomed_image)
        
        # 画像を表示
        self.result_label.configure(image=photo, text="")
        self.result_label.image = photo  # 参照を保持

    def handle_drop(self, event):
        file_path = event.data
        # Windowsのパス形式を修正（ダブルクォートと中括弧を削除）
        file_path = file_path.strip('{}').strip('"')
        print(f"Processed path: {file_path}")  # デバッグ用
        
        try:
            # PIL形式で画像を読み込み
            image = Image.open(file_path)
            
            # PIL画像をnumpy配列に変換してからOpenCV形式に変換
            image_array = np.array(image)
            if len(image_array.shape) == 3:  # カラー画像の場合
                self.current_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:  # グレースケール画像の場合
                self.current_image = image_array
            
            # 画像表示の更新
            self.update_image_display(image)
            
            # ボタンを有効化
            self.object_detect_btn.configure(state="normal")
            self.combine_btn.configure(state="normal")
            self.output_btn.configure(state="normal")
            
            print(f"Successfully loaded: {file_path}")
            print(f"Buttons enabled")  # デバッグ用
            
        except Exception as e:
            print(f"Error loading image: {e}")

    def handle_drag_enter(self, event):
        # ドラッグ開始時の視覚的フィードバック
        self.drop_area.configure(relief="sunken")

    def handle_drag_leave(self, event):
        # ドラッグ終了時の視覚的フィードバック
        self.drop_area.configure(relief="solid")  # grooveからsolidに変更

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            # ドラッグ＆ドロップと同じ処理を実行するため、
            # イベントオブジェクトをエミュレート
            class DummyEvent:
                pass
            event = DummyEvent()
            event.data = file_path
            self.handle_drop(event)

    def create_outline(self):
        """輪郭線作成の処理を実行"""
        # デバッグ用のprint文を追加
        print(f"Starting create_outline")
        print(f"Current image: {self.current_image is not None}")
        
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
            
        try:
            # グレースケール変換と二値化（Otsu）
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # オブジェクトが白になるように調整
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)

            # パラメータ設定
            gap = 20         # オブジェクトの輪郭からの間隔
            thickness = 3    # 赤線の太さ

            # ギャップ分だけ膨張した画像
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            dilate_gap = cv2.dilate(binary, kernel_gap, iterations=1)

            # ギャップ＋赤線の太さ分だけ膨張した画像
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
            dilate_thick = cv2.dilate(binary, kernel_thick, iterations=1)

            # 差分を取って赤線を描く領域を取得
            ring = cv2.subtract(dilate_thick, dilate_gap)

            # 結果画像の作成
            result = self.current_image.copy()
            result[ring == 255] = (0, 0, 255)  # 赤線を描画

            # 結果を表示用に変換
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            # update_image_displayを使用して表示
            self.update_image_display(result_pil)

            print("輪郭線の作成が完了しました")
            
        except Exception as e:
            print(f"Error processing image: {e}")

    def on_mousewheel(self, event):
        """マウスホイールでのズーム処理"""
        if not self.current_display_image:
            return

        # Windowsの場合はevent.delta、Linuxの場合はevent.numを使用
        if hasattr(event, 'delta'):
            if event.delta < 0:  # 下スクロール（縮小）
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            else:  # 上スクロール（拡大）
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        else:
            if event.num == 5:  # 下スクロール（縮小）
                self.zoom_factor = max(self.min_zoom, self.zoom_factor * 0.9)
            elif event.num == 4:  # 上スクロール（拡大）
                self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)

        self.update_image_display(self.current_display_image)

    def calculate_image_center_and_bottom(self, binary_image):
        """画像の重心と最下点を計算"""
        # 輪郭を検出
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 最大の輪郭を使用
        main_contour = max(contours, key=cv2.contourArea)
        
        # 重心を計算
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None, None
        
        center_x = int(M["m10"] / M["m00"])
        
        # 最下点のy座標を求める
        bottom_y = max(point[0][1] for point in main_contour)
        
        return center_x, bottom_y

    def combine_base(self):
        """台座の合成処理（輪郭線を保持）"""
        print("Starting combine_base function")
        
        if self.current_image is None:
            print("Error: 画像が読み込まれていません")
            return
        
        try:
            print(f"Current image shape: {self.current_image.shape}")
            
            # グレースケール変換と二値化
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # オブジェクトが白になるように調整
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            
            # 輪郭線の作成（create_outlineの処理を再利用）
            gap = 20
            thickness = 3
            
            kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            dilate_gap = cv2.dilate(binary, kernel_gap, iterations=1)
            
            kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + thickness) + 1, 2 * (gap + thickness) + 1))
            dilate_thick = cv2.dilate(binary, kernel_thick, iterations=1)
            
            ring = cv2.subtract(dilate_thick, dilate_gap)
            
            # 重心と最下点を計算
            center_x, bottom_y = self.calculate_image_center_and_bottom(binary)
            print(f"Calculated points - center_x: {center_x}, bottom_y: {bottom_y}")
            
            if center_x is None or bottom_y is None:
                print("Error: 画像の特徴点を検出できませんでした")
                return
            
            # 結果画像の作成（輪郭線と特徴点を両方表示）
            result = self.current_image.copy()
            
            # 赤い輪郭線を描画
            result[ring == 255] = (0, 0, 255)
            
            # 特徴点を描画（輪郭線の上に重ねて表示）
            cv2.circle(result, (center_x, bottom_y), 10, (0, 255, 0), -1)  # 緑の点
            cv2.circle(result, (center_x, bottom_y), 12, (255, 255, 255), 2)  # 白い縁取り
            
            # 結果を表示
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self.update_image_display(result_pil)
            
            print(f"重心のX座標: {center_x}, 最下点のY座標: {bottom_y}")
            print("Processing completed successfully")
            
        except Exception as e:
            print(f"Error combining base: {e}")
            import traceback
            traceback.print_exc()

def main():
    root = TkinterDnD.Tk()
    # ドラッグ&ドロップを有効化
    root.drop_target_register(DND_FILES)
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 