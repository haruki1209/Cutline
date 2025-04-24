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
        self.work_image = None  # 作業用画像（RGBA）
        self.outlined_image = None
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

        # GUI レイアウト
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # 左側：アップロード＆操作
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)

        # 画像アップロード
        self.upload_frame = ttk.LabelFrame(self.left_frame, text="画像アップロード", padding="20")
        self.upload_frame.grid(row=0, column=0, sticky="nsew")
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
        self.drop_area.grid(row=0, column=0, padx=20, pady=(20,5), sticky="nsew")

        self.select_btn = ttk.Button(
            self.upload_frame,
            text="ファイルを選択",
            command=self.select_file,
            style="Upload.TButton"
        )
        self.select_btn.grid(row=1, column=0, pady=(0,20))

        # 操作フレーム
        self.operation_frame = ttk.LabelFrame(self.left_frame, text="操作", padding="20")
        self.operation_frame.grid(row=1, column=0, pady=(10,0), sticky="ew")
        for i in range(3):
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
            text="選択中の台座: 16mm",
            font=("Helvetica", 9),
            foreground='#666666'
        )
        self.selected_base_label.grid(row=1, column=0, columnspan=3, pady=(5,0))

        # 台座オプション（ラジオ＆実行）
        self.base_options_frame = ttk.Frame(self.operation_frame)
        for size in ["16mm", "14mm", "12mm", "10mm"]:
            rbtn = ttk.Radiobutton(
                self.base_options_frame,
                text=f"台座 {size}",
                variable=self.base_var,
                value=size,
                command=self.update_selected_base_label
            )
            rbtn.pack(anchor="w", pady=5)
        self.apply_base_btn = ttk.Button(
            self.base_options_frame,
            text="合成実行",
            command=self.combine_base
        )
        self.apply_base_btn.pack(pady=5)
        self.base_options_frame.grid_remove()

        # 右側：結果表示
        self.result_frame = ttk.LabelFrame(self.main_frame, text="処理結果", padding="20")
        self.result_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
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
        self.result_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.result_label.bind('<MouseWheel>', self.on_mousewheel)
        self.result_label.bind('<Button-4>', self.on_mousewheel)
        self.result_label.bind('<Button-5>', self.on_mousewheel)

        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_area.dnd_bind('<<DragEnter>>', self.handle_drag_enter)
        self.drop_area.dnd_bind('<<DragLeave>>', self.handle_drag_leave)

    def toggle_base_options(self):
        if self.base_options_frame.winfo_ismapped():
            self.base_options_frame.grid_remove()
        else:
            self.base_options_frame.grid(row=2, column=0, columnspan=3, pady=(10,0))

    def update_selected_base_label(self, *args):
        self.selected_base_label.configure(text=f"選択中の台座: {self.base_var.get()}")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp"),("All","*.*")])
        if path:
            class E: pass
            e = E(); e.data = path
            self.handle_drop(e)

    def handle_drop(self, event):
        fp = event.data.strip('{}').strip('"')
        try:
            img = Image.open(fp)
            self.current_image = img
            # work_imageを初期化（RGBA形式）
            self.work_image = img.convert("RGBA")
            self.update_image_display(img)
            self.outline_btn.configure(state="normal")
            self.combine_btn.configure(state="normal")
            self.output_btn.configure(state="normal")
        except Exception as e:
            print("Error loading:", e)

    def handle_drag_enter(self, event):
        self.drop_area.configure(relief="sunken")
    def handle_drag_leave(self, event):
        self.drop_area.configure(relief="solid")

    def update_image_display(self, pil_img):
        self.current_display_image = pil_img
        w,h = pil_img.size
        zw,zh = int(w*self.zoom_factor), int(h*self.zoom_factor)
        photo = ImageTk.PhotoImage(pil_img.resize((zw,zh), Image.Resampling.LANCZOS))
        self.result_label.configure(image=photo, text="")
        self.result_label.image = photo

    def on_mousewheel(self, event):
        if not self.current_display_image: return
        if hasattr(event,'delta'):
            self.zoom_factor *= 1.1 if event.delta>0 else 0.9
        else:
            self.zoom_factor *= 1.1 if event.num==4 else 0.9
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
        self.update_image_display(self.current_display_image)

    def get_binary_mask(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if np.mean(mask)>127:
            mask = cv2.bitwise_not(mask)
        return mask
    
    def create_outline(self):
        try:
            # 1. 元画像取得＆二値マスク
            img_bgr = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            binm = self.get_binary_mask(img_bgr)

            # 2. モルフォロジーでリング状の輪郭抽出
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.gap+1, 2*self.gap+1))
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(self.gap+self.thickness)+1, 2*(self.gap+self.thickness)+1))
            ring = cv2.subtract(cv2.dilate(binm, k2), cv2.dilate(binm, k1))
            
            # 3. work_imageに赤色で輪郭線を描画
            work_np = np.array(self.work_image)
            work_bgr = cv2.cvtColor(work_np, cv2.COLOR_RGBA2BGR)
            work_bgr[ring == 255] = (0, 0, 255)  # 輪郭線部分を赤色に設定
            self.work_image = Image.fromarray(cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGBA))
            
            # 4. 表示を更新
            self.update_image_display(self.work_image)

        except Exception as e:
            print("Error in create_outline:", e)
            traceback.print_exc()

    def combine_base(self):
        try:
            # 1. work_imageから処理を開始
            work_np = np.array(self.work_image)
            
            # 2. 元画像から輪郭を取得
            img_bgr = cv2.cvtColor(work_np, cv2.COLOR_RGBA2BGR)
            mask = self.get_binary_mask(img_bgr)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main = max(cnts, key=cv2.contourArea)

            # 3. トリミング
            x, y, w, h = cv2.boundingRect(main)
            crop = img_bgr[y:y+h, x:x+w]

            # 4. 二値マスク(トリミング後)
            binm_crop = self.get_binary_mask(crop)

            # 5. 足元ライン関連
            y_max = max(pt[0][1] for pt in main)
            delta = 5
            feet_pts = [pt[0] for pt in main if pt[0][1] >= y_max - delta]
            if not feet_pts:
                feet_pts = [pt[0] for pt in main]
            x_left = min(p[0] for p in feet_pts)
            x_right = max(p[0] for p in feet_pts)
            y_feet = y_max

            # 6. 台座準備
            key = self.base_var.get()
            fn = self.base_parts.get(key, "")
            sz = self.base_sizes.get(key, (200, 40))
            if os.path.exists(fn):
                ped = cv2.resize(cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA), sz)
                pedestal = Image.fromarray(ped)
            else:
                pedestal = Image.new("RGBA", sz, (128, 128, 128, 255))
            pw, ph = pedestal.size

            # 7. キャンバス作成 (台座分だけ縦に長くする)
            cw, ch = crop.shape[1], crop.shape[0]
            W, H = max(cw, pw), ch + ph

            # 8. 補助線の位置を計算
            # "30%上" のライン(y_30)を計算
            y_30 = int(y_feet * 0.7)
            if y_30 < 0: 
                y_30 = 0  # 万一マイナスなら補正

            # 30%のラインの間で最も左端と右端の点を見つける
            x_left_30 = cw  # 初期値を最大値に
            x_right_30 = 0  # 初期値を最小値に
            
            # 30%のラインから足元までの間で走査
            for y_pos in range(y_30 - y, y_feet - y + 1):
                # 各行で白いピクセルを探す
                for x_pos in range(cw):
                    if binm_crop[y_pos, x_pos] == 255:
                        x_left_30 = min(x_left_30, x_pos)
                        x_right_30 = max(x_right_30, x_pos)
            
            # 見つからなかった場合は足元ラインの左右を流用
            if x_left_30 == cw:
                x_left_30 = x_left - x
            if x_right_30 == 0:
                x_right_30 = x_right - x

            # 9. 補助線の座標を定義（元の画像の座標系に変換）
            # 台座の一番上の水平線
            horizontal_guide_pedestal = [
                (x_left_30 + x, y_feet),
                (x_right_30 + x, y_feet)
            ]
            # 左側垂直ライン
            vertical_left = [
                (x_left_30 + x, y_30),
                (x_left_30 + x, y_feet)
            ]
            # 右側垂直ライン
            vertical_right = [
                (x_right_30 + x, y_30),
                (x_right_30 + x, y_feet)
            ]

            # 10. 赤線と青線のマスクを作成
            red_mask = np.zeros(work_np.shape[:2], dtype=np.uint8)
            blue_mask = np.zeros(work_np.shape[:2], dtype=np.uint8)
            
            # 赤線の検出 (R>200, G<50, B<50)
            red_condition = (
                (work_np[:, :, 0] > 200) &  # R > 200
                (work_np[:, :, 1] < 50) &   # G < 50
                (work_np[:, :, 2] < 50)     # B < 50
            )
            red_mask[red_condition] = 255
            
            # 11. 青線を描画
            draw = ImageDraw.Draw(self.work_image)
            line_color = (0, 0, 255, 255)  # 青色
            lw = 8  # 線の太さ
            
            # 台座の水平線と垂直2本を描画
            draw.line(horizontal_guide_pedestal, fill=line_color, width=lw)
            draw.line(vertical_left, fill=line_color, width=lw)
            draw.line(vertical_right, fill=line_color, width=lw)
            
            # 12. 青線のマスクを作成
            work_np = np.array(self.work_image)
            blue_condition = (
                (work_np[:, :, 0] < 50) &   # R < 50
                (work_np[:, :, 1] < 50) &   # G < 50
                (work_np[:, :, 2] > 200)    # B > 200
            )
            blue_mask[blue_condition] = 255
            
            # 13. 交差部分を検出
            intersection_mask = cv2.bitwise_and(red_mask, blue_mask)
            
            # 14. 交差部分の輪郭を取得
            contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 15. 交差部分を紫色でマーク
            for cnt in contours:
                # 交差部分の中心を計算
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 交差部分を紫色でマーク
                    cv2.circle(work_np, (cx, cy), 5, (255, 0, 255, 255), -1)
            
            # 16. 最終的な輪郭を抽出
            final_mask = cv2.bitwise_or(red_mask, blue_mask)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 17. 最大の輪郭のみを残す
            if contours:
                outer_contour = max(contours, key=cv2.contourArea)
                outer_mask = np.zeros_like(final_mask)
                cv2.drawContours(outer_mask, [outer_contour], -1, 255, thickness=cv2.FILLED)
                
                # 18. 不要な内部の線を削除
                remove_mask = ((final_mask == 255) & (outer_mask == 0))
                work_np[remove_mask, 3] = 0  # アルファチャンネルを0に設定して透明化
            
            # 19. 結果を保存して表示
            self.work_image = Image.fromarray(work_np)
            self.update_image_display(self.work_image)

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