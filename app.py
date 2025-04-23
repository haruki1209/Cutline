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
        self.current_display_image = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # モルフォロジーのパラメータ（調整可能）
        self.gap = 25
        self.thickness = 5

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

            # 2. モルフォロジーでリング状の輪郭抽出 (シンプルに輪郭だけ)
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.gap+1, 2*self.gap+1))
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(self.gap+self.thickness)+1, 2*(self.gap+self.thickness)+1))
            ring = cv2.subtract(cv2.dilate(binm, k2), cv2.dilate(binm, k1))
            
            # 3. 赤色で輪郭線を描画（太くする）
            res = img_bgr.copy()
            # カーネルサイズを大きくして膨張処理
            kernel = np.ones((3, 3), np.uint8)
            thick_ring = cv2.dilate(ring, kernel, iterations=2)
            res[thick_ring == 255] = (0, 0, 255)
            pil = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            self.outlined_image = pil
            self.update_image_display(pil)

        except Exception as e:
            print("Error in create_outline:", e)
            traceback.print_exc()

    def combine_base(self):
        try:
            # 元画像処理
            src = self.outlined_image or self.current_image
            img = np.array(src.convert("RGB"))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 二値マスク＆輪郭取得
            mask = self.get_binary_mask(img_bgr)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main = max(cnts, key=cv2.contourArea)
            
            # トリミング
            x, y, w, h = cv2.boundingRect(main)
            crop = img_bgr[y:y+h, x:x+w]
            binm_crop = self.get_binary_mask(crop)
            cropped = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
            
            # 足元ライン計算
            y_max = max(pt[0][1] for pt in main)
            delta = 5
            feet_pts = [pt[0] for pt in main if pt[0][1] >= y_max - delta]
            if not feet_pts:
                feet_pts = [pt[0] for pt in main]
            x_left = min(p[0] for p in feet_pts) - x
            x_right = max(p[0] for p in feet_pts) - x
            y_feet = y_max - y
            cl = x_left
            
            # 台座準備
            key = self.base_var.get()
            fn = self.base_parts.get(key, "")
            sz = self.base_sizes.get(key, (200, 40))
            if os.path.exists(fn):
                ped = cv2.resize(cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA), sz)
                pedestal = Image.fromarray(ped)
            else:
                pedestal = Image.new("RGBA", sz, (128, 128, 128, 255))
            pw, ph = pedestal.size
            
            # 合成画像準備
            cw, ch = cropped.size
            W, H = max(cw, pw), ch + ph
            comp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            cx = (W - cw) // 2
            px, py = (W - pw) // 2, y_feet
            comp_right_x = px + pw + 65
            
            # 上端位置の計算
            x_rel = comp_right_x - cx
            y_end_rel = y_feet
            for y_rel in range(y_feet, -1, -1):
                if 0 <= y_rel < binm_crop.shape[0] and 0 <= x_rel < binm_crop.shape[1] and binm_crop[y_rel, x_rel] == 255:
                    y_end_rel = y_rel
                    break
            
            left_foot_x = cx + cl
            right_foot_x = comp_right_x
            foot_y = y_feet
            top_y = y_end_rel
            
            # キャラクター画像をペースト
            comp.paste(cropped, (cx, 0), cropped)
            
            # ガイド線描画
            draw = ImageDraw.Draw(comp)
            pad = 15
            line_color = (255, 0, 0, 255)
            line_width = 10
            intersection_point = (comp_right_x, foot_y)
            
            # 水平線と垂直線を描画
            horizontal_guide = [(cx+cl-pad, foot_y), (comp_right_x+pad, foot_y)]
            draw.line(horizontal_guide, fill=line_color, width=line_width)
            vertical_guide = [(comp_right_x, foot_y), (comp_right_x, top_y-pad)]
            draw.line(vertical_guide, fill=line_color, width=line_width)
            
            # 交点を補強
            dot_radius = 5
            draw.ellipse([
                (intersection_point[0]-dot_radius, intersection_point[1]-dot_radius),
                (intersection_point[0]+dot_radius, intersection_point[1]+dot_radius)
            ], fill=line_color)
            
            # NumPy配列に変換して処理
            comp_np = np.array(comp)
            h_img, w_img = comp_np.shape[:2]
            
            # キャラクターマスクの作成（アルファチャネルベース）
            character_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            character_mask[comp_np[:,:,3] > 100] = 255
            
            # ―― 1) 隙間を塞いでシルエット化
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(character_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # ―― 2) Flood‑Fill で穴を埋める
            ff_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
            flood = closed.copy()
            cv2.floodFill(flood, ff_mask, (0, 0), 255)
            silhouette = cv2.bitwise_not(flood)       # ="穴埋め済み"物体領域
            
            # ―― 3) 最外周輪郭だけ取得
            contours, _ = cv2.findContours(
                silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            
            # ―― 4) 外周を描く → edge_maskに保存
            edge_mask = np.zeros_like(character_mask)
            cv2.drawContours(edge_mask, contours, -1, 255, thickness=3)  # 少し太くして確実に保護
            
            # ガイド線を保護するマスク
            guide_mask = np.zeros((h_img, w_img), dtype=bool)
            
            # ガイド線の水平部分
            for y in range(max(0, foot_y-line_width), min(h_img, foot_y+line_width)):
                for x in range(max(0, left_foot_x-pad), min(w_img, right_foot_x+pad)):
                    guide_mask[y, x] = True
            
            # ガイド線の垂直部分
            for y in range(max(0, top_y-pad), min(h_img, foot_y+line_width)):
                for x in range(max(0, right_foot_x-line_width), min(w_img, right_foot_x+line_width)):
                    guide_mask[y, x] = True
            
            # 内側領域定義（シンプルに）
            inside_area = np.zeros((h_img, w_img), dtype=bool)
            margin = 15
            for y in range(h_img):
                for x in range(w_img):
                    if y < foot_y - margin and x < right_foot_x - margin:
                        inside_area[y, x] = True
            
            # 赤線検出
            red_mask = np.zeros((h_img, w_img), dtype=bool)
            red_mask[(comp_np[:,:,0] > 230) & (comp_np[:,:,1] < 50) & (comp_np[:,:,2] < 50) & (comp_np[:,:,3] > 200)] = True
            
            # 外周輪郭をマスクとして保護
            edge_protect = edge_mask > 0
            
            # 削除マスク（赤線かつ内側領域かつ外周でなくガイド線でない）
            remove_mask = red_mask & inside_area & ~edge_protect & ~guide_mask
            
            # デバッグ用
            debug_mask = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            debug_mask[guide_mask] = [0, 255, 0]  # ガイド線保護領域は緑
            debug_mask[remove_mask] = [0, 0, 255]  # 削除対象は青
            Image.fromarray(debug_mask).save("debug_masks.png")
            
            # 削除実行
            comp_np[remove_mask, 3] = 0
            
            # 結果表示
            result = Image.fromarray(comp_np)
            self.update_image_display(result)
            
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