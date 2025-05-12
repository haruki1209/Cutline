【1. 開発環境のセットアップ】
# 仮想環境の作成と有効化
python -m venv venv
venv\Scripts\activate

# 必要なパッケージのインストール
pip install opencv-python
pip install pillow
pip install tkinterdnd2
pip install pyinstaller

【2. 実行方法】
# 開発環境で実行
venv\Scripts\activate
python app.py

3. 実行ファイル（.exe）の作成
# 仮想環境を有効化
venv\Scripts\activate

# PyInstallerでexeファイルを作成
pyinstaller --onefile --windowed --add-data "nichidai_base_*.png;." app.py