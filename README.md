# Vscan Air CL — Webカメラ追跡 / Rerun

プローブに装着した5面のArUcoマーカをWebカメラで検出し、プローブとアタッチメントのOBJをRerun上で動かします。カメラを固定したときの、カメラに対する位置・姿勢を表示します。

`assets/Vscan_marker_attachment/marker_geometry_42mm.json` の **DICT_4X4_50 / ID 0〜4 / 黒枠32 mm / 台42 mm** を使用します。資料のDesktopパスではなく、リポジトリ内のassetsを読み込みます。

## まずデモを動かす

Python 3.14と[uv](https://docs.astral.sh/uv/)を使用します。リポジトリのルートで実行してください。

```bash
uv sync
uv run probe-tracker demo
```

Rerunが開き、左にプローブ・アタッチメント・マーカの3D表示、右に検出画像、下に状態と再投影誤差を表示します。デモは10秒間です。カメラは使わず、移動するマーカ画像を生成して**実際のArUco検出・姿勢推定に通します**。録画のタイムラインを動かして再生できます。

```bash
# デモを記録し、あとから開く
uv run probe-tracker demo --no-viewer --save output/demo.rrd --no-realtime
uv run rerun output/demo.rrd
```

## 腎臓にビームを当てる

`--kidney` を付けると、右腎モデルを小さな立方体（ボクセル）で表示し、プローブの動きに合わせて走査断面との交差を判定します。通常起動では「腎臓の位置・大きさ」操作パネルも開きます。

```bash
# カメラなしで、現在の交差・履歴・非交差を確認
uv run probe-tracker demo --kidney

# 時間制限なしで、操作パネルから移動・拡大縮小を試す
uv run probe-tracker demo --kidney --max-frames 0

# 実物のプローブで操作（プレビュー画面のRキーで履歴を消去）
uv run probe-tracker track --camera 0 --kidney --preview

# 腎臓と走査断面の状態も録画し、あとから再生
uv run probe-tracker demo --kidney --no-viewer --save output/kidney_demo.rrd --no-realtime
uv run rerun output/kidney_demo.rrd
```

- **緑**は現在交差している立方体、**黄**は過去に交差した立方体です。現在の交差を優先して緑で表示します。
- 未走査の外形は、はっきりしたオレンジ色の枠線で表示します。内部まで判定し、立方体間の描画上の隙間は判定に影響しません。
- 状態欄に交差の有無・個数と、累積着色数・割合を表示します。割合の分母は腎臓全体のボクセル数で、現在の交差も履歴に含まれます。
- プレビュー画面を選択して **R** を押すと、現在の交差を残して累積履歴を消去します。次の有効フレームから履歴を再蓄積します。履歴は実行ごとに初期化されます。
- **LOST** の間はプローブ・走査断面・緑の現在色を消し、判定不可（`UNAVAILABLE`）と表示します。固定した腎臓と黄色の履歴は保持し、再検出すると判定を再開します。

操作パネルでは次の操作ができます。画像処理中も操作を受け付け、連続クリックの移動量を合計した最新位置・大きさを次の描画に反映します。

- **位置**：X・Y・Zの数値を入力して適用するか、各軸の＋／−ボタンで移動します。移動量は1・5・10・25 mmから選べます。
- **大きさ**：スライダー、数値入力、＋／−ボタンで**10〜500％**に変更します。100％が元の実寸で、腎臓中心を保って全方向を同じ倍率で拡大・縮小します。
- **初期配置へ戻す**：起動時の位置と倍率に戻します。着色履歴は保持します。
- **履歴を消去**：現在の交差を残して黄色の履歴を消します。
- **終了**：終了ボタンまたは操作パネルを閉じることで追跡を終了します。Ctrl+CやプレビューのQ/Escでも終了できます。

移動や拡大・縮小は表示と交差判定の両方に反映され、LOST中でも配置を変更できます。履歴は腎臓の各ボクセルに付いて動きます。拡縮時はボクセルの一辺も同じ倍率で変わり、ボクセルの個数は変わりません。位置・倍率の変更も録画され、`.rrd`のタイムラインを戻すと当時の配置で表示されます。

操作していない間、腎臓は**カメラ座標に固定**されます。カメラを固定したままプローブを動かしてください。起動時に次のオプションを指定できます。

| オプション | 初期値 | 意味 |
|---|---|---|
| `--kidney-position-mm X Y Z` | `0 -180 400` | 腎臓の境界ボックス中心。カメラ基準でX右・Y下・Z前方、単位mm |
| `--kidney-rotation-deg RX RY RZ` | `0 0 0` | 腎臓中心まわりの回転。固定軸X→Y→Zの順、単位度 |
| `--kidney-scale` | `1` | 腎臓の初期倍率（0.1〜5）。例：`1.5`で150％ |
| `--no-kidney-controls` | 未指定 | 操作パネルを開かずに実行 |
| `--voxel-mm` | `3` | 立方体の一辺（1〜10 mm） |
| `--beam-depth-cm` | `15` | レンズ曲面からの走査深さ（0より大きく24 cm以下） |
| `--beam-angle-deg` | `60` | 走査断面の表示角（0より大きく60度以下） |

```bash
uv run probe-tracker track --kidney --preview \
  --kidney-position-mm 0 -180 400 --kidney-rotation-deg 0 0 30 \
  --beam-depth-cm 15 --beam-angle-deg 60 --voxel-mm 3
```

初期配置は既存の10秒デモで交差と非交差を確認できる位置です。実物では取り付け方向や作業範囲に合わせて位置・角度を指定します。プローブ表示と判定には同じ平滑化済み姿勢を使います。

`--no-viewer`で録画する場合は操作パネルも開きません。明示的に`--kidney-controls`を付ければパネルだけ開けます。パネルはPythonのTkinterを使います。Tkinterが使えない環境では`--no-kidney-controls`を指定し、起動オプションで配置を設定してください。既に保存した録画の再生画面からは操作できないため、配置を変更するときは`track`または`demo`を起動します。

走査断面はプローブOBJのコンベックス側レンズに合わせた、厚みゼロの有限な扇形です。各立方体と面積を持って交差すると着色し、点・辺だけの接触は除外します。実際のビームの厚みや音響強度を再現するものではありません。右腎STLは `assets/kidney/` に同梱し、外部のDesktopフォルダには依存しません。`--kidney` を省略すると従来の追跡表示になります。

## Webカメラで追跡する

```bash
uv run probe-tracker track --camera 0
```

少なくとも1面のマーカ全体をカメラに見せます。複数面が見えると安定しやすくなります。Rerun画面を閉じても取得プロセスは継続するので、終了は実行元で **Ctrl+C**。`--preview` を付けるとOpenCVの小窓でも確認でき、そこで **Q / Esc** で終了できます。

- macOSでカメラを開けない場合は、システム設定 → プライバシーとセキュリティ → カメラで、実行元のターミナルまたはCodexのアクセスを確認します。外付けカメラは `--camera 1` などを試します。
- 初回は水平画角60°を仮定します。概略の動作確認用で、位置・距離の正確さは保証しません。`--fov 70` のように変更できます。
- 校正ファイル `config/camera.json` がある場合は自動で使用します。
- マーカを見失ったとき、または整合する姿勢が得られないときは **LOST** と表示し、現在の3Dプローブを非表示にします。再検出すると復帰します。
- `--smooth 0.35` が標準の平滑化係数です。`--smooth 1` で平滑化を無効にできます。画像上の軸と誤差は平滑化前、3Dモデルと位置表示は平滑化後です。

```bash
# 表示しながら録画
uv run probe-tracker track --camera 0 --save output/session.rrd

# 別の校正ファイル・取得解像度を指定
uv run probe-tracker track --camera 1 --width 1280 --height 720 --calibration config/external_camera.json

# カメラの録画ファイルや静止画を処理
uv run probe-tracker track --video path/to/video.mp4 --calibration config/camera.json
uv run probe-tracker track --image assets/Vscan_marker_attachment/attachment_on_probe.png
```

静止画の組立レンダーは検出確認用です。そのレンダーのカメラ内部パラメータは付属しないため、姿勢・距離の正解値比較には使えません。未校正設定ではIDを検出できても姿勢の品質チェックでLOSTになります。姿勢までの動作確認には `demo` を使用してください。

### LOSTが多い場合の診断

映像上部とRerunの状態欄に、直近60フレームの結果を表示します。ターミナルには2秒ごとの累計と終了時の集計も表示します。LOSTの瞬間を目で追う必要はありません。

```bash
uv run probe-tracker track --camera 0 --width 1280 --height 720 --preview --diagnostics output/tracking_diagnostics.jsonl
```

普段の動かし方で20〜30秒試し、Qで終了します。診断ファイルには各フレームの判定理由、ID、マーカの画素寸法、歪み補正後の四隅、再投影誤差を保存します。再実行時はセッションを追記します。

| 表示・ログの理由 | 意味・切り分け |
|---|---|
| NO ID / `no_markers` | 使用可能なIDがない。距離・ぼけ・反射・遮蔽などを確認。生のIDも記録するので未知ID・重複IDの除外と区別できる |
| POSE / `pose_rejected` | IDは得られたが、再投影誤差・奥行き・面の向きのいずれかが条件を満たさない。これだけで原因を校正誤差に断定はできない |
| CONFLICT / `inconsistent_faces` | 複数面が検出されたが、整合する姿勢に2面以上が同意せず、過去の姿勢でも選べない。貼付位置・IDの面配置・紙の上方向・校正値を確認 |
| CORNERS / `invalid_corners` | 使用可能な四隅になっていない（退化した形、極小領域など） |

現在の実装は1フレームの不採用でLOSTにし、履歴をリセットします。複数面の不整合が続くと、そのままでは復帰しにくい場合があります。一度1面だけを正面寄りに見せて復帰するかが切り分けになります。診断追加によって採用条件や4 pxの閾値は変えていません。

## カメラを校正する

1. チェスボードを生成し、拡大縮小なしの **100%** で印刷します。マーカラベルのPDFとは別の用紙です。

   ```bash
   uv run probe-tracker board --output output/chessboard.svg
   ```

   標準は内側の交点9×6、1マス20 mm、余白込み240×180 mmです。A4横向きで印刷でき、白黒の境界間が20 mmであることを実測します。平らで硬い板に固定してください。

2. 追跡と同じカメラ・解像度で校正を起動します。

   ```bash
   uv run probe-tracker calibrate --camera 0 --width 1280 --height 720
   ```

3. ボード全体が検出されたら **Space** で採用。画面の中央・端、距離、傾きを変え、15枚以上集めます。同じ構図の連続採用は抑制されます。
4. **C** で計算し `config/camera.json` に保存します。**Q / Esc** は保存せず終了します。再投影誤差も出力されます。
5. `uv run probe-tracker track --camera 0` で追跡します。

カメラの画角・クロップ・フォーカス設定を変更した場合は再校正してください。同じアスペクト比の単純な画像リサイズには内部パラメータを拡縮しますが、異なるアスペクト比はエラーにします。同じ比率でもクロップが変わるカメラモードには別の校正が必要です。レンズ歪みを補正した映像を検出・Rerun表示の双方に使用します。

## 座標と表示の意味

全体の単位は **mm**。カメラ座標はOpenCVの **X右・Y下・Z前方** です。推定するのはマーカキューブ中心の `camera_from_cube` です。

資料の組立変換は `probe_from_cube` なので、OBJを表示するときは逆行列を合成します。

```text
camera_from_probe = camera_from_cube @ inverse(probe_from_cube)
```

プローブOBJを原点中心に移動し直したり、メートルとして読んだりしません。プリント用アタッチメントOBJは別の造形座標なので、組立データと照合した変換を適用します。詳細は [assetsの検証記録](docs/asset_validation.md) を参照してください。

1面のときは平面PnPの候補解、複数面のときは各面の候補と非平面の組合せを評価します。再投影誤差、カメラ前方、面の向き、面同士の整合性を確認してから採用します。重複IDや未知のIDは使用しません。単面の正面付近には姿勢の曖昧さが残り、手・反射・ブレで黒枠が欠けると追跡できません。

**カメラ校正とプローブの取り付け校正は別です。** 付属JSONのプローブ位置はCAD上の推定値です。紙・接着層の厚さ、固定位置、再装着のずれは含まれません。表示位置はプローブモデルの原点であり、超音波画像面やレンズの接触点の校正値ではありません。今回の実装は追跡・表示の試作です。

## 構成と検証

```text
src/probe_tracking/
  geometry.py     添付JSONと座標変換
  tracking.py     ArUco検出・姿勢推定・平滑化
  camera.py       内部パラメータの読込と解像度整合
  calibration.py  チェスボードとカメラ校正
  synthetic.py    合成マーカ画像の生成
  viewer.py       Rerunへの形状・姿勢・映像・状態の出力
  kidney_scan.py  固定した腎臓の配置・現在の交差・累積履歴
  kidney_view.py  腎臓ボクセルと走査断面の時系列表示
  kidney_controls.py  腎臓の移動・拡大縮小・履歴消去の操作パネル
  kidney_voxels.py / ultrasound_geometry.py / probe_scan_profile.py
                  腎臓のボクセル化・有限断面との交差・レンズ形状
  cli.py          実機・動画・デモの起動
```

```bash
uv run pytest
uv run ruff check src test
```

テストは5面の姿勢復元、複数面の外れ値、重複・未知ID、検出ロスト、座標変換、校正値の復元を含みます。合成デモとRerun記録を確認しています。実物を用いた追跡精度・カメラ校正の検証は別途必要です。

各オプションは `uv run probe-tracker --help`、`uv run probe-tracker track --help` で確認できます。`uv run python src/main.py demo` も使用できます。

実装時の参照: [OpenCVのPnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)、[カメラ校正](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html)、[Rerun Transform3D](https://rerun.io/docs/reference/types/archetypes/transform3d)、[Pinhole](https://rerun.io/docs/reference/types/archetypes/pinhole)。
