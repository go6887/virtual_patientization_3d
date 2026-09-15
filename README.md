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
  cli.py          実機・動画・デモの起動
```

```bash
uv run pytest
uv run ruff check src test
```

テストは5面の姿勢復元、複数面の外れ値、重複・未知ID、検出ロスト、座標変換、校正値の復元を含みます。合成デモとRerun記録を確認しています。実物を用いた追跡精度・カメラ校正の検証は別途必要です。

各オプションは `uv run probe-tracker --help`、`uv run probe-tracker track --help` で確認できます。`uv run python src/main.py demo` も使用できます。

実装時の参照: [OpenCVのPnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)、[カメラ校正](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html)、[Rerun Transform3D](https://rerun.io/docs/reference/types/archetypes/transform3d)、[Pinhole](https://rerun.io/docs/reference/types/archetypes/pinhole)。
