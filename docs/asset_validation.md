# 同梱アセットの座標確認

2026-09-11 にリポジトリ内の資料を確認。
ユーザーが指定した Desktop のパスは存在しなかったため、`assets/` の同名資料を参照した。

- `Vscan_marker_attachment/attachment_on_probe.png` と `attachment_front_view.png` を目視確認。
- `marker_labels_32mm.pdf` の全1ページを150 dpiで描画して確認。ArUco `DICT_4X4_50`、ID 0〜4、黒枠外寸32 mm、紙40 mm、キューブ42 mm。面のIDと紙の上方向はJSONと一致。
- `Vscan_Air_CL/README_ja.txt` によるとOBJは1単位=1 mm。プローブのXは幅、Yは厚み、Zは長さ、正面は−Y、コンベックス側は+Z。
- `marker_geometry_42mm.json` の四隅は黒枠の外周。印刷時の白い余白を32 mmに含めない。

OpenCVによる照合でも、JSON内の各6×6白黒パターンは辞書のID 0〜4と完全一致。
PDFの描画画像からID 0〜4、`attachment_on_probe.png` からID 0・1・2、
`attachment_front_view.png` からID 3を検出した。

## プローブOBJとマーカーキューブ

JSONの `nominal_probe_assembly_transform` は、キューブからプローブOBJへの変換。

```text
p_probe = R @ p_cube + [0, 49, -8]  # mm

R = [-1  0  0]
    [ 0  0  1]
    [ 0  1  0]
```

したがってキューブを親にプローブOBJを配置する場合は、この変換の逆を使う。
JSONはCAD上の推定変換と明記している。実機との一致、再装着によるずれ、超音波画像面との対応は未校正。

## 印刷用アタッチメントOBJとマーカーキューブ

印刷用 `Vscan_marker_attachment.obj` はID0をベッド側に向けた別座標系。
次の変換で、JSONのキューブ座標と一致する。

```text
p_cube = diag(-1, +1, -1) @ p_print + [0, 0, 21]  # mm
```

根拠は同梱 `Vscan_marker_attachment_assembly.blend` の
`Vscan marker attachment | printable` メッシュとの頂点照合。
印刷用OBJからBlender組立座標（プローブ座標）への変換は以下。

```text
p_probe.x = p_print.x
p_probe.y = 70 - p_print.z
p_probe.z = p_print.y - 8
```

Blender 5.2.1で両メッシュの13,848頂点を照合し、最近傍頂点距離の最大値は
0.00000430 mm、平均値は0.00000112 mmだった。上記とJSONの逆変換を合成して
印刷用OBJからキューブへの変換を求めた。

この検証が保証するのは同梱CAD同士の整合性であり、実物の取付位置ではない。

## カメラなしデモの検証

`synthetic.py` は実際のArUco画像を3D四隅へ投影し、白い余白、裏面除去、
奥行き順の描画を行う。生成時の正解姿勢を追跡器へ渡さず、画像検出とPnPで姿勢を復元した。
水平画角60度、歪みなし、`demo_pose()` の0〜40秒を0.5秒間隔で検証した結果は以下。

| 解像度 | 姿勢を復元できたフレーム | 使用した面数 | 最大並進誤差 | 最大回転誤差 | 最大再投影RMS |
|---|---:|---:|---:|---:|---:|
| 1280×720 | 81 / 81 | 3 | 約2.72 mm | 約0.77度 | 約0.92 px |
| 640×480 | 81 / 81 | 2〜3 | 約5.52 mm | 約1.67度 | 約1.40 px |

これは画像の画素化を含む合成画像テストの結果。実写の精度評価ではない。
