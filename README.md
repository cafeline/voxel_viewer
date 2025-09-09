# Voxel Viewer with Comparison

Open3Dを使用した高速なボクセル可視化ツール。2つのMarkerArrayを比較して一致・不一致を色分け表示します。

## 機能

- **高速レンダリング**: Open3Dによる効率的な3D描画
- **比較可視化**: 2つのMarkerArrayの差分を視覚的に表示
  - 🟢 **緑**: 一致する点（両方に存在）
  - 🔴 **赤**: 不一致の点（片方のみに存在）
- **半透明表示**: 重なりを見やすくする半透明レンダリング
- **1回のみの受信対応**: データは1度受信したら保持

## 使用方法

### 1. 起動

```bash
# ビルド
colcon build --packages-select voxel_viewer
source install/setup.bash

# Viewerを起動
ros2 launch voxel_viewer voxel_viewer.launch.py
```

### 2. テスト実行

```bash
# 別ターミナルでテストデータを発行
ros2 run voxel_viewer test_comparison

# 期待される結果:
# - 緑: 512点（8x8x8の重複領域）
# - 赤: 488点（非重複領域）
```

### 3. 実際のデータで使用

```bash
# decompressed_viewerと同時起動
ros2 launch voxel_viewer demo_with_decompressed.launch.py
```

## パラメータ

- `voxel_size`: ボクセルサイズ (default: 0.1)
- `point_size`: 点の描画サイズ (default: 5.0)
- `tolerance`: 点の一致判定の許容誤差 (default: 0.001)
- `background_color`: 背景色 [R,G,B] (default: [0.1, 0.1, 0.1])
- `show_axes`: 座標軸表示 (default: true)

## トピック

### Subscribe
- `/occupied_voxel_markers` (MarkerArray): 占有ボクセル
- `/pattern_markers` (MarkerArray): パターンマーカー

## 比較アルゴリズム

1. 両方のMarkerArrayを受信
2. 点をボクセルグリッドに丸め込み
3. 集合演算で一致・不一致を判定
4. 色分けして表示

## 操作方法

- **マウス左ドラッグ**: 回転
- **マウス右ドラッグ**: 移動
- **スクロール**: ズーム
- **R**: 視点リセット

## 注意事項

- MarkerArrayは1度ずつしか送信されません
- 両方のデータが揃ってから比較表示されます
- Open3Dでは真の透明度設定が難しいため、色の明度で半透明を表現