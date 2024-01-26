# Sensor_App

# to doリスト
    ・nが入力されたときデータを削除し、１つ前からやり直す。
    ・MACアドレス3を選択するとが何故か1のセンサーが接続（ED:60 → EF:7D）

## Bluetooth設定
`hcitool dev`<br>
システムに接続されているBluetoothデバイス（Bluetoothドングルを含む）の一覧を表示させ、利用可能なデバイスがあるか確認する。

`bluetoothctl scan on`<br>
周囲のBluetoothデバイスのスキャンを開始する。Metawerが表示されているか確認する。