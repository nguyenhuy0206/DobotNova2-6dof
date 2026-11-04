# Dobot Nova 2 Control

Giao diện điều khiển cục bộ dành riêng cho Nova 2, 6 trục, trên macOS.

## Khởi động

```sh
cd nova2_control
chmod +x run.sh
./run.sh
```

Mở <http://127.0.0.1:8765>. Controller phải ở TCP/IP secondary-development
mode và Mac dùng Ethernet `192.168.5.100/24`; robot LAN1 là `192.168.5.1`.

Ứng dụng chỉ lắng nghe localhost. Nó không quét mạng, không nhận lệnh thô từ
trình duyệt, và giới hạn motion thành bước khớp tương đối 0.1–2° ở tốc độ tối
đa 20%. Các vị trí do bạn lưu được ghi vào `positions.json`.
