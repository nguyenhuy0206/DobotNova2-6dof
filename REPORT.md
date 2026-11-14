# Báo cáo 2026-11-12 → 2026-11-13 — Astra explore + pick block (Dobot Nova 2)

Nhánh: `refactor/modular-astra-layout` (chưa commit, ~76 file thay đổi/mới). Test: **140 passed** (`.venv/bin/python -m pytest -q`, 34 s).

**Tóm tắt:**
- Pipeline chạy được từ đầu đến cuối: tự calibration camera → khám phá → gắp → đặt lên bàn kim loại → tự review và cập nhật skill/notes.
- **3/15 episode thật thành công**, gồm cả episode gần nhất `20261112-181222-0f8c62`.
- Vấn đề lớn nhất còn lại là **chi phí token**: 15,2M token / 705 request / $25,53 trong ngày.

---

## 1. System design

### 1.1 Kiến trúc tổng thể
```
scripts/run_vision.sh  ─► depth helper (sudo, :8766) + UI/Debug (:8765, 2 camera)
scripts/run_calibration.sh ─► auto_calibrate (10% speed) ─► data/memory/hand_eye_env.json
scripts/run_explore.sh ─► build_explore()
        │
        ├─ Harness (agent/harness.py) ⇄ AstraClient (OpenAI Responses API, gpt-6-astra)
        │     instructions + notes gửi lại mỗi call, prompt_cache_key, truncation=auto
        ├─ Toolkit: primitives + perception + calibration + memory tools (allowlist explore.yaml)
        ├─ Primitives (actions/primitives.py) ─► Nova2 driver (TCP 29999/30004) + PGC gripper (Modbus RTU)
        ├─ Scene/look (perception/scene.py): D455 RGB-D + wrist webcam
        ├─ RobotViewGuard: robot (tape đỏ) phải luôn thấy trong D455
        └─ Learning (agent/learning.py): episode log → reflect → skills/notes/outcome
```

### 1.2 Các module mới / thay đổi chính
| Module | Vai trò |
|---|---|
| `scripts/run_vision.sh` | 1 lệnh: kiểm tra USB → `sudo -v` → dọn port 8765/8766 → depth helper (`--watch-pid`) → chờ ≤ 40 s → mở `#debug` → chạy UI |
| `drivers/nova2.py` | `PositiveSolution`/`InverseSolution`, `SetCollisionLevel`, MovJ/MovL luôn gửi `User=0,Tool=0`, không xuất `-0.0000` |
| `actions/primitives.py` | Hệ toạ độ fingertip, guard bàn, approach zone, contact sensing, view guard, giới hạn gripper |
| `perception/robot_view.py` | `find_marker`, `RobotViewGuard`, `SelfCalibration` (record/solve/LOO/outlier) |
| `perception/auto_calibration.py` | Tự lên kế hoạch 24 điểm ứng viên, kiểm IK trước, lấy 12 điểm |
| `agent/explore.py` | `ExploreSession`: preflight, run_episode, reflect, collision level |
| `agent/learning.py` | Tool reflect chỉ cho phép ghi memory, `review_all`, `OutcomeRecorder` |
| `memory/notes.py`, `memory/workcell.py` | Notes (≤ 800 ký tự/note, được chèn vào prompt), độ cao bàn |
| `memory/skills.py` | Skill có version, `revise()` lưu lịch sử (5 bản), `delete` |

### 1.3 Hệ toạ độ (bài học quan trọng)
- Controller đang dùng **Tool 1** với offset (−20,4; −105,4; 63,2) mm. Pose trả về từ feedback/GetPose **không phải flange**, lệch khoảng 125 mm.
- **Flange thật** lấy bằng `PositiveSolution(joints,0,0)`. Fingertip = flange + 118 mm theo trục Z của flange.
- Sau `ResetRobot`, tool ngầm định bị đổi. Vì vậy **mọi lệnh MovL/MovJ đều ghi rõ `Tool=0`**. Lệnh relative được quy đổi thành MovL tuyệt đối, và phép xoay lấy tâm tại fingertip.
- Nếu FK lỗi, hệ thống **từ chối đoán** và báo lỗi, không lặng lẽ dùng giá trị fallback.

### 1.4 Lớp an toàn (động cơ và mặt bàn)
| Guard | Cách làm |
|---|---|
| Tốc độ | Astra bị chặn ≤ 10%; calibration chạy 10% |
| Mặt bàn | `table_z` = 41,96 mm (dạy bằng fingertip). Fingertip ≥ bàn + 10 mm, flange ≥ bàn + 60 mm. Kiểm **trên cả đường đi**: nội suy tuyến tính + slerp góc; lệnh joint kiểm FK mỗi 2° |
| Với tới | Kiểm `InverseSolution` trước; nếu bị từ chối thì không gửi lệnh |
| Approach zone | Dưới bàn + 120 mm: mỗi bước ≤ 25 mm, speed 3%. Robot đứng yên quá 0,8 s → coi là chạm → `ContactError` + lùi 15 mm |
| View guard | Sau mỗi lệnh di chuyển phải thấy tape đỏ trong D455 (min_area 40 px), nếu không thì **hoàn tác** lệnh |
| Gripper | Mở 800, đóng tối đa đến 500 (50%); `set_gripper` < 500 bị từ chối |
| Controller | `SetCollisionLevel(3)` |

### 1.5 Self-calibration camera môi trường
- Mỗi điểm lấy vị trí 3D của marker (median 5 frame, loại nếu nhiễu > 15 mm) ghép với flange pose, rồi giải bằng `solve_with_marker_offset`.
- Chất lượng đo bằng leave-one-out; tự loại outlier, giữ tối thiểu 8 điểm.
- **Calibration của ngày khác tự động bị cất sang file `.old`**, vì setup camera thay đổi mỗi ngày.
- `run_calibration.sh --center x,y,z --span 100 --points 12` in phần sanity check và ghi một note calibration.

### 1.6 Vòng học (episode)
1. Mỗi lần chạy là một episode, log vào `data/episodes/<id>.jsonl` kèm ảnh.
2. Sau mỗi episode, Astra **reflect** chỉ với tool memory: `record_episode_outcome`, `save_skill`, `write_note`, `list_skills`, `read_notes`.
3. Kết quả được ghi vào `data/episodes/index.jsonl` (lần chấm mới nhất có hiệu lực).
4. `--review N|ID` chấm lại các episode cũ. `--review-all` review mọi grasp episode cùng lúc (có quyền `delete_skill`).
5. Notes được chèn vào instructions ở **mọi** call. Trước 18:08 có bug: instructions chỉ được gửi ở call đầu (Responses API không mang `instructions` qua `previous_response_id`), nên các episode trước đó gần như "mù" từ lượt 2.

---

## 2. Những gì đã test

### 2.1 Calibration
| Lần | Kết quả |
|---|---|
| 1 | Timeout. Nguyên nhân: tool ngầm định đổi sau ResetRobot → đã sửa bằng `Tool=0` + kiểm tra đến nơi theo flange |
| 2 | Kém: điểm chạm giới hạn tầm với, quá ít điểm → đã thêm ứng viên thích nghi và loại outlier |
| 3 | **GOOD**, LOO 3,1 mm |
| 17:10 (đang dùng) | **GOOD**, LOO **3,9 mm**, 12 điểm. Các file cũ: `hand_eye_env.20261112-*.old.json` |

Bàn được dạy lúc 16:04 bằng cách chạm fingertip (`table_z` = 41,96 mm).

### 2.2 Episodes (15 episode thật, chưa tính 2 lần smoke test lúc 16:10)
| Episode | Thời lượng | Lượt model | Tool calls | look | Kết quả |
|---|---|---|---|---|---|
| 164350 | 20,9′* | 47 | 51 | 11 | ✅ blue lên bàn (yellow tuột) |
| 164927 | 13,4′* | 18 | 19 | 4 | ❌ đẩy rơi green/yellow |
| 171203 | 4,7′ | 53 | 55 | 9 | ❌ grip 783 nhưng block không lên |
| 171718 | 3,9′ | 42 | 60 | 18 | ❌ đóng về 500, holding=false |
| 172140 | 6,7′ | 67 | 101 | 35 | ✅ yellow lên bàn |
| 173421 | 1,9′ | 28 | 41 | 13 | ❌ trượt, yellow rơi xuống bàn |
| 173631 | 3,7′ | 52 | 54 | 18 | ❌ 2 lần thử blue trượt |
| 174024 | 0,2′ | 5 | 6 | 1 | ❌ dừng: tưởng có tay người trong vùng làm việc |
| 174329 | 1,8′ | 29 | 41 | 13 | ❌ trượt |
| 174537 | 1,7′ | 28 | 40 | 14 | ❌ trượt |
| 174731 | 1,7′ | 32 | 36 | 12 | — bị ngắt |
| 175134 | 0,9′ | 16 | 17 | 5 | ❌ contact sensing dừng đúng lúc (z 62,8) |
| 175254 | 5,5′ | 97 | 98 | 20 | ❌ 6 lần thử (4 trượt, 2 huỷ) |
| 175851 | 1,8′ | 32 | 32 | 8 | ❌ hạ chạm block, đẩy rơi |
| **181222** | 8,3′ | 95 | 94 | 39 | ✅ **green, lần thử 3** |

\* thời lượng có tính cả thời gian reflect/chờ.

- **Tỉ lệ:** 3/15 thành công. Sau bản sửa instructions (18:08) là 1/1.
- **Tool lỗi:** 0–2 mỗi episode, đều là guard chặn đúng (contact, IK, giới hạn grip). Không có lần nào robot chạm bàn hay bị lỗi motor.

### 2.3 Cài đặt thắng (181222)
- Lệch **rz −30°** so với hướng ban đầu (rz tuyệt đối 70,9°).
- Hạ **8 mm** dưới mặt trên của block.
- Căn giữa bằng ảnh cổ tay. Astra tự học được: `dy +10 mm` (hệ tool) ≈ block dịch **~56 px sang phải** trong ảnh wrist.
- Grip dừng ở 776–780 (block nằm giữa 2 ngón, không cần đóng hết). Block giữ được qua lần nhấc 15 mm, quá trình chuyển, và khi đặt ở (−230, −440, 165) → z 55.

### 2.4 Lỗi đã gặp và cách sửa
| Lỗi | Sửa |
|---|---|
| Toán fingertip lệch ~125 mm (do Tool 1) | Đo offset bằng FK, dùng flange thật |
| Dùng nhầm calibration của hôm trước | Tự cất file theo ngày |
| Timeout khi di chuyển | `Tool=0` rõ ràng, MovL tuyệt đối |
| Tape đỏ của robot bị nhận là block / block nhỏ bị lọc mất | Lọc theo z (`object_z_max = bàn + 150`), `min_area_px` = 50 |
| Hạ nhanh làm block dịch chuyển | Approach zone 25 mm / 3% + contact sensing, mở gripper 800 |
| Mất instructions từ lượt 2 | Gửi lại mỗi call (+ `prompt_cache_key`) |
| Arg JSON bị cắt làm crash / `review-all` crash | Trả lỗi về cho model, tăng output budget |
| argparse không nhận `--center -300,…` | Ghi liền `--center=-300,…` |
| `run_vision.sh` trông như bị treo | In tiến độ, chờ 40 s |

---

## 3. Hiệu năng, log, token

### 3.1 Token / chi phí (OpenAI dashboard, ngày 12/11)
| Chỉ số | Giá trị |
|---|---|
| Tổng token | **15.198.317** |
| Request | **705** (log `explore.log` ghi 718 HTTP request, tính cả retry) |
| Cache hit | 95,6% |
| Chi phí | **$25,53** (credit còn $2,54) |
| Trung bình | ~21,5k token đầu vào / request |
| Riêng episode 181222 | ≈ **4,9M token** (1/3 cả ngày), 95 lượt, ngữ cảnh cuối ~79k token |

**Nguyên nhân:**
1. `previous_response_id`: mỗi call tính tiền lại **toàn bộ lịch sử**, kể cả mọi ảnh, nên chi phí tăng theo bình phương số lượt.
2. Ảnh nằm mãi trong ngữ cảnh (39 ảnh ở 181222).
3. 94/95 lượt chỉ có 1 tool call; `look` sau gần mọi bước.
4. Skill gần như không được dùng: chỉ `hover_over_block`; hạ, căn giữa (~13 lượt), gắp và đặt đều làm tay.
5. `max_steps` = 150; review lặp nhiều lần (`review-all` khoảng 32k ký tự + 13 ảnh).

### 3.2 Log và dữ liệu
| File | Nội dung |
|---|---|
| `data/episodes/*.jsonl` (17 file) + ảnh | Mỗi sự kiện start/model/tool/end/outcome |
| `data/episodes/index.jsonl` | Kết quả chấm từng episode |
| `data/logs/explore.log` (2360 dòng) | Harness + executor |
| `data/logs/nova2.log` (1404 dòng) | Lệnh TCP gửi controller |
| `data/logs/depth_helper.log` | RealSense helper |
| `data/memory/hand_eye_env.json` | Calibration 17:10 |
| `data/memory/workcell.json` | `table_z` = 41,96 |
| `data/memory/skills_explore.json` | 8 skill (+ backup `before-*`) |
| `data/memory/astra_notes.jsonl` | 28 notes (+ backup) |

### 3.3 Trạng thái memory
- **Skills:** `hover_over_block` v1 (1 ok), `descend_to_check`, `grasp_and_lift`, `raise_to_transfer_height`, `wrist_align_pick_checkpoint`, `pick_block`, `lift_retained_block_to_transfer`, `place_on_table` v2.
  - Có chồng chéo: `pick_block` / `grasp_and_lift`, và `raise_to…` / `lift_retained…`.
  - Chưa có thống kê thành công, trừ skill hover.
- **Notes:** 28 dòng, tất cả được chèn vào mọi call; có dòng trùng và các dòng `grasp_trials` cũ (rz 40,9).

---

## 4. Việc còn tồn tại & kế hoạch (chưa làm, chờ duyệt)
**Thứ tự đề xuất: P0 → P3 → P1 → P2 → P4.**

- **P0. Chi phí:**
  - Ngữ cảnh có giới hạn: bỏ `previous_response_id`; mỗi call chỉ gửi digest + khoảng 6 lượt gần nhất + ảnh mới nhất.
  - Ảnh D455 dùng `detail: low`.
  - Ghi `usage` của từng call vào log.
  - Trần token cho mỗi episode; `max_steps` 150 → 60.
  - Gộp nhiều tool call trong một lượt.
  - Mục tiêu: giảm 5–10 lần.
- **P1. Skill:**
  - Đóng gói chuỗi thắng của 181222 thành `approach_block` → (Astra tự căn giữa) → `grasp_and_lift(depth=8)` → `place_on_table`.
  - Gộp các skill trùng; kiểm bằng simulator.
- **P2. Memory:** gộp notes (≤ ~12 dòng), giữ một dòng `grasp_best`, lưu trữ các dòng `grasp_trials` cũ; review ngắn sau mỗi episode.
- **P3. Đo lường:** `scripts/episode_report.py` in thành công / số lần thử / số lượt / token / $ / thời gian cho từng episode.
- **P4. Dọn dẹp:**
  - Lỗi `zoom=[0,0,0,0]` bị hiểu thành crop 8×8.
  - Một block bị phát hiện thành 2–3 object.
  - Cân nhắc `grasp_offset_px` trong `look` cổ tay.
  - Cập nhật README, commit theo từng nhóm.

---

## 5. Cập nhật 2026-11-13 — Harness tiết kiệm token (đã làm, chưa chạy robot)
Mục tiêu tiếp theo: **pick cả 3 block lên bàn kim loại**, dựa trên lần retry thành công của episode 181222.

| Phase | Đã làm | File |
|---|---|---|
| 1. Token accounting | Ghi `usage` thật của API mỗi request (input, cached, output, reasoning, latency, số ảnh); ghi `metrics` cho mỗi episode | `agent/metrics.py` |
| 2. Ngữ cảnh có giới hạn | Bỏ `previous_response_id`. Mỗi request chỉ gồm TASK, STATE do harness đo, digest tất định (≤ 4000 ký tự), 4 lượt gần nhất và kết quả mới nhất | `agent/context.py` |
| 3. Ảnh chọn lọc | Chỉ gửi ảnh ở lần đầu mỗi camera, khi model yêu cầu, sau sự cố, hoặc khi phát hiện vật không rõ. Không gửi lại ảnh trùng. Ảnh env dùng `detail` low | `agent/vision.py` |
| 4. Action chunk | L tối đa 4 ở không gian trống, ≤ 2 khi căn chỉnh, 1 khi hạ hoặc kẹp. Dừng ở mọi lỗi hoặc ở ranh giới gripper/skill | `agent/chunking.py` |
| 5. Skill chạy cục bộ | Thêm check, routine (`descend_to`, `lift_to`, `wrist_center`) và recovery (chỉ mở gripper hoặc nhấc lên) | `actions/local_steps.py`, `skill_runner.py` |
| 6. Gọi Astra theo sự kiện | Event router tự quyết abort / dừng / gửi ảnh mà không cần gọi LLM | `agent/events.py` |
| 7. Memory | Skill mới hoặc sửa có trạng thái `candidate`, chỉ chạy live khi operator đồng ý, và lưu evidence. Có `--promote` (≥ 2 lần thành công có giám sát) và `--rollback`. Reflection: 1 lượt, ≤ 6 call | `memory/skills.py`, `explore.py` |
| 8. Đánh giá | 4 variant V0–V3, mock world tái lập được, báo cáo episode | `scripts/eval_variants.py`, `scripts/episode_report.py` |

**Skill rút ra từ 181222** (`agent/pick_place_skills.py`, trạng thái candidate):
- `pick_block_wrist`: hover ở z_top+95 → hạ tới z_top+30 → căn giữa bằng ảnh cổ tay (đáy khung ≈ (355,140) px, 5,6 px/mm theo dy, −2,6 px/mm theo dx) → hạ tới z_top−8 → kẹp → kiểm tra grip < 850 → nhấc → kiểm tra camera env thấy block đã rời chỗ.
- `place_block_on_table`: nhấc tới bàn+123 → đi tới (x,y) → hạ tới bàn+13 → mở → nhấc → kiểm tra cổ tay trống và env thấy block gần (x,y).
- Toạ độ, z_top và rz là tham số. Mỗi lần gọi là một lần thử; nếu lỗi, Astra đổi rz hoặc depth.

**Đánh giá trên mock world** (robot giả lập + Astra theo kịch bản; token là **ước lượng** theo kích thước request, không phải số bị tính phí):

| Kịch bản | Variant | Model calls | Token ước lượng | Context lớn nhất | Ảnh gửi | So với V0 |
|---|---|---|---|---|---|---|
| primitives (như 181222, 3 block) | V0 | 132 | 4.700.010 | 67.634 | 58 | 100% |
| | V1 | 132 | 553.725 | 4.395 | 2 | 11,8% |
| | V2 | 58 | 2.052.289 | 65.945 | 58 | 43,7% |
| | V3 | 58 | 270.925 | 5.076 | 2 | **5,8%** |
| skills (pick/place) | V0 | 10 | 49.463 | 6.776 | 2 | 1,1% của primitives V0 |
| | V3 | 7 | 27.490 | 4.156 | 1 | **0,6%** của primitives V0 |

Ở mọi variant, robot thực hiện cùng số lệnh chuyển động, cả 3 block đều lên bàn và không có lỗi. Test: **167 passed**.

**Còn lại:**
- Chưa chạy robot thật. Lần chạy đầu nên dùng `--mode confirm` hoặc live có giám sát: mỗi skill candidate sẽ hỏi y/N.
- Mapping px/mm và điểm đích (355,140) mới đo trên một block (green, ở z_top+30); cần xác nhận với blue và yellow.
- Kiểm tra `lifted` dựa trên pixel camera env. Nếu cánh tay che khuất, kết quả là "ambiguous" và Astra nhận ảnh để tự quyết.
- **Sự cố khi smoke test:** `build_explore` đã cất calibration của hôm qua đi và xoá 12 điểm trong `calibration_session.json`. File `hand_eye_env.json` đã được khôi phục; 12 điểm thì không còn. Đây vẫn là hành vi thiết kế cho lần chạy tiếp theo (ngày khác thì phải calibrate lại).

---

## 6. Episode thật đầu tiên với V3: `20261113-164841` và các sửa đổi

**Kết quả:** FAIL, 0 block lên bàn. 9/9 lần gọi `pick_block_wrist` đều hỏng. Episode dừng vì chạm trần token.

| Chỉ số | Giá trị |
|---|---|
| Model calls | 54 (mỗi lượt 1 tool call) |
| Token (từ API) | 404.148 = input 401.858 (cached 254.877, tức 63%), output 2.290 |
| Context lớn nhất | 8.241 token (đã có giới hạn; episode 181222 là ~79k) |
| Ảnh gửi | 11 |
| Thời gian | 6,6 phút, trong đó model mất 195 s |

**Lượt gọi bị phí:**
- `get_state`: 20 lượt, dù STATE đã có sẵn trong context.
- Nhấc tay từng bước 23 mm: 13 lượt.
- Tự gọi `look` sau mỗi lần hỏng: 11 lượt.

Chỉ có 9 lượt `run_skill` là việc thật.

**Vì sao gắp hỏng:**
1. **z_top lấy từ camera env là 87,6 mm cho cả 3 block.** Hôm qua giá trị này là 69–71 mm, và vị trí x/y của hộp vẫn như cũ. Calibration hôm nay có LOO tệ nhất 18 mm. Vì vậy skill hover và kẹp cao hơn khoảng 18 mm: chỗ thì kẹp vào không khí (500), chỗ thì kẹp vào mép trên rồi block không lên (658), chỗ thì block nằm ngoài tầm nhìn camera cổ tay.
2. **Frame cổ tay đọc trễ ngay sau chuyển động:** `wrist_sees` báo không thấy, nhưng `look` ngay sau đó lại thấy blue với diện tích 7096 px.
3. Skill v1 không tự nhấc ra khỏi approach zone. Recovery chỉ nhấc tới z_top+30. Kết quả: `move_to_pose` bị từ chối (67 mm trong zone) và Astra phải tự nhấc từng bước.
4. Bản review ghi notes "grip=unknown" vì trace của skill bị cắt ở 300 ký tự.

**Đã sửa (174 test đều qua):**
- `pick_block_wrist` v2 (tự nâng cấp ở lần chạy tới; bản v1 cùng 9 lần hỏng được giữ trong history):
  - bước đầu tiên tự nhấc ra khỏi approach zone;
  - recovery nhấc tới z_top+95;
  - precondition `z_top_plausible`: so z_top với độ cao mặt block đã dạy, lệch quá 8 mm thì từ chối và nói rõ lý do.
- `scripts/teach_table.py --block-top`: dạy độ cao mặt block bằng cách chạm đầu ngón đã đóng. Giá trị này hiện trong STATE dưới tên `block_top_z`.
- Đọc frame cổ tay: chờ 0,4 s rồi đọc lại tối đa 3 frame trước khi kết luận "không thấy".
- Harness tự chụp ảnh sau sự cố rồi gửi kèm request kế tiếp, Astra không phải tốn một lượt gọi.
- Instructions: không gọi `get_state`/`list_skills` để đọc lại STATE; pick hỏng thì gọi lại skill ngay, không tự nhấc từng bước.
- `list_skills` gọn hơn (không kèm steps; dùng `name=…` để xem đầy đủ một skill).
- Ảnh: đốm nhỏ (< 30% diện tích) và vùng trắng không bị coi là phát hiện trùng.
- Review nhận đủ trace của skill (tối đa 1500 ký tự).

**Ước tính lần tới** nếu skill chạy ổn: khoảng 12–20 lượt gọi, ~100–150k token cho 3 block.

## 7. Phân tích token các episode 13/11 (164841 → 174925) và hướng tối ưu

### 7.1 Số liệu (từ event `usage`/`metrics` trong `data/episodes/*.jsonl`)

| Episode | Lượt gọi | Input | Cached | Input không cache | Output | Ảnh | Context max | Kết quả |
|---|---|---|---|---|---|---|---|---|
| 164841 | 54 | 401,858 | 254,877 | 146,981 | 2,290 | 11 | 8,241 | 0 block, 9 lần skill v1 hỏng |
| 171355 | 16 | 118,893 | 73,905 | 44,988 | 1,359 | 13 | 8,357 | 0 block, cổ tay không thấy block |
| **173033** | **6** | 38,467 | 24,445 | 14,022 | 301 | 2 | 7,001 | **pick v3 thành công trong 1 lượt**; place báo hỏng sai |
| 174925 | 52 | 397,973 | 254,228 | 143,745 | 4,424 | 22 | 8,514 | 1 block (blue) bằng primitives |

Mỗi lượt gọi tốn 5,4k–8,5k token input. Phần cache được luôn cố định (4,809–4,927 token = instructions + tools). Phần không cache khoảng 2–3,6k token mỗi lượt. Context không phình theo số lượt (bounded hoạt động đúng). Output và reasoning chỉ khoảng 1% tổng. Một ảnh khoảng 300–450 token, tổng ảnh khoảng 2%.

**Kết luận: chi phí gần như tỉ lệ thuận với số lượt gọi.** Muốn giảm token thì phải giảm số lượt, chứ thu nhỏ từng request không đáng kể.

### 7.2 Một block tốn bao nhiêu

| Cách pick | Lượt | Input | Không cache | Model latency |
|---|---|---|---|---|
| Skill `pick_block_wrist` v3 (173033, blue) | 1 | 6,320 | 1,431 | 3 s |
| Primitives, Astra tự căn bằng look/move (174925, blue, lượt 2–39) | 38 | 290,248 | 104,466 | 163 s |

**Chênh khoảng 46 lần về input và 73 lần về phần không cache.** Ở 174925 operator trả lời N cho candidate skill (2 lần), nên Astra chuyển sang cặp `look wrist` → `move_relative`, mỗi bước 1 lượt (19 cặp). Cách này đúng như instructions cho phép, nhưng tốn đúng thứ mà `wrist_center` làm cục bộ miễn phí.

173033 là bằng chứng v3 chạy đúng trên robot thật: `wrist_search` offset [0,0] → `wrist_center` 2 vòng → `probe_top` đo được **79.5** (camera env ước 78.7) → kẹp ở 71.3 → grip 661 holding → `lifted` ok. Không cần dạy `block_top`.

### 7.3 Lãng phí đã sửa (so với 164841)
- `get_state`: 20 lượt → 0 lượt ở 174925.
- Nhấc tay khỏi approach zone: skill tự làm.
- `look` sau sự cố: harness tự chụp (171355: 10 lần auto-look, không tốn lượt).

### 7.4 Hướng tối ưu, xếp theo hiệu quả

| # | Việc | Ước tính tiết kiệm | Công sức |
|---|---|---|---|
| 1 | **Để skill chạy:** trả lời `y` cho candidate ở `--mode confirm`; thêm 1 lần thành công có giám sát rồi `--promote pick_block_wrist` (hiện 1/2) | 1 block: 38 lượt → 1–3 lượt (~95%) | Vận hành, không cần code |
| 2 | **Sửa check `wrist_clear` của place (báo hỏng sai ở 173033):** nhả xong chỉ nhấc 25 mm, block nằm ngay dưới camera nên vẫn chiếm 15,418 px > 15,000 → `still_held`, dù gripper báo "arrived (no object)" position 800. Nên lấy trạng thái gripper làm bằng chứng chính, ảnh chỉ để phụ (hoặc nhấc cao hơn trước khi check) | Tránh 3–4 lượt hồi phục mỗi block; không mất evidence để promote | Nhỏ |
| 3 | **Giới hạn lượt cho mỗi block** (vd. 12): quá thì harness báo Astra bỏ qua/ dừng, không cho servo tay 40 lượt | Chặn trường hợp xấu (~250k token) | Nhỏ |
| 4 | **Khi operator từ chối candidate:** hỏi một lần cho cả episode thay vì mỗi lượt, và thông báo không gợi ý "Use primitives" (gợi ý đó dẫn tới 38 lượt servo) mà gợi ý dừng/xin duyệt | Tránh đường đắt khi không có ý định dùng primitives | Nhỏ |
| 5 | **Sắp xếp context để cache dài hơn:** hiện STATE (đổi mỗi lượt) đứng ngay sau TASK nên cache dừng ở ~4,9k. Đổi thành TASK → EPISODE SO FAR (chỉ thêm dòng) → EARLIER TURNS → STATE → RESULTS | Phần không cache mỗi lượt ước giảm ~20–35% | Nhỏ, có test |
| 6 | **Nếu vẫn phải dùng primitives:** khi đang trong approach zone, harness gắn sẵn kết quả detect cổ tay (chỉ dữ liệu, không ảnh) vào kết quả `move_relative`, bỏ được lượt `look` riêng (không thêm tool mới) | Đường servo tay: 38 → ~20 lượt | Trung bình |
| 7 | Ảnh cổ tay `detail=low` khi servo; rút gọn tool schema | ~2% / chủ yếu giảm latency | Nhỏ, ưu tiên thấp |

**Dự kiến khi làm #1 + #2:** mỗi block khoảng 2–3 lượt (pick + place + đôi khi 1 lượt look), 3 block khoảng 10–15 lượt, **~80–110k input, ~30–40k không cache**. Con số này thấp hơn khoảng 4 lần so với 174925 (1 block) và khoảng 45 lần so với 181222 (4.9M cho 1 block, V0).

### 7.5 Chi tiết thêm về lần chạy 174925 (output cuối của run)
- **Dừng vì hết ngân sách token** (`max_episode_tokens` 400,000, đã dùng 402,397) khi đang tiếp cận green ở z ≈ 105. Giới hạn hoạt động đúng: dừng episode, không dừng robot giữa chừng một động tác.
- **38 phút chạy, nhưng chỉ khoảng 4,5 phút là làm việc thật.** Hai lần dừng dài nhất là lúc chờ operator trả lời y/N: 938 s ở bước 40 (`place_block_on_table`) và 1,052 s ở bước 41 (`pick_block_wrist`, sau đó bị từ chối). Model chỉ mất 209 s.
- **Phần tự review tốn 5 lượt, 40,750 token** (khoảng 10% của episode), và lại lưu thêm một candidate mới `grasp_aligned_cube_test_lift`.
- **Thư viện skill đang bị rác:** có 9 candidate do review tạo ra mà chưa chạy lần nào (`descend_to_check`, `grasp_and_lift`, `pick_block`, `place_on_table` v2, …). Mô tả của `wrist_align_pick_checkpoint` còn ghi "Validated" dù trạng thái là candidate. Phần gợi ý skill trong instructions (top-3, khoảng 630 token) đang kéo theo cả `pick_block`, một bản dựng lại chưa ai test, có thể làm Astra chọn nhầm.
- **Hướng sửa thêm:**
  - review chỉ được lưu skill mới khi episode có một chuỗi thành công cụ thể và chưa có skill nào làm việc đó;
  - candidate không chạy lần nào trong khoảng 2 ngày thì chuyển sang `deprecated`;
  - gợi ý skill ưu tiên skill builtin hoặc validated;
  - lời nhắc duyệt candidate chỉ hỏi một lần cho mỗi skill trong một episode.

## 8. Tổng kết ngày 2026-11-13

Nhánh `refactor/modular-astra-layout`, vẫn chưa commit. Test: **179 passed** (`.venv/bin/python -m pytest -q`, 55 s).

### 8.1 Đã làm (theo thứ tự trong ngày)
1. **Harness tiết kiệm token** (mục 5):
   - đo `usage`/`metrics` thật;
   - context có giới hạn, bỏ `previous_response_id`;
   - chỉ gửi ảnh khi cần;
   - chunking an toàn (L = 1–4);
   - event router;
   - skill chạy cục bộ có check/routine/recovery;
   - vòng đời skill candidate → validated (có `--promote`/`--rollback`);
   - so sánh V0–V3 trên mock (`scripts/eval_variants.py`).
2. **Phân tích episode thật đầu tiên với V3 (164841) và sửa** (mục 6):
   - skill tự ra khỏi approach zone;
   - recovery nhấc tới z_top+95;
   - đọc lại frame camera cổ tay;
   - auto-look sau sự cố;
   - instructions không gọi `get_state` thừa;
   - `list_skills` gọn hơn.
3. **Lỗi 429 với key mới:** project của key hết credit (`insufficient_quota`), không phải bị giới hạn tốc độ. Đã thêm `QuotaExhausted`: dừng ngay, robot stop, báo rõ lý do, không retry mãi.
4. **Episode 171355:** 11 lần pick đều hỏng vì camera cổ tay không thấy block (x/y từ camera env lệch). Thêm routine `wrist_search`: tự dò ngang ±30 mm, mỗi bước ≤ 20 mm, không thấy thì quay về chỗ cũ.
5. **Bỏ việc dạy mặt block:** chỉ dạy mặt bàn kim loại (theo yêu cầu). Thêm routine `probe_top`:
   - ngón kẹp khép ở 500, hạ từng bước ≤ 8 mm ở tốc độ 1%;
   - Z lúc cảm biến va chạm báo CONTACT = mặt block thật (`measured_top`);
   - tối đa xuống z_top − 25, không chạm thì báo `top_not_found`.

   `pick_block_wrist` v3 = search → center → probe → center lại ở measured_top+30 → kẹp ở measured_top−8 → check grip → nhấc → env xác nhận.
6. **Phân tích token 4 episode** và các hướng tối ưu xếp hạng (mục 7).

### 8.2 Episode thật hôm nay

| Episode | Lượt | Token | Kết quả / bài học |
|---|---|---|---|
| 164841 | 54 | 404k | 0 block. z_top của env sai khoảng 18 mm, có 20 lượt `get_state` thừa → sửa ở mục 6 |
| 170755, 170840, 171119 | 0 | 0 | Dừng vì 429 (key hết credit) → `QuotaExhausted` |
| 171355 | 16 | 120k | 0 block. Camera cổ tay không thấy block → `wrist_search` |
| **173033** | **6** | **39k** | **Pick v3 thành công ngay lần đầu** (measured_top 79.5, grip 661). Place báo `still_held` sai |
| 174925 | 52 | 402k | 1 block (blue) đặt lên bàn. Operator từ chối candidate nên Astra căn tay 38 lượt; dừng vì hết ngân sách 400k token; 33/38 phút là chờ prompt y/N |

So với ngày 12/11 (V0): một block thành công tốn khoảng 4.9M token. Hôm nay bằng skill chỉ mất **1 lượt, khoảng 6k token cho phần pick**.

### 8.3 Sự cố về dữ liệu
Một lần smoke test lúc chiều gọi `build_explore()` trên dữ liệu thật. Lần chạy đó đã cất `hand_eye_env.json` sang `.old` (đã khôi phục) và xoá trắng `calibration_session.json`: **mất 12 điểm calibration**, không khôi phục được. Lúc 17:07 đã calibration lại (LOO median 4.3 mm, max 12.5 mm). Từ giờ smoke test đều patch các hàm này và dùng bản sao skills tạm.

### 8.4 Việc tiếp theo
- Chạy `scripts/run_explore.sh --mode confirm`, trả lời `y` cho candidate skill. Cần thêm 1 lần pick thành công có giám sát để `--promote pick_block_wrist`.
- Sửa check `wrist_clear` của place: dùng trạng thái gripper làm chính.
- Giới hạn số lượt cho mỗi block.
- Khi operator từ chối candidate: hỏi một lần cho cả episode và không gợi ý primitives.
- Đổi thứ tự context để cache được nhiều hơn (mục 7.4).
- Nạp credit cho key trước lần chạy tới.
