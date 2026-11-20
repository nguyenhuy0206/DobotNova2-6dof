# Discussion log

Nhật ký các cuộc thảo luận thiết kế. Mới nhất ở trên cùng.

---

## 2026-11-20 — Thiết kế verifier (trọng tài) cho Astra

**Người tham gia:** operator, Claude Code (Opus 5.5), Codex CLI 0.170 (`gpt-6.1-sol`). Codex chạy ở chế độ read-only, đọc code, không sửa file. Trao đổi 2 vòng.
**Trạng thái:** đang thiết kế. Chưa sửa code.

### Vấn đề

- Astra hay lặp lại gần như y nguyên một bước. Lý do: feedback chỉ cho biết đạt hay không, ví dụ `grasp missed: fingers met nothing`. Câu đó không nói vì sao trượt.
  - Ví dụ 19/11, các episode 175951, 180422, 180814: 5 lần khép gripper ở đúng xy env camera báo, cả 5 lần trượt. 2 lần thành công lại nằm lệch khoảng 2 cm, ở offset (−17, −17) và (−20, 0).
- Hiện Astra vừa thực hiện task vừa tự chấm điểm, nên có nguy cơ "hack" để qua task.
- Mục tiêu: một verifier đáng tin, Astra không đọc, không sửa và không lách được, và **mở rộng được sang task sau này** (xếp chồng, phân loại, cắm, có thể cả lau hay rót). Chưa xây ngay.

### Các đường tự chấm điểm đang có trong code (đã kiểm)

| Chỗ | Vấn đề |
|---|---|
| `src/actions/local_steps.py:71` `check()` | Astra viết file cùng tên là ghi đè được check gốc (holding, lifted, placed): quy tắc "Astra's version wins" |
| `src/actions/skill_runner.py` | Astra tự chọn danh sách check của skill; `success_conditions` có thể rỗng; auto-promote đếm các lần "verified" của chính nó |
| `src/agent/explore/learning.py:89` `OutcomeRecorder.record()` | Nhận `success` thẳng từ Astra; mỗi lần review lại thì ghi đè kết quả cũ |
| `src/memory/skills.py:190` `rollback()` | Trả lại trạng thái đã duyệt cũ dù routine hay check bên dưới đã bị đổi |
| `src/student/weights/teacher.py:93` `run_python()` | Chạy Python do Astra viết dưới cùng OS user. `-I` và lọc biến môi trường **không** chặn đọc ghi file |
| `src/actions/astra_code.py` | Docstring tự ghi: sandbox "is a guard rail, not a security boundary" |
| `src/agent/events.py:97` | Kết quả skill được phân loại dựa trên kết quả do chính skill trả về |
| `src/memory/knowledge.py:131, :433` | Output của routine và `record_measurement` của Astra vào thẳng bộ nhớ |
| `src/agent/harness.py:136` `_take_image()` | Ảnh đặt tên `step-tool.jpg` nên có thể ghi đè trong một lượt; frame không có timestamp lúc chụp |
| `src/agent/repeats.py` | Chỉ bắt lặp liên tiếp, giống hệt nhau. Gọi xen kẽ hoặc đổi tham số một chút là lọt |

### Kiến trúc thống nhất

Nguyên tắc: **verifier kiểm trạng thái thế giới, không kiểm code hay hành động.** Astra giải task bằng cách nào cũng phải qua cùng một trọng tài.

```
cảm biến → [1] bằng chứng + ước lượng trạng thái → [2] predicate → [3] task spec → verdict (ledger)
                                                                             └→ [4] feedback số đo cho Astra
```

1. **Bằng chứng / trạng thái (operator sở hữu)**
   - Lưu bằng chứng gốc trước khi gộp: quan sát không sửa được, có timestamp, sequence id và nguồn.
   - Gộp thành một pose duy nhất sẽ che mất chỗ các nguồn không khớp.
2. **Predicate**
   - Trả về `true / false / unknown`, kèm số đo và id của bằng chứng.
   - **Mỗi predicate có quy tắc bằng chứng riêng**, không dùng luật chung "≥2 nguồn". Lý do: RGB và depth dùng chung calib nên có thể cùng sai; vị trí và trạng thái gripper dùng chung phần cứng.
   - Với predicate hình học: nếu cả vùng bất định nằm trọn trong điều kiện thì `true`, nằm trọn ngoài thì `false`, vắt qua ranh giới thì `unknown`.
   - Mỗi predicate phải được kiểm định (precision/recall trên episode có nhãn) trước khi dùng.
3. **Task spec (operator viết)**
   - Gồm goal, invariant, ràng buộc ban đầu và deadline.
   - Astra **được biết task là gì** (vật nào, đặt ở đâu). Chỉ giấu cách chấm và các cấu hình held-out.
4. **Feedback**
   - Chỉ chứa số đo, không chứa ngưỡng, không ra lệnh. Ví dụ: `du=+18 px, dv=−11 px, edge 7°, visibility partial`.
   - Có sổ ghi các lần thử, nhóm theo *ý định vật lý + vật + trạng thái ước lượng trước khi làm*, có dung sai. So khớp `(action, state)` tuyệt đối thì quá giòn.

**Kỳ vọng do LLM đề ra** (module Expected Outcomes của arXiv 2411.05474) có thể viết bằng chính các predicate này và được kiểm như feedback. Nhưng **chỉ goal do operator đặt mới được tính điểm.**

### Phân quyền

Mối đe dọa thực tế không phải Astra cố tình phá. Đó là áp lực tối ưu: tự sửa check cho skill qua, ghi một số đo tiện cho mình, hay promote skill nhờ một lần may mắn. Mục tiêu: **muốn đổi cách chấm phải có thao tác của operator.**

- **Judge**
  - Chạy ở process riêng, dưới user operator.
  - Code, dependency, spec, calib và ledger của judge được bảo vệ, và không import từ thư mục Astra ghi được.
  - Judge tự lấy bằng chứng qua adapter tin cậy, không nhận số đo Astra khai.
- **`run_python` của phiên teacher:** chạy dưới **OS user hạn chế**, chỉ ghi được trong thư mục work. `chmod` hay kiểm hash lúc đầu episode là không đủ, vì cùng user thì tự đổi lại được.
- **Ai gọi judge:** harness, không phải Astra. Judge chạy theo primitive thực sự được thực thi, kể cả routine lồng nhau, lỗi, timeout và crash.
- **Tính điểm:** success, promote và metrics **chỉ** đọc từ ledger. Lời khai của Astra ghi cạnh verdict, không bao giờ nằm trong các trường có thẩm quyền.
- **Duyệt skill:** gắn với digest của toàn bộ dependency (routine, check, default, calib, phiên bản verifier). Rollback không được trả lại trạng thái đã duyệt nếu digest đã khác.
- **Audit:** soi các ca Astra nói ok mà judge nói fail, và cả **mẫu ngẫu nhiên các ca được chấm đạt**.

### Xử lý `unknown`

- `unknown` không bao giờ thỏa goal.
- Vi phạm invariant đã thấy thì giữ nguyên, dù sau đó bị che khuất.
- Không chắc chắn liên quan an toàn: dừng hoặc hạn chế chuyển động ngay.
- Không chắc chắn về goal: cho nhìn lại trong giới hạn thời gian; nhìn lại không reset bộ đếm lần thử.
- Hết giờ: ghi `inconclusive`, không được điểm, **vẫn tính vào mẫu số**.
- Ca `invalid` (lỗi hạ tầng) chỉ được loại khi có bằng chứng lỗi tin cậy, và tỷ lệ này báo cáo riêng.
- Gây che khuất không bao giờ cải thiện được điểm hay xóa được bằng chứng.

### v0 cho pick-and-place

**Phạm vi:**
- 1 cube nhận diện duy nhất;
- camera cố định, bàn đã biết;
- chỗ đặt trống, không có cube cùng màu gây nhiễu;
- sai số đặt rộng.

**Predicate:**

| Predicate | Bằng chứng (cảm biến hiện có) | Lỗi chính |
|---|---|---|
| `grasp_aligned(o)`, *chỉ tư vấn, không tính điểm* | Wrist: tâm cube so với tâm ngón, theo độ cao và hướng tiếp cận. Báo **pixel** cho đến khi phép quy đổi sang mm được kiểm định | Ảnh cũ, nhầm blob, tâm tham chiếu đổi theo độ cao |
| `held_after_lift(o)` | Robot thực sự nâng; PGC báo đang cầm; wrist thấy cube giữa hai ngón sau khi nâng. Env thấy cube biến mất khỏi chỗ cũ thì **không đủ** | Cầm nhầm vật; ở gần bị hiểu là đang cầm; bằng chứng lệch thời gian |
| `placed_on_table(o, R)` | Cube nằm trọn trong **đa giác vẽ trên ảnh env** (tránh dựa vào xy D455 đang nghi lệch); gripper đã mở và đã rút ra; cube còn ở đó qua nhiều lần nhìn | Lơ lửng bị hiểu là đã đặt; nhầm vật; ảnh cũ |

Mọi predicate qua chung một cổng kiểm tra: nhận diện đúng vật, ảnh khớp kết quả, calib còn hiệu lực, thứ tự và độ mới của quan sát. Thiếu điều kiện nào thì ra `unknown`.

**Spec:**
```yaml
task: pick_place_cube_v0
object: operator_bound_cube
destination: fixed_table_polygon
initial: [object_identifiable, object_outside_destination]
required_sequence: [held_after_lift(object), placed_on_table(object, destination)]
deadline_s: 60
unknown: no_credit
```

**Ledger:**
- File JSONL do judge sở hữu, mỗi dòng gồm: `evaluation_id, episode_id, physical_attempt_id, task_hash, skill_dependency_hash, verifier_hash, calibration_hash, input_manifest_hash, predicate_results, evidence_references, status (passed|failed|inconclusive), success_credit, reason_codes`.
- Tắt auto-promote. Operator promote dựa trên các dòng ledger.

### Mở rộng sang task khác

| Task | Dùng lại nguyên | Phải thêm |
|---|---|---|
| Xếp chồng 3 cube | Ranh giới judge, ledger, provenance, logic 3 giá trị, gán vật, `held_after_lift`, máy trạng thái tuần tự | `on(a,b)`: chiều cao tương đối, độ chồng mặt đáy, đỡ được, vẫn đứng sau khi rút gripper |
| Cắm peg vào lỗ | Ranh giới judge, ledger, cổng bằng chứng, máy tuần tự | Ước lượng hình học peg và lỗ; `fully_seated`. Thẳng hàng **không** chứng minh đã cắm vào; phần khớp bị che có thể cần thêm cảm biến |

Kết luận: **task mới trong cùng một họ đã kiểm định thường chỉ cần spec.** Dùng lại code không có nghĩa là dùng lại được kiểm định: kiểm định grasp cube không áp sang grasp peg.

### Đo độ tin cậy của chính verifier

- Đo trên **phán quyết của cả task**, không chỉ độ chính xác từng predicate.
- Bộ nhãn phải có ca âm khó: nhầm vật cùng màu, lơ lửng trên chỗ đặt, rơi, chạm đồ gá, đặt rồi bị xô, bị che, ảnh cũ.
- Theo dõi tỷ lệ chấm nhầm đạt, chấm nhầm trượt, tỷ lệ bỏ phiếu trắng và độ trễ. Chia theo vùng làm việc, vật, ánh sáng và mức che khuất.
- 0 lần chấm nhầm đạt trên N ca âm độc lập thì cận trên 95% xấp xỉ 3/N. Các frame lặp lại không phải là ca độc lập.
- Trước mỗi phiên, kiểm điểm tham chiếu cố định: fiducial, khoảng cách đã biết, mặt bàn, serial, độ phân giải và intrinsics camera.
- Phát hiện drift (ví dụ camera bị xê dịch) thì vô hiệu calib và các predicate liên quan cho tới khi operator kiểm định lại.

### Giới hạn của dữ liệu cũ

- Lưu trữ có 84 episode và 831 lần `look`.
- Không có timestamp lúc chụp hay frame id. Không có depth thô, chỉ có giá trị depth đã xử lý. 25 đường dẫn ảnh bị tham chiếu lặp lại.
- Vì vậy dữ liệu cũ dùng được để kiểm logic quyết định và đo độ chính xác của `grasp_aligned`, nhưng **không** dùng để chứng nhận kết quả hay chấm điểm promote.
- Chia tập dev và eval theo episode, không theo frame.

### Để sau (quá sớm cho lab 1 người)

STL đầy đủ, scene graph tổng quát, sensor fusion xác suất, biện pháp chống suy ngược ngưỡng, auto-promote và tự động hóa kiểm định thống kê, hạ tầng theo dõi drift liên tục, chuỗi audit mật mã.

### Ghi chú khác

- Độ lệch khoảng 2 cm khi grasp **chưa chứng minh** D455 bị bias. Extrinsics, hình học TCP, độ trễ và điểm tham chiếu của grasp cũng có thể là nguyên nhân.
- Verifier task không thay thế các lớp an toàn ở primitive và controller (`self_collision`, table floor, approach zone, contact stop).

### Việc tiếp theo đề xuất (chưa làm)

1. Thu hồi quyền tự chấm:
   - judge ledger là nguồn duy nhất cho success, promote và metrics;
   - check của judge không bị ghi đè được;
   - `OutcomeRecorder` thôi nhận `success` từ Astra.
2. Gắn timestamp và sequence id cho frame, đặt tên ảnh không trùng. Rẻ nhất, và mọi bước sau phụ thuộc vào nó.
3. Viết `grasp_aligned` và replay offline trên 175951, 180422, 180814 để đo độ chính xác.
4. Cho `run_python` của phiên teacher chạy dưới OS user hạn chế trước lần chạy teacher tiếp theo.
