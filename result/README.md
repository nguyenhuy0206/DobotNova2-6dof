# Result: Astra picks the green cube by itself (12 Nov 2026)

`astra_green_cube_20261112.mp4` (5 min 08 s, real time, captions burned in; the same captions as `.srt`)

The clip shows the first successful pick in which **Astra, the AI planner, worked out the grasp on its own**. Nobody gave it coordinates, angles or a recipe. It was only told to practise gripping a cube and putting it on the metal table. During the run it:
- chose the green cube;
- learned how the wrist camera image moves when the gripper moves;
- turned the gripper to line up with the cube's edges;
- tried three times, using what it learned after each attempt.

Run details:
- **Episode:** `20261112-181222-0f8c62`, live mode, model `gpt-6-astra`.
- **Result:** success. The green cube was moved onto the metal table and the overhead camera confirmed it.
- **Recording:** the phone recording starts at 18:13:14, 51 s after the episode started. The times below are the video time, matched to the episode log.

| Video | What Astra does |
|---|---|
| 0:00 | Has already looked with both cameras and chosen the green cube. Its plan: grip 8 mm below the top and use the wrist camera to centre the cube first |
| 0:06–0:48 | **Learns the wrist camera.** Makes small test moves (10–15 mm) and looks after each one to see how the cube shifts in the image |
| 0:48–1:09 | **Lines up the fingers.** Turns the gripper +10°, then −40°, until the fingers face the cube's sides (final angle 70.9°) |
| 1:09 | Saves a note: moving 10 mm sideways shifts the cube about 56 px in the wrist image |
| 1:13–1:39 | Centres the cube between the fingers using that rule |
| 1:39–1:50 | **Trial 1:** lowers in 19 mm steps |
| 1:50 | The fingers touch the cube at z = 62.8 mm. The safety layer stops the arm and backs it off. Astra records the trial, lifts and shifts 15 mm |
| 2:06–2:36 | **Trial 2:** lowers to about 8 mm below the top and closes. The gripper reports "object caught" |
| 2:36–3:17 | Lifts 15 mm as a test. Both cameras show the cube is **still on the box**, so trial 2 failed. Astra records why |
| 3:17–3:51 | **Trial 3:** opens, lifts, shifts another 15 mm and lowers again |
| 3:51 | Closes: "object caught" (grip 776) |
| 3:55–4:22 | Lifts 15 mm as a test. The wrist and overhead cameras **both show the cube in the gripper** |
| 4:22 | Saves the lesson: gripper turned −30°, grip 8 mm below the top, centre with a +15 mm shift |
| 4:22–5:08 | Lifts to the transfer height in steps, checking the wrist camera each time |
| after the clip | Carries the cube to (−230, −440), lowers it, releases it 13 mm above the table, lifts, and confirms with the overhead camera that the cube is on the table |

Why this run matters:
- Its settings became the built-in skills `pick_block_wrist` and `place_block_on_table` (`src/agent/pick_place_skills.py`): rz −30°, grip 8 mm below the top, and the wrist-pixel target with its scale.
- In those skills, a whole pick is **one** AI request instead of the many requests seen here.
- This run used the original harness: 95 AI requests and about 4.9M tokens for one cube.
- With the token-efficient harness, the skill-based pick of 13 Nov (episode `20261113-173033`) took 1 request (about 6.3k tokens).

Notes:
- The safety stop at 1:50 is the arm's contact sensing working as intended. It is not a crash.
- The original iPhone file (`IMG_5748.MOV`, 320 MB) stays local and is git-ignored. The published MP4 has no audio and no metadata (no phone model, time or location).
