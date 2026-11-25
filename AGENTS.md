# FR3 ＋ Allegro 仿真基线

运行入口为本目录的 `./run.sh`。整机包位于 `reproduction/fr3/fr3_allegro_sim`，臂手配置和手部算法位于 `src/allegro_finger_teleop`。发布记录与复现实验方法见 README.md、VERSION.md。

验证回放使用 `./run.sh --replay --headless`，单物体阻抗抓取使用 `./run.sh --impedance-grasp --replay --headless`。授权模型、构建、视觉环境和操作者录制保存在本机，源码包按版本标签生成。
