"""Render the measured comparison and copy compact deliverables into docs."""
from pathlib import Path
import json,sys,shutil,csv
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from perception_mediapipe.summarize import plt,savefig,NAMES,COLORS,MODELS,REPORT
speed=json.load(open(root/'.runtime/comparison/speed.json'));s=json.load(open(REPORT/'summary.json'));segments=json.load(open(REPORT/'segment_metrics.json'))
fig,axs=plt.subplots(1,2,figsize=(12.8,5))
for i,k in enumerate(MODELS):
 v=speed['models'][k];axs[0].bar(i,v['median_ms'],color=COLORS[k]);axs[0].plot([i,i],[v['median_ms'],v['p95_ms']],color='#222',lw=1.5);axs[0].scatter([i],[v['p95_ms']],marker='_',color='#222');axs[0].text(i,v['p95_ms']+1,f"{v['median_ms']:.1f} / {v['p95_ms']:.1f}",ha='center')
 axs[1].bar(i,v['measured_processing_hz'],color=COLORS[k]);axs[1].text(i,v['measured_processing_hz']+1,f"{v['measured_processing_hz']:.1f}",ha='center')
for ax in axs:ax.set_xticks(range(3));ax.set_xticklabels([NAMES[k] for k in MODELS],fontsize=9)
axs[0].set_title('同一批 128 张图：RGB-D 已到达 → 原始位姿');axs[0].set_ylabel('耗时 ms：柱为中位数，黑线至 P95');axs[0].set_ylim(0,60)
axs[1].set_title('不含相机等待、GUI、机器人控制的处理速率');axs[1].set_ylabel('实测串行处理帧数 / 秒');axs[1].set_ylim(0,60);axs[1].axhline(30,ls='--',lw=1,color='#444');axs[1].text(.05,31,'当前相机采集上限 30 FPS',fontsize=9)
fig.suptitle('MediaPipe 明显减少计算耗时，但不能把处理速率直接当成相机帧率或闭环延迟',fontsize=13)
fig.tight_layout();savefig(fig,'04_处理速度')
lines=['# HaMeR 与 MediaPipe 位姿对照结果','',
'本轮已完成工程副本、两种 MediaPipe 运行入口，以及同源 RGB-D/IMU 对照。原工程和封存版本未改。',
'','**结论：这批数据中，HaMeR 的总体转角一致性略好；MediaPipe 3D 明显更快，部分左移误转改善，但主动俯仰、横滚会被低估；直接用二维关键点取深度构建掌部姿态，当前实现整体更不稳定。没有一种方法在所有指标上胜出。**','',
'## 数据与比较条件','',
'- 录像一：`hand_motion_observation_20260911T153019_6Wdm`，66 段用户标注。',
'- 录像二：`hand_motion_observation_20260911T202058_Ime9`，13 段用户标注。',
'- 共 4,054 个评估时刻，4,053 个取得同步 RGB-D。按原始 RGB 帧号和源时间匹配，深度与 RGB 时间差不超过半个相机周期；保留失败记录。',
'- 三种方法使用同一帧 RGB-D、相同的 MediaPipe 检测配置、相同评估时刻。HaMeR 重新推理，不是用旧结果冒充本次计算；HaMeR 形状参数沿用该录像记录的参数。',
'- 精度表比较的是**原始位姿估计：关闭掌面辅助，尚未进入共用的 SO(3)/ROS 控制滤波**。这是模型与坐标构建方式的比较，不是旧版带辅助的完整控制链与新版之间的抓取成功率比较。',
'- MediaPipe 两种模式共用一个 Hands Full 模型。RGB-D 模式用二维关键点与真实深度构建掌部方向；3D 模式用预测的三维关键点构建方向。两种模式都用真实深度定位腕点 0。',
'- 测速：128 张共同可识别的图像，预先读入内存，5 次预热后计算；CPU 8–11，i5-12400F，RTX 2070。三个模式都重新运行检测，HaMeR 的额外网络和几何计算也计入；不计磁盘、相机等待、GUI 和机械臂控制。','',
'## 总体结果','',
'| 指标 | HaMeR，无辅助 | MediaPipe + RGB-D | MediaPipe 3D |','|---|---:|---:|---:|']
for label,field,fmt in [('跨时刻总转角差 RMSE，° ↓','macro_rmse_deg','.2f'),('静止位置散布 RMS，mm ↓','static_position_spread_rms_mm','.2f'),('静止姿态散布 RMS，° ↓','static_rotation_spread_rms_deg','.2f')]:
 lines.append('| '+label+' | '+' | '.join(format(s['models'][k][field],fmt) for k in MODELS)+' |')
lines.append('| 完整位姿有效率 | '+' | '.join(f"{100*s['models'][k]['valid_frames']/s['models'][k]['frames']:.2f}%"for k in MODELS)+' |')
for label,field in [('处理耗时中位数，ms ↓','median_ms'),('处理耗时 P95，ms ↓','p95_ms'),('离线串行处理速率，Hz ↑','measured_processing_hz')]:lines.append('| '+label+' | '+' | '.join(f"{speed['models'][k][field]:.1f}"for k in MODELS)+' |')
lines+=['','总转角差的计算：对片段中的每对共同有效时刻，分别算视觉与 IMU 的相对转角，两者相减后计算 RMSE；再对 79 个片段等权平均。图中的曲线则是相对每段首个共同有效帧的总转角。','',
'静止散布相对各片段自身均值计算，再取静止片段的中位数，包含人的真实微动。腕口中心与关键点 0 并非同一个解剖位置，**这些数值不能当成绝对位置精度**。','',
'![处理速度](04_处理速度.png)','',
'## 哪类动作表现不同','',
'| 意图分组：总转角差 RMSE，° | HaMeR | MediaPipe + RGB-D | MediaPipe 3D |','|---|---:|---:|---:|']
for cat in ['平移','主动俯仰','主动横滚','主动偏航','其他／组合','静止']:
 lines.append('| '+cat+' | '+' | '.join(f"{s['models'][k]['category_macro_rmse_deg'][cat]:.2f}" for k in MODELS)+' |')
lines+=['','![误差与有效率](02_误差与有效率.png)','','不能只看总体平均，也不能只选某段表现好的左移。下面三段能看出真实取舍：','',
'| 录像与动作 | IMU 转角峰值 | HaMeR | MediaPipe + RGB-D | MediaPipe 3D |','|---|---:|---:|---:|---:|']
for lab,idx,title in [('earlier',2,'录像一 3:16–3:24，慢速左移'),('later',8,'录像二 2:03–2:06，主动俯仰'),('later',9,'录像二 2:16–2:19，主动横滚')]:
 rr={x['model']:x for x in segments if x['session']==lab and x['id']==idx};lines.append('| '+title+f" | {rr['hamer']['imu_peak_deg']:.1f}° | "+' | '.join(f"{rr[k]['model_peak_deg']:.1f}°"for k in MODELS)+' |')
lines+=['','![同源动作曲线](01_动作曲线.png)','','## 原因与适用范围','',
'- **部分左移：** HaMeR 出现明显的额外转动，MediaPipe 3D 在这些片段更接近 IMU。但其他平移片段没有同样改善，因此平移组总体只从 4.51° 变为 4.35°。',
'- **主动俯仰和横滚：** MediaPipe 的三维深度成分是网络预测，未充分恢复部分出平面转动；示例中约 53° 的俯仰只估到 35°。因此“平移时更少转”不能一概解释成识别更准确。',
'- **二维关键点 + 深度：** 关键点间距短，局部深度差会放大成方向变化；图像关键点对应关节位置，采到的却是皮肤表面深度。噪声、表面曲率和遮挡均可能影响所建坐标系。当前方法没有足够抵抗这些影响，静止方向散布约 6.20°。这只是否定当前实现，不能推广为所有 RGB-D 手姿方法都差。',
'- **位置：** 两种 MediaPipe 共用同一条腕点深度定位算法，因此位置指标相同。位置散布更小，并不能证明腕点绝对位置更准确，也不能证明人手真实弧线被消除了。',
'- **速度：** 约 46.3 ms 降为 20.1 ms，减少约 57%。约 49 Hz 是离线处理速率；相机仍是 30 FPS，当前预览仍保留 15 Hz 显示上限。机械臂控制滤波、求解和执行滞后没有因本次替换而被调小。','',
'## 证据边界','',
'- IMU 未完成独立逐轴外参标定。这次比较的是不依赖固定安装轴向的跨时刻总转角，不是欧拉角各轴误差，也不是完整绝对姿态精度。',
'- 时间对齐沿用此前估计的单一偏移，没有为了让新模型更好而重新调时间。把 IMU 对齐人为移动 ±40 ms 后，总体 RMSE：HaMeR 约 4.20–4.25°，MediaPipe 3D 约 4.65–4.68°，RGB-D 约 8.03–8.04°。本次总体排序没有改变；这并不构成硬件同步精度认证。',
'- 数据来自同一操作者的两次录制，79 段不是 79 个独立受试者，没有统计学显著性或跨人泛化结论。',
'- HaMeR 的腕口坐标与 MediaPipe 掌部坐标的构造不同，所以比较对象是三条位姿估计方案，差异不能全归因于网络本身。','',
'## 工程与运行验证','',
'- 副本：`/home/dongtian/myhandarmtest1-mediapipe-comparison-20260915`；原 `teleop-repair` 与封存版本未修改。',
'- 新前端实际打开过本机相机及预览窗口。MediaPipe 3D 通过原 UDP → ROS → 约束控制 → EGM → Gazebo 链路，回放输入使末端产生约 11.2 cm 位移；这项验证只证明链路接通，不是精准跟随或抓取验收。',
'- 两种模式的零位接口检查：同一张静止图重复输入，按 C 前无有效控制包；首个启用包与它保存的 C 参考完全一致。Gazebo 测试在 C 前仍测到约 1.53 mm、0.28° 的仿真初始微动，不能声称整台机械臂绝对静止。',
'- 保留原映射增益、零位操作、滤波参数、控制器、自碰撞、250 Hz 参考生成和场景；不向控制器提供 Gazebo 环境真值。新模式不加载 HaMeR 网络和旧掌面辅助。','',
'## 建议','',
'若当前重点是低延迟、检验更轻的视觉前端，优先试副本中的 **MediaPipe 3D**；若当前更重视俯仰、横滚的转角还原，原 **HaMeR** 在这批数据中仍更好。**当前二维关键点 + RGB-D 方向方案不建议作为正式控制默认项。** 本次没有自动混合两种模型，也没有修改增益去掩盖角度低估。','',
'运行方式见工程根目录 [README_MEDIAPIPE.md](../../README_MEDIAPIPE.md)。逐段数据见 [逐动作对照.csv](逐动作对照.csv)，图的 PDF 和机器可读 summary.json 同目录保存。','',
'参考输出定义：[MediaPipe Hands 官方文档](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)；[RealSense 官方离线对齐示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-with-software-device.py)。']
(REPORT/'对比报告.md').write_text('\n'.join(lines)+'\n')
json.dump(speed,open(REPORT/'speed.json','w'),indent=2)
with (REPORT/'总体指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','rotation_rmse_deg','static_position_spread_mm','static_rotation_spread_deg','valid_rate','median_ms','p95_ms','processing_hz'])
 for k in MODELS:
  a=s['models'][k];b=speed['models'][k];w.writerow([k,a['macro_rmse_deg'],a['static_position_spread_rms_mm'],a['static_rotation_spread_rms_deg'],a['valid_frames']/a['frames'],b['median_ms'],b['p95_ms'],b['measured_processing_hz']])
doc=root/'docs/mediapipe_comparison_20260915';doc.mkdir(parents=True,exist_ok=True)
for f in REPORT.iterdir():
 if f.is_file():shutil.copy2(f,doc/f.name)
print(doc)
