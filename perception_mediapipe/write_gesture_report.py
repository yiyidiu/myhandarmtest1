"""Build the source-linked Chinese gesture comparison report from measurements."""
import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception_mediapipe.analyze_gestures import OUT,RESULT,REPORT,motion_metrics
from perception_mediapipe.analyze_six_groups import load,angle,dump
from perception_mediapipe.six_group_geometry import measured_points,PALM


def read(name):return list(csv.DictReader(open(REPORT/(name+'.csv'),encoding='utf-8-sig')))
def val(r,k):return float(r[k])if r.get(k)not in [None,'','None']else np.nan
def fmt(x,d=2):
    if isinstance(x,str):return x
    return f'{x:.{d}f}'if np.isfinite(x)else'不可估计'
def table(headers,rows):return '\n'+'| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x)for x in r)+' |'for r in rows)+'\n'
def weighted(rows,key):return sum(val(r,key)*int(r['n'])for r in rows)/sum(int(r['n'])for r in rows)


def main():
    summary=json.load(open(REPORT/'summary.json'));sums=read('按手势汇总指标');phases=read('逐阶段六组指标');cycles=read('完整手势周期指标')
    coupling=read('去除HaMeR整体旋转后的建轴偏转');shape=read('手指弯曲及模型形变');quiet=read('参考近静止窗口稳定性');coverage=read('有效率及失败分类')
    paired=read('腕环相对其他建轴的配对增益');clock=read('时钟对齐诊断');timing=json.load(open(REPORT/'processing_timing.json'));features=read('逐帧手指状态及配准质量')
    ann=json.load(open(OUT/'gesture_annotations.json'));confirmed=[1,2,3,4,5,6,8]
    doc=[];add=doc.append
    # Check the informative quiet closing interval and two concrete anomalies.
    d=load('s1',RESULT);t=d['t'];ids=np.flatnonzero((t>=190.5)&(t<192)&d['imu_good']&np.logical_and.reduce([d[g]['valid']for g in 'ABCDEF']))
    direct={g:motion_metrics(d,ids,g,ids[0])for g in 'ABCDEF'}
    i=int(np.argmin(abs(t-272.48413348197937)));r=d['mp'][i];cache=Path(d['meta']['frame_cache_directory']);raw=np.load(cache/r['file']);p=measured_points(np.asarray(r['detection']['xy'])[PALM],raw['depth'],d['meta']['depth_scale'],d['meta']['intrinsics'])
    x=p[1]-p[4];y=p[1:].mean(0)-p[0];y/=np.linalg.norm(y);width=np.linalg.norm(x-y*np.dot(x,y))*1000
    checks=dict(quiet_closing=dict(session='s1',start=190.5,end=192.,n=len(ids),imu_peak_deg=float(angle(d['imu'][ids]).max()),gyro_peak_deg=float(angle(d['gyro'][ids]).max()),groups=direct),
        accepted_F_extreme=dict(session='s1',t=t[i],frame=r['frame_number'],measured_points_m=p,orthogonal_width_mm=width,orientation_valid=r['groups']['F']['valid'],position_valid=r['groups']['F']['position_valid'],reason='Five-point orientation can be unreliable near an edge-on hand even when numerical depth checks pass.'),
        IMU_inconsistency=dict(session='s1',start=187.0194435119629,end=188.154438495636,fused_relative_angle_deg=37.91346897369121,gyro_relative_angle_deg=.6845860646731154,visual_B_relative_angle_deg=.7807599748833772))
    dump(REPORT/'具体片段复核.json',checks)
    add('# 张手、屈指与握拳对遥操作手部朝向的影响：A–F 无滤波回放报告\n')
    add('日期：2026-09-17。工作目录：`/home/dongtian/myhandarmtest1-mediapipe-comparison-20260915`。本次为原始 RGB-D 离线重放；没有启动实物机械臂。\n')
    add('**本次补上的核心检验是：手指改变时，模型有没有把整个手估计为转动，以及建轴方法有没有在同一模型上额外引入转动。两者分别统计。** 已完成两份原始录像的 8,259 帧 A–F 重放，包含 7 次可确认的屈指／握拳周期、1 次屈指不确定的快速俯仰周期，以及 13 段张手对照。所有失败和异常周期均保留。\n')
    add('主要发现：① 在一个参考近静止的屈指片段中，IMU 约转 1.87°，HaMeR 原始朝向最大转 9.54°，存在明显不一致；② 去掉同一次 HaMeR 预测的整体旋转后，腕环和四区域建轴仍有小幅形变偏转，五点法几乎保持不变；③ 腕环没有显示稳定的整体角度误差优势；④ MediaPipe 2D＋真实深度部分平均差异较小，但抖动和极端错误更大；⑤ 一次手势周期的 IMU 融合角与陀螺积分严重不一致，不能把它的全部差异归因于视觉。\n')
    add('**关于 A/C 的设计更正：** 本报告保留六条计算路径的原始结果，但它们不代表六种实质不同的朝向估计方法。当前固定手形参数条件下，C 的五个 MANO 关节点由与 A 相同的整体旋转生成，A/C 在理想旋转计算中只差固定基底，属于近等价读出。C 应定位为一致性对照，不能把其较小内部变化解释成独立算法的抗手指干扰优势。此前没有先检查这一结构性等价关系，是实验设计解释上的遗漏。\n')
    add('## 1. 用了哪些真实数据，哪些手势得到确认\n')
    add('两份录制会话均为 2026-09-15，读取原始 `rgbd.db3` 与 `imu_time_estimated.csv`。实际相机元数据为 RealSense D435I，RGB 640×480、约 30 Hz；深度单位 0.001 m。IMU 有效采样率约 49.64 Hz。时间以会话创建时刻为零，不是屏幕录像播放器的时间。每个选定区间内的原始相机帧全部处理，不按模型成功与否挑帧。安装传感器的准备阶段不列为手势实验。\n')
    rows=[]
    for s in summary['sessions']:
        m=s['metadata'];rows.append([s['label'],Path(m['source']).name,m['requested_frames'],s['imu_valid'],s['common_valid']])
    add(table(['会话','原始目录末级名称','重放帧','IMU 时间有效帧','A–F 朝向与 IMU 同时有效帧'],rows))
    add('最终有 **7,711 个共同有效时刻**用于全部所选区间的角度比较；其中 8 个动作周期合计 4,548 帧、7 个已确认屈指周期合计 4,317 帧。其余帧仍计入可用率和失败明细。连续视频帧不是独立被试；本次没有新增被试，也没有足够元数据支持多被试统计推断。\n')
    rows=[]
    for s,aa in ann.items():
        for cy in sorted({a['cycle_id']for a in aa if a['cycle_id']}):
            ac=[a for a in aa if a['cycle_id']==cy];a=next(a for a in ac if a['phase']=='flexed_hold');r=next(r for r in cycles if int(r['cycle_id'])==cy and r['group']=='A')
            title={1:'正面半握',2:'侧向半握',3:'横向四指闭合抓握',4:'横向抓握与保持',5:'正面半握',6:'侧向半握',7:'快速俯仰，屈指不确定',8:'侧向四指闭合抓握'}[cy]
            rows.append([cy,s,f'{min(x["start"]for x in ac):.1f}–{max(x["end"]for x in ac):.1f}',title,r['n'],fmt(val(r,'imu_excursion_deg')),fmt(val(r,'imu_gyro_mae_deg'))])
    add(table(['周期','会话','区间 / s','画面确认内容','六组共同有效帧','整周期 IMU 最大转角 / °','融合角与陀螺积分 MAE / °'],rows))
    add('“闭合抓握”指四根长指明显弯曲闭合，不要求拇指覆盖四指，不等于测过握力的紧握拳。手势标签来自助手对原始 RGB 的目视复核，边界约有 ±0.5 s 不确定性，未经过独立人工复核。周期 7 初次粗看曾标作短促屈指，逐帧复核后无法排除只是整体俯仰，因此保留在全周期姿态结果中，并从“确定屈指是否跟出”的评价中单列；原始帧没有删除。\n')
    add('每行展示周期开始张手、屈指保持、结束张手的中点原图。仅裁剪和等比例缩放，亮度、对比度均未改；帧号、裁剪范围和源文件保存在 `figures/image_provenance.json`。\n\n![周期1–4原始手势](figures/01_gesture_examples_1.png)\n\n![周期5–8原始手势](figures/01_gesture_examples_2.png)\n')
    add('## 2. 六组方法与控制变量\n')
    add(table(['组','输入及朝向构建','位置原点'],[
        ['A','HaMeR 原始整体旋转；撤销左手图像镜像，乘固定中性掌部基底','HaMeR 腕关节，经同帧实测深度放到相机系'],
        ['B','同一次 HaMeR 的 16 个腕口网格顶点，与中性腕环作鲁棒相似变换配准','16 点腕环中心，经同帧实测深度定位'],
        ['C','同一次 HaMeR 的 0、5、9、13、17 五个三维关节点读出朝向；当前条件下为 A 的近等价一致性对照','腕关节 0'],
        ['D','同一次 HaMeR 的近腕、远掌、拇指侧、小指侧四块固定网格区域，各取中心建轴','腕关节 0'],
        ['E','MediaPipe 预测的世界三维关键点，使用与 C 相同的五点建轴公式','2D 腕点＋实测深度'],
        ['F','只用 MediaPipe 的 2D 像素点＋D435I 同帧实测深度恢复五个表面点，使用 C 的建轴公式','实测腕部表面点']]))
    add('C/E/F 的纵轴为“腕点→四指根均值”，横轴为“小指根→食指根”在纵轴正交平面内的投影，第三轴为叉积。A/B/D 只用固定中性模板对齐这一轴约定，没有根据 IMU 误差调整姿态。B 的配准允许尺度变化，配准质量与绝对姿态正确性是不同概念。\n')
    add('A/C 近等价的原因已经核对到当前 MANO 运动学树：实验关节 5、9、13、17 对应 MANO 关节 1、4、10、7，它们的父节点均为腕部根节点 0；关节静态位置先由固定 β 的形状回归得到，再作运动学变换。手指在这些关节处旋转，不会移动关节中心自身。它们不是五个独立从图像估计出来的三维点。\n')
    add('令 R(t) 表示完成左手约定后的 HaMeR 整体旋转。在固定 β 和理想旋转矩阵下，A 的输出为 R(t)C₀，C 的输出为 R(t)Cβ，其中 C₀、Cβ 都是常量；因此两者的相对转角大小理论相同。当前计算存在很小的数值残余：C 相对 A 的额外变化平均约 0.0036°、最大约 0.0193°，不能据此声称独立估计收益。即使让 A 使用未经固定换基的原生 XYZ，这一相对总转角指标也不会因此改变；换坐标轴名称无法制造新的姿态信息。\n')
    add('D 每块取 12 个互不重复的固定 MANO 顶点；中性模板选择条件包括根骨蒙皮权重 >0.45、掌长归一化位置 0.05–0.88、掌宽归一化绝对位置 ≤0.60，再按四个预设解剖目标选点。包含掌侧和背侧顶点，因此这里测的是固定区域中心法，不是另加 RGB-D 掌面姿态修正。\n')
    add('**固定左手；原系统的额外掌面辅助关闭；所有姿态时域滤波、IMU 拟合姿态修正、丢帧保持关闭。** A–D 共用一次 HaMeR 推理、同一个裁剪框和每份录制已保存的固定手形参数 β；固定 β 是固定手形条件，不是时间滤波。所有组使用相同原始 RGB 帧，A–D 与 E/F 也共用 MediaPipe 检测定位入口。因此这里比较的是工程当前组合，不等于使用 HaMeR 官方默认检测器的完整系统比较。\n')
    add('F 从每个 2D 点附近 5×5 深度窗口取中值：至少 8 个有效深度、距离 0.15–2.5 m、深度中值绝对偏差不超过 12 mm；腕到指根距离须在 25–180 mm。建轴还要求掌长不少于 15 mm、正交后的掌宽不少于 10 mm。这些是实现中的检查阈值，不是自然规律，也不能保证点的位置正确。这里的深度邻域中值是单帧采样规则，不是手部姿态的时间滤波。F 的朝向代码不读取 `world` 三维预测；深度不足直接失败，不回填历史深度或预测 z。E/F 的检测网络内部会产生三维预测，但只有 E 的姿态计算使用它。\n')
    add('## 3. 零位、时间对齐与“误差”究竟指什么\n')
    add('每个完整动作周期只在开始的首个六组共同有效时刻设一次零位，屈指、握拳、重新张开期间不重置。对照段分别有自己的零位。另行输出“阶段内重设零位”的局部诊断和“整份录像只设一次零位”的结果，不与主表混用。\n')
    add('主指标为相对转角大小：`αg(t)=angle(Rg(t0)ᵀ Rg(t))`，`e(t)=αg(t)−αIMU(t)`。MAE 是 `|e|` 平均值，RMSE 是均方根，P95 是绝对差异的第 95 百分位；最大值不是每帧抖动幅度。**这衡量视觉与参考的转动幅度一致性，不是已经获得独立真值的完整三维姿态误差。** 两者转过同样角度、转轴却不同，主指标仍可能很小。\n')
    add('IMU 文件明确标为估计时间、没有相机硬件同步。采用两段预先选定的张手转动对照，分别比较 A/E 的约 0.1 s 姿态变化速度与陀螺角速度，在 ±300 ms、10 ms 网格内做互相关；每份录像将四个时移估计取中位数，六组统一使用，绝不按握拳片段或各组误差分别调整。s1 为 IMU 取相机时刻 **+20 ms**，s2 为 **+25 ms**。其余张手转动段用于检查，不用于选时移。该分辨率不代表时间同步精度；s2 验证段两种视觉估计给出 −20/+10 ms，也说明不能宣称毫秒级真同步。\n')
    add('在同一有效帧集上另外计算不加时移及 ±40/±100 ms 的参考敏感性；零偏移敏感性值与其他时移使用相同帧集，便于直接比较。没有把误差大或左右手原始判别不一致的有效帧删掉。原始识别器报告过右手时，推理仍按用户要求固定为左手。\n')
    add('纯陀螺参考按估计采样时间作梯形积分，没有根据视觉拟合零偏；长期仍会积累误差。融合角与陀螺积分来自同一 IMU，并非两套独立测量系统；两者短时接近也不能证明传感器相对手部绝无滑动。\n')
    add('## 4. 各组相对参考差异有多大\n')
    add('以下主表是 **7 个已确认屈指／抓握周期，4,317 个共同有效时刻**，按帧加权；包括参考出现异常的周期 1，因此必须与后面的参考敏感性表一起看，不能直接写成“绝对精度”。\n')
    sr=[r for r in sums if r['scope']=='confirmed_gesture_cycles'and r['gesture']=='all']
    add(table(['组','MAE / °','RMSE / °','P95绝对差异 / °','最大绝对差异 / °','按阶段等权 MAE / °','自身有效帧按阶段 MAE / °'],[[r['group']]+[fmt(val(r,k))for k in ['mae_deg','rmse_deg','p95_abs_deg','max_abs_deg','macro_mae_deg','own_valid_macro_mae_deg']]for r in sr]))
    add('按帧加权会给长保持段更高权重；按阶段等权让每个阶段权重相同。自身有效帧列保留各组额外成功帧，仍使用同一个周期零位，用来检查“只看共同成功帧”是否改变结论。这些都是描述统计，未将相邻帧当成独立样本做显著性检验。\n')
    rows=[]
    for state,title in [('open','张开'),('half','半握保持'),('closed','四指闭合保持'),('closing','正在屈指'),('opening','正在重新张开')]:
        rr=[next(r for r in sums if r['scope']=='confirmed_gesture_cycles'and r['gesture']==state and r['group']==g)for g in 'ABCDEF']
        rows.append([title,rr[0]['common_frames']]+[fmt(val(r,'mae_deg'))for r in rr])
    add('再按这 7 个周期内的实际手势状态拆开，以下都是同一周期零位下的 MAE，单位为度：\n')
    add(table(['状态','共同有效帧','A','B','C','D','E','F'],rows))
    add('状态表包含周期 1 的参考异常。半握和闭合时的相机视角、真实转腕幅度及零位间隔并不相同，因此不能把“某状态平均差异更小”解释成“这种手势本质上更容易测准”。后面的短时近静止检验和同模型内部形变分解，专门减少这类混杂。\n')
    add('完整周期结果如下。周期 1 的 IMU 参考异常和周期 7 的标签不确定已直接标注，不能仅用颜色把它们当成视觉失败。\n\n![所有周期的角度差异](figures/02_cycle_angle_error.png)\n')
    add('**参考异常的具体证据：** s1 的 187.019→188.154 s，IMU 融合姿态变了 37.91°，纯陀螺积分只变了 0.68°，腕环视觉只变了 0.78°。整个周期 1，融合角与陀螺积分平均差 23.36°；B 对融合角的 MAE 为 20.94°，对陀螺积分却只有 2.99°。这不能解释为已经证明视觉错了 20°，也不能据此把陀螺积分认作独立真值；传感器融合、安装和参考一致性需要额外核实。\n')
    rows=[]
    for g in 'ABCDEF':
        rr=[r for r in cycles if r['group']==g and int(r['cycle_id'])not in[1,7]]
        rows.append([g,fmt(weighted(rr,'mae_deg')),fmt(weighted(rr,'gyro_reference_mae_deg'))])
    add('把参考异常周期 1 单列，并将屈指不确定周期 7 单列后，剩余 6 个确认周期共 **3,930 个共同有效时刻**。这是明确标出的敏感性结果，不替代保留全部数据的主结果。\n')
    add(table(['组','相对 IMU 融合角 MAE / °','相对纯陀螺积分 MAE / °'],rows))
    add('排名还依赖动作：B 在周期 5 优于 E，E 在周期 2、3、6 更小；F 在多数周期平均差异较小，但在周期 8 最大角度差异达到 84.58°。因此没有一种方法在所有维度和所有姿态上均占优。13 段张手对照的按帧 MAE 也单独列出，不能直接把其与握拳表的差值解释成“握拳造成的因果增量”，因为动作轨迹和零位条件不同。\n')
    add(table(['组','张手对照 MAE / °','张手对照 P95 / °'],[[g]+[fmt(val(next(r for r in sums if r['scope']=='open_controls'and r['gesture']=='all'and r['group']==g),k))for k in ['mae_deg','p95_abs_deg']]for g in 'ABCDEF']))
    add('## 5. 手指动作究竟干扰了哪一层\n')
    add('### 5.1 参考近静止时，整体朝向仍出现变化\n')
    add(f's1 的 190.5–192.0 s 是一次明确屈指。同步规则下共有 {len(ids)} 帧可比较，IMU 融合角相对变化最大 {fmt(checks["quiet_closing"]["imu_peak_deg"])}°，陀螺积分最大 {fmt(checks["quiet_closing"]["gyro_peak_deg"])}°。两种参考在这个短片段接近，与周期前部的参考异常应分别讨论。\n')
    add(table(['组','局部角度 MAE / °','视觉最大相对转角 / °','X变化RMS / °','Y变化RMS / °','Z变化RMS / °'],[[g]+[fmt(direct[g][k])for k in ['mae_deg','model_excursion_deg','axis_x_rms_deg','axis_y_rms_deg','axis_z_rms_deg']]for g in 'ABCDEF']))
    add('这表明“手指在动、参考整体姿态变化不大”时，视觉仍输出了额外朝向变化。它是与手势同时出现的直接现象；由于未把手腕机械固定、未独立监测传感器相对手部滑动，不能将其中每一度都严格归因于手指本身。A–D 的大部分变化共同出现，说明仅替换建轴不足以解决。\n')
    add('### 5.2 去掉同一次 HaMeR 整体旋转后，建轴还偏了多少\n')
    add('为隔离模型内部的形变效应，计算 `Hg(t)=RA(t)ᵀ Rg(t)`，再测它相对周期初始值的转角。它消去了 HaMeR 原始整体旋转，留下 B/C/D 相对 A 的内部变化；这是几何分解，不是滤波，也没有修改六组输出。该检验不需要 IMU，所以从周期共同零位开始使用所有 A–D 有效预测，不因 IMU 时间无效而丢掉模型形变数据。\n')
    rows=[]
    for g in 'ABCD':
        rr=[r for r in coupling if r['group']==g and int(r['cycle_id'])in confirmed]
        rows.append([g,fmt(weighted(rr,'extra_axis_deg_mean'),4),fmt(max(val(r,'extra_axis_deg_max')for r in rr),4),sum(int(r['n'])for r in rr)])
    add(table(['组','额外建轴偏转平均值 / °','额外建轴偏转最大值 / °','有效预测帧'],rows))
    add('A 的零值是“减掉 A 自身”这一数学定义，绝不是 A 有零误差。C 在本次固定 β 的 MANO 输出里，腕点与四个指根相对整体根坐标几乎不变，所以跟 A 很接近；B 的腕口顶点和 D 的掌部网格区域会受关节姿态造成的网格形变影响，因此仍有小幅变化。C 的内部建轴变化小，首先是固定手形和根节点运动学结构的结果，不能作为五点法独立估计更准确、更抗干扰的证据。腕环的整体准确性收益仍应与 A 的原始整体旋转基线比较，本次没有显示稳定收益。\n')
    rr=[r for r in coupling if r['group']=='B'and int(r['cycle_id'])in confirmed]
    add(f'同一分解下，腕环中心相对腕关节的模型内位移平均约 {fmt(weighted(rr,"ring_center_motion_mean_mm"),3)} mm、最大 {fmt(max(val(r,"ring_center_motion_max_mm")for r in rr),3)} mm；四区域中心模型内位移平均约 {fmt(weighted(rr,"region_center_motion_mean_mm"),3)} mm、最大 {fmt(max(val(r,"region_center_motion_max_mm")for r in rr),3)} mm。五个掌部关节点的最大模型内变化约 {fmt(max(val(r,"palm_joint_motion_max_mm")for r in rr),3)} mm。这些是模型坐标中的形变量，不是实物末端定位精度，也不是测到了人手真实软组织变形。\n')
    add('![整体姿态与模型形变分解](figures/03_pose_and_shape_decomposition.png)\n\n图 a：短屈指片段，所有曲线在该片段起点重设零位，仅作为局部诊断；缺测处断线。图 b：整个周期 1 去掉 A 整体旋转后的建轴偏转，灰色是图 a 时段；图 c：手指弯曲特征，未经平滑；图 d：全部 8 个周期的内部建轴最大变化，第 7 个周期仍保留但不作为确定屈指样本。曲线相近或重合本身是结果。\n')
    add('### 5.3 腕环有没有实测增益\n')
    rows=[]
    for other in 'ACD':
        rr=[r for r in paired if r['comparison']=='B-'+other and int(r['cycle_id'])in confirmed];x=np.array([val(r,'delta_mae_deg')for r in rr])
        rows.append(['B−'+other,fmt(x.mean(),4),fmt(x.min(),4),fmt(x.max(),4),int((x<-.1).sum()),int((x>.1).sum())])
    add(table(['对比','周期 MAE 差值均值 / °','最小差值 / °','最大差值 / °','B改善>0.1°的周期数','B变差>0.1°的周期数'],rows))
    add('这里用 A–D 自己共同有效、IMU 有效的帧，不让 F 的深度失败决定腕环比较的样本。负值才代表 B 更小。0.1°仅是便于描述差异量级的界线，不是统计显著性或精度认证。现有结果支持“差异小、随动作变号”，不支持稳定的腕环准确率收益。\n')
    add('## 6. 人手模型有没有完整跟出屈指、握拳与重新张开\n')
    add('A–D 是同一个 HaMeR 网格，换坐标轴不会生成四个不同的手指模型；E/F 也共用一次 MediaPipe 关键点检测。分别记录 HaMeR 三维、MediaPipe 三维，以及二维图像投影的手指弯曲。指标是四根长指 PIP＋DIP 两段弯曲角之和除以 π，再取平均；0 接近直指，1 表示较强弯曲。另输出腕点→指根与第一指节之间的夹角，补充检查指根变化；它不等于经解剖标定的 MCP 真值角。\n')
    rows=[]
    for cy in confirmed:
        cell=[]
        for model in ['hamer_bend','mp3d_bend','mp2d_bend']:
            rr=[next(r for r in shape if int(r['cycle_id'])==cy and r['model']==model and r['phase']==phase)for phase in ['open_before','flexed_hold','open_after']]
            cell.append(' → '.join(fmt(val(r,'long_finger_mean_bend'),3)for r in rr))
        rows.append([cy]+cell)
    add(table(['周期','HaMeR：张开→屈指→张开','MP三维：张开→屈指→张开','MP二维投影：张开→屈指→张开'],rows))
    add('7 个确认周期中，两个模型的平均长指弯曲均随屈指上升、再张开下降，粗动作顺序得到跟出。HaMeR 开手时这个特征约 0.08–0.14，MP 三维在若干开手段仍约 0.18–0.30，存在明显模型状态差异；不能在没有真实关节角的情况下把这些差异直接判成谁的关节角误差更小。二维弯曲很受观察方向和遮挡影响，不能与三维幅值当作同一物理量比较，更不能据此给 F 增添预测三维。逐指特征全部在 CSV 中。\n')
    add('目前不能声称“每根手指的真实角度都完整、精确跟出”：没有手指关节真值，未做单指独立动作、拇指对掌、捏合等受控重复，手势起止标签也不够精确到测几十毫秒的动作响应延迟。周期 7 不强行计为模型漏跟或成功。\n')
    add('整体转动的幅值跟随也不能只看相关系数：7 个参考较可解释周期中，有的相关性超过 0.95，幅值仍有衰减或放大。下面展示 B/E/F；A/C/D 的完整数值与 B 同表保存在附件。比例 1 表示局部转动幅度相近；高相关不代表幅值准确或转轴正确。\n')
    rows=[]
    for cy in range(2,9):
        rs=[next(r for r in cycles if int(r['cycle_id'])==cy and r['group']==g)for g in 'BEF']
        rows.append([str(cy)+('（屈指不确定）'if cy==7 else'')]+[f'{fmt(val(r,"amplitude_gain"),3)} / {fmt(val(r,"correlation"),3)}'for r in rs])
    add(table(['周期','B：幅值比例 / 相关','E：幅值比例 / 相关','F：幅值比例 / 相关'],rows))
    add('## 7. 稳定性、抖动与不该动的轴\n')
    add('从非转动对照阶段划分互不重叠的 1 s 小窗，仅用参考筛选：融合姿态和陀螺积分的相对变化均不超过 2°，至少 20 个共同有效帧。得到 **31 个近静止窗口**。没有用视觉误差挑窗口，全部窗口都画出。2°和 1 s 是明确的分析定义，不代表手腕完全静止或已经被夹具固定。\n')
    rows=[]
    for g in 'ABCDEF':
        rr=[r for r in quiet if r['group']==g]
        rows.append([g]+[fmt(np.median([val(r,k)for r in rr]))for k in ['orientation_rms_deg','adjacent_step_p95_deg','position_scatter_rms_mm','axis_x_rms_deg','axis_y_rms_deg','axis_z_rms_deg']]+[sum(int(r['jumps_gt10_deg'])for r in rr)])
    add(table(['组','窗口朝向RMS中位数 / °','窗口逐帧变化P95的中位数 / °','位置散布RMS中位数 / mm','X变化RMS中位数 / °','Y变化RMS中位数 / °','Z变化RMS中位数 / °','相邻有效帧>10°次数'],rows))
    add('朝向 RMS 围绕每个窗口自己的平均姿态计算；参考 IMU 对应 RMS 的中位数约 **0.33°**。三轴列是各方法相对于该窗口初始坐标的旋转向量分量；它们有助于看到附带的轴变化，但因缺少固定 IMU→手掌的外参，不冒充三个轴分别的真实角度误差。另有“运动偏离主要转轴的比例”供查看，但多轴真实动作本身也会使该比例变大，不能直接等同串轴率。\n')
    add('HaMeR A–D 的近静止抖动相近，E 在少数窗口仍可出现较大异常，F 总体抖动更明显。31 窗口中只有 1 个来自闭合抓握保持，故不能把窗口汇总写成充分验证了“所有握拳状态的静态精度”。位移散布还包含人手真实平移和深度噪声；没有独立平移参照，所以本报告不给出伪造的毫米级绝对定位精度。\n\n![稳定性与有效率](figures/04_stability_and_availability.png)\n')
    add('左图每点是一个实际 1 s 窗口，黑横线是中位数，不是置信区间。右图是各手势状态的完整位姿可用率，使用该状态所有原始输入帧，包括失败帧。\n')
    add('## 8. 可用率、检测丢失、配准失败与“有效但错误”\n')
    rows=[]
    for g in 'ABCDEF':
        rr=[r for r in coverage if r['gesture']=='all'and r['group']==g];n=sum(int(r['n'])for r in rr);v=sum(int(r['orientation_valid'])for r in rr);p=sum(int(r['pose_valid'])for r in rr)
        rows.append([g,n,v,fmt(100*v/n),p,fmt(100*p/n),max(int(r['longest_loss_frames'])for r in rr),fmt(max(val(r,'longest_loss_s')for r in rr),3)])
    add(table(['组','输入帧','有效朝向','朝向可用率 / %','完整位姿有效帧','完整位姿可用率 / %','最长连续失效帧数','失效首末采样跨度 / s'],rows))
    add('“朝向有效”是程序得到可用的旋转矩阵；“完整位姿”还要求位置有效。单帧失效的首末采样跨度会是 0 s，不等于没有中断；按约 30 Hz 至少损失一次输出。这里统计的是离线输入帧上的结果比例，不是机械臂实时控制频率。\n')
    add(table(['实际问题','数量和影响','含义'],[
        ['未检测到手','5 帧，影响 A–F','检测器未返回手，不是坐标轴算法失败'],
        ['HaMeR 裁剪可见面积不足','16 帧，影响 A–D','手部检测存在，但裁剪框越出图像较多，推理入口拒绝；不是模型计算崩溃或腕环配准失败'],
        ['腕环配准失败','本次 0 帧','不能因此宣称全场景都不会失败；配准成功只证明几何可拟合'],
        ['HaMeR 实测深度表面支持不足','另 3 帧，仅位置无效','朝向仍可用，但没有可靠三维位置'],
        ['E 腕点实测深度不足','另 3 帧，仅位置无效','E 的预测三维朝向仍可输出'],
        ['F 实测深度不足','12 帧','五个点中至少一点没有足够同帧有效深度'],
        ['F 正交掌宽退化','12 帧','横向几何太窄，方向对噪声敏感'],
        ['F 掌部深度几何不一致','12 帧','腕到指根实测距离超出检查范围']]))
    add(f'特别需要区分“拒绝输出”和“输出了错误结果”：s1 的 **272.484 s**，F 的数值检查通过，但相对转角为 160.78°、IMU 为 38.75°，差 **122.03°**。这一帧五点实测深度约 931–946 mm，正交掌宽仅 **{fmt(width)} mm**，刚高于 10 mm 的检查界线。近侧视、关键点聚拢时仍可能产生很不可靠的掌面朝向。这里的 122.03°是相对参考差异，不是断言在相邻 33 ms 内实际跳了 122.03°。F 仍需专门的几何可信度研究，不能因为可用率接近 100% 就认定可靠。\n')
    add('## 9. 延迟和有效输出速度\n')
    add(f'单独用 {timing["input_frames"]} 个原始输入帧做计算耗时复测，按会话和手势状态均匀选取，不按成功与否选帧；六条完整路线各重复 3 次、随机路线顺序，每条路线预热 4 次。每次重新检测，从已在内存中的 RGB-D 到一个未滤波位姿，包含检测、HaMeR 推理（A–D）、建轴及深度位置计算。使用 RTX 2070、HaMeR fp16、固定 CPU 核 8–11。\n')
    rows=[]
    for g in 'ABCDEF':
        r=timing['models'][g];rows.append([g,r['n'],r['valid'],fmt(r['valid_median_ms']),fmt(r['valid_p95_ms']),fmt(1000/r['valid_median_ms'],1)])
    add(table(['组','测量次数','有效位姿次数','有效计算耗时中位数 / ms','有效耗时P95 / ms','中位耗时倒数 / Hz'],rows))
    add('最后一列只是串行处理预算的倒数，不是测得的整套系统输出帧率。没有包含相机曝光与传输、文件读取、显示、控制器排队、通信和机械臂运动，因此**真实端到端延迟尚未测量**。也不能把回放中所有帧离线算完解释为实时 30 Hz 都能跟上。成功和失败尝试在耗时 JSON 中分别记录；本次抽样的计时帧全部有效。\n')
    rows=[]
    for g in 'BEF':
        rr=[r for r in cycles if r['group']==g and int(r['cycle_id'])in confirmed and r.get('apparent_lag_ms')]
        rows.append([g,'、'.join(f'{r["cycle_id"]}:{val(r,"apparent_lag_ms"):.0f}'for r in rr),sum(r['lag_search_boundary']=='True'for r in rr)])
    add('还输出了视觉角度曲线相对 IMU 的最佳相关时移，但下面的跨度和边界命中说明它不适合当作硬件延迟：\n')
    add(table(['组','周期:表观时移 / ms','达到±300ms搜索边界的周期数'],rows))
    add('表观时移会混合时钟偏差、参考异常、幅值误差和真实算法跟随差异；周期 1 的参考已异常，更不能用其最佳时移解释系统反应速度。手势标签的 ±0.5 s 边界误差也不足以测几十毫秒的握拳响应。\n')
    add('## 10. 一致性、长期零位与结论的适用范围\n')
    add('7 次确认手势里，屈指—再张开的粗动作趋势得到重复观察；但姿态误差的大小、模型之间的优劣和幅值衰减随观察方向与动作改变，不能宣称跨姿态、跨会话的一致精度。原始轴方向已经统一，但未做刚性的 IMU→手掌外参标定，不能把三轴分量对齐的外观当作完整三维准确性。\n')
    fixed=read('单零位全录像指标');rows=[]
    for s in ['s1','s2']:
        for g in 'BEF':
            r=next(r for r in fixed if r['session']==s and r['group']==g);rows.append([s,g,fmt(val(r,'zero_t'),3),fmt(val(r,'mae_deg')),fmt(val(r,'imu_gyro_mae_deg'))])
    add('为了避免通过反复归零掩盖长期问题，另用每份录像第一个共同有效时刻作唯一零位，跨所有已选区间保持；结果如下。这不是录像全时段评测，因为没有重放未选区间。\n')
    add(table(['会话','组','唯一零位时刻 / s','相对融合角 MAE / °','融合角与积分参考 MAE / °'],rows))
    add('这批数据仍不支持“设置一次零位后长期保持某个角度精度”。B 原点与 A/C/D 的腕关节原点也不同，平移比较必须讲明原点；本次已输出位置散布与三轴跨度，但没有独立位置真值，无法给出机器人末端真实运动误差。\n')
    add('## 11. 还缺什么数据，下一次如何补成可发表的实验\n')
    add('**当前不缺能重放的握拳视频，但缺少能把“手指导致的朝向误差”单独隔离出来的受控记录。** 现有大部分手势周期同时转腕，不能据此计算纯手指动作的因果误差，也不能从两个会话推断群体表现。\n')
    add('最直接的一组补录是：把独立姿态参照刚性固定到掌部／腕部，并用支撑或夹具限制手腕运动；保持相同相机和光照，连续记录“张手保持—半握—张开—闭合抓握—张开”，每种姿态重复数次，同时保存原始 RGB、配准深度、原始 IMU 和可核对的时间戳。先做这一小组就能判断屈指时模型是否仍输出明显整体转动。传感器固定必须不妨碍手指动作，且不能仅依靠松动绑带宣称刚性关系。\n')
    add('随后分别补：① 手势固定时的三个单轴转动，用独立参照标定转轴并测串轴；② 腕姿态固定时逐指、拇指对掌和捏合，测遮挡与模型形变；③ 已知长度／位置的平移轨迹；④ 相机事件与机器人实测运动共用可对照时间标记，测端到端延迟。各组都保持本次无滤波条件，再另做滤波收益实验，避免一次同时改变多个变量。重复次数与被试数应在预实验方差可见后安排，不能事后把视频帧数当成独立样本量。\n')
    add('对 SCI 研究价值的直接判断：本次已经支持把问题拆为“视觉整体朝向估计”“网格／关节建轴对手指形变的敏感性”“深度几何可观测性与失败识别”“参考姿态可靠性”四部分。最有用的新证据是这些层次表现不同；现阶段不支持把腕环包装为已证实提高整体精度的核心创新，也不能把 F 的低平均差异包装为全面更优。若要提出新方法，先用上面的固定手腕小实验建立可反驳的基线，再验证它是否减少额外旋转且不压制真实转腕。\n')
    add('## 12. 明细、原始输出与重现实验\n')
    add('关键明细均在本报告同目录：\n')
    rows=[['按手势汇总指标.csv','按状态及已确认／全周期／对照段的 MAE、RMSE、P95、有效率与时间敏感性'],['完整手势周期指标.csv','8个完整周期的误差、幅值、相关、参考差异和表观时移'],['逐阶段六组指标.csv','每个阶段的误差、三轴活动、抖动、位置散布'],['去除HaMeR整体旋转后的建轴偏转.csv','与 IMU 无关的内部建轴与中心形变'],['手指弯曲及模型形变.csv','每个手势阶段的逐指与指根特征'],['逐帧手指状态及配准质量.csv','原始逐帧指形特征、左右手原始报告和腕环质量'],['逐帧角度误差与三轴变化.csv','可追溯到源时刻的共同有效角度结果'],['各组自身有效帧指标.csv','各自成功帧的敏感性检查'],['有效率及失败分类.csv','所有输入帧的有效计数和失败原因'],['逐帧失败明细.csv','每次检测／几何／位置失败的时刻与原因'],['参考近静止窗口稳定性.csv','全部31个参考近静止窗口'],['时钟对齐诊断.csv','对齐对照段、验证段及相关性'],['单零位全录像指标.csv','跨所选片段不重置零位的结果'],['processing_timing.json','独立计时抽样与全部逐次耗时'],['具体片段复核.json','屈指局部变化、IMU异常和F接受的大误差实例']]
    add(table(['文件','内容'],[[f'[{name}]({name})',meaning]for name,meaning in rows]))
    add('原始录制文件未改动。重放帧缓存、每帧模型输出和 HaMeR 原始网格保留在 `.runtime/gesture_pose_20260917/`；CSV 与图中没有手工替换结果。`results/s1`、`results/s2` 内包含 `mp.jsonl`、`hamer.jsonl`、`hamer_raw_geometry.npz`、时间对齐说明和输入元数据。图形 PDF/SVG/PNG 与图源脚本可编辑；大体积原始录像与缓存不纳入 Git。\n')
    add('重新分析现有回放结果的命令（不会启动机械臂或采集相机）：\n\n```bash\ncd /home/dongtian/myhandarmtest1-mediapipe-comparison-20260915\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/analyze_gestures.py\n/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/plot_gesture_results.py\n/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/write_gesture_report.py\n```\n')
    (REPORT/'手势变化对手部姿态影响_A-F完整报告.md').write_text('\n'.join(doc),encoding='utf-8')
    print(REPORT/'手势变化对手部姿态影响_A-F完整报告.md')

if __name__=='__main__':main()
