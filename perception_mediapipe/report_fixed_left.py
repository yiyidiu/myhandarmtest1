"""Report the fixed-left intervention without overwriting automatic-hand results."""
import csv,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perception_mediapipe.report_six_groups import table,f,NAMES
P=ROOT/'docs/six_group_fixed_left_20260917'
OLD=ROOT/'docs/six_group_unfiltered_20260917'
OUT=ROOT/'.runtime/six_group_fixed_left_20260917'


def main():
    s=json.load(open(P/'summary.json'));old=json.load(open(OLD/'summary.json'))
    rows=json.load(open(P/'segment_metrics.json'));timing=json.load(open(P/'processing_timing.json'))['models']
    fixed=list(csv.DictReader(open(P/'整段录像固定零位指标.csv',encoding='utf-8-sig')))
    frame_rows=list(csv.DictReader(open(P/'逐帧误差与三轴变化.csv',encoding='utf-8-sig')))
    event=min((r for r in frame_rows if r['session']=='later'and r['group']=='B'),key=lambda r:abs(float(r['t'])-173.8070559501648))
    policy=[];invariance={}
    for label in ['earlier','later']:
        now=[json.loads(x)for x in open(OUT/label/'mp.jsonl')]
        before=[json.loads(x)for x in open(ROOT/'.runtime/six_group_20260917'/label/'mp.jsonl')]
        assert len(now)==len(before)
        for a,b in zip(now,before):
            assert a['t']==b['t'] and a['frame_number']==b['frame_number']
        used_right=sum(bool(x['detection']['is_right'])for x in now if x.get('detection'))
        assert used_right==0
        disagreements=sum(x.get('handedness_mismatch',False)for x in now)
        policy.append(dict(session=label,total=len(now),detector_right_disagreements=disagreements,applied_right_frames=used_right))
        invariance[label]={}
        for g in 'EF':
            status_changes=0;max_matrix_diff=0.
            for a,b in zip(now,before):
                x,y=a['groups'][g],b['groups'][g]
                status_changes+=int(x['valid']!=y['valid'])
                if x['valid']and y['valid']:
                    max_matrix_diff=max(max_matrix_diff,float(np.max(abs(np.array(x['rotation'])-np.array(y['rotation'])))))
            invariance[label][g]=dict(validity_changes=status_changes,max_rotation_matrix_difference=max_matrix_diff)
    (P/'手别应用核对.json').write_text(json.dumps(dict(policy=policy,EF_unchanged_check=invariance),ensure_ascii=False,indent=2))
    parts=[]
    def add(t):parts.append(t.strip()+'\n')
    def fig(name,caption):add(f'![{caption}]({P/(name+".png")})\n\n{caption}')
    add('''# 固定左手后的六组无滤波对比

本轮把“输入始终是左手”落实到实际推理与坐标转换。原自动手别实验单独保留。

**手别固定在推理之前生效。检测器的原始分类仍记录，分类偶尔报右手时继续按左手处理，不删帧、不冻结、不补上一帧。**

六组实验继续关闭时间滤波、掌面辅助、IMU拟合修正、视角偏置和跨帧保持。F组仍只使用同一帧真实深度，不借用预测三维点。''')
    add('## 1. 修改及实际生效范围')
    add('''离线：A—D所有可推理图像均向HaMeR传入左手身份，使用对应的输入镜像、三维恢复与左手参考模板；A—D共享单次HaMeR输出。E/F的应用身份也固定左手，关键点建轴公式不变。

实时入口：MediaPipe直接运行入口默认固定左手，窗口显示LEFT FIXED；HaMeR实时启动脚本显式传入固定左手参数，检测子进程在进入手部选择、连续检测确认和HaMeR推理前提供左手身份，同时保留原始分类。

这轮没有启动相机或机械臂。已用原来误判的真实缓存图像验证两个实时检测入口均返回“应用左手、原始分类右手”；实时控制的其他参数没有在本轮修改。此模式按用户提供的单左手输入条件工作。''')
    add(table(['录像','采样时刻','检测器原始报右手','实际按右手处理'],[
        ['录像一'if x['session']=='earlier'else'录像二',x['total'],x['detector_right_disagreements'],x['applied_right_frames']]for x in policy]))
    add('原来14个分类反转时刻全部保留。固定的是已知物理身份，不是修改图像、修正参考角或对姿态做时间滤波。')
    add('## 2. 原126°异常帧复测')
    add(table(['条件','IMU相对转角/°','B组相对转角/°','绝对差/°'],[
        ['原自动手别：误用右手',f(24.66867143348794),f(151.0688988012726),f(126.40022736778467)],
        ['本轮固定左手',f(event['imu_angle_deg']),f(event['model_angle_deg']),f(abs(float(event['angle_error_deg'])))]]))
    add('图像时间173.8071 s，参考时间仍为该动作段首个共同有效帧。图像、裁剪、网络权重、形状参数和建轴方法未换；本轮修正手别处理后，该帧的大跳变消失。不能由这一帧推断其他类型的误转也已经解决。')
    diagnostic=OUT/'residual_jump_crop_check.json'
    if diagnostic.exists():
        d=json.load(open(diagnostic))
        (P/'残余跳变裁剪对照.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
        add('### 固定手别后仍有的原始姿态跳变')
        add('35.24°是与IMU参考的最大角度差，不能当作相邻帧的跳变角。另核对到录像一244.21995→244.28660 s：两帧实际与原始分类均为左手、手部关键点均在图像内，间隔66.65 ms，IMU变化0.73°；A/B/C/D输出分别变化16.007/16.005/16.005/15.898°。变化已出现在HaMeR原始整体朝向。')
        add(table(['图像','裁剪框','相对前帧原始输出的变化/°'],[
            ['前帧'if x['image_frame']==0 else'后帧','前帧'if x['crop_frame']==0 else'后帧',f(x['change_from_first_deg'],3)]for x in d['results']]))
        add('该小对照保持模型、权重、左手身份、形状参数和其他设置相同。同一图像仅换裁剪框，输出改变7.71°；图像换成后帧但保持前帧裁剪框，输出改变9.51°。这说明该例对裁剪与图像变化敏感；两个因素存在耦合，不能把两项直接相加作误差分解。本诊断未修改实际裁剪逻辑，也未加入滤波或改变六组数据。')
    add('## 3. 完整数据重跑后的总体结果')
    add(table(['组别','原MAE/°','固定左手MAE/°','新逐帧RMSE/°','新最大绝对差/°','有效姿态','处理中位数/ms'],[
        [g+' '+NAMES[g],f(old['models'][g]['macro_mae_deg'],3),f(s['models'][g]['macro_mae_deg'],3),
         f(s['models'][g]['pooled']['rmse_deg'],3),f(s['models'][g]['pooled']['max_abs_deg'],2),
         str(s['models'][g]['valid_orientation'])+'/'+str(s['models'][g]['frames']),f(timing[g]['median_ms'],2)]for g in 'ABCDEF']))
    add('MAE先在每个动作片段内计算，再对79个片段等权平均；逐帧RMSE与最大差直接使用配对有效样本。所有指标比较的是相对总转角与IMU参考的一致性，不是独立逐轴真值精度。处理耗时不包括相机、显示或机械臂执行。')
    add(f"固定左手后，B组最大绝对差从126.40°降至{s['models']['B']['pooled']['max_abs_deg']:.2f}°，但并未消除全部误转。A—D的片段MAE仍然接近；本轮仍未观察到稳定的腕环角度精度优势。")
    add(table(['录像','总采样时刻','参考时间有效','六组和IMU共同有效'],[
        [x['session'],x['frames'],x['imu_valid'],x['six_methods_and_imu_valid']]for x in s['reference_validity']]))
    add('输入仍为同一批4,053个去重采样时刻、相同79个标注动作，时间偏移及参考有效规则未变。由于手别影响模型输出有效性，共同有效样本数允许发生真实变化，并在上表明示。')
    fig('01_整体误差与腕环增益','固定左手后的六组总体误差与腕环配对差值。')
    add('### 腕环与其他HaMeR建轴方法')
    add(table(['比较','平均MAE差/°','腕环更低的片段数','降低超过0.1°','升高超过0.1°'],[
        [x['comparison'],f(x['mean_delta_mae_deg'],4),str(x['B_lower_error_count'])+'/'+str(x['n']),x['B_lower_by_more_than_0_1deg'],x['B_higher_by_more_than_0_1deg']]
        for x in s['paired']if x['session']=='all']))
    add('负值有利于腕环；0.1°仅为描述刻度。结论应结合差值大小和逐片段一致性，不按第四位小数决定优劣。')
    add('## 4. 平移、旋转、跟随幅值及三轴活动')
    cats=['平移','俯仰','横滚','偏航','静止','组合动作','丢失片段']
    add(table(['动作']+list('ABCDEF'),[[c]+[f(np.mean([x['mae_deg']for x in rows if x['group']==g and x['category']==c]))for g in 'ABCDEF']for c in cats]))
    add('上表为按动作分层的片段MAE均值，单位°。')
    fig('02_按动作类别的误差','动作分类和原实验相同。')
    add(table(['录像二动作','IMU峰值/°']+list('ABCDEF'),[
        [label,f(next(x['imu_peak_deg']for x in rows if x['session']=='later'and x['id']==i))]+[
            f(next(x['model_peak_deg']for x in rows if x['session']=='later'and x['id']==i and x['group']==g))for g in 'ABCDEF']
        for i,label in [(4,'向左平移'),(8,'俯仰'),(9,'横滚'),(10,'偏航'),(13,'原裁剪丢失片段')]]))
    fig('03_动作跟随与误转实例','保留原来容易误判的片段，查看固定手别后还有哪些误差。')
    fig('04_非目标轴活动','非目标轴活动包括真实伴随转动；没有IMU—手部轴向标定，不能称为串轴真值误差。')
    add('## 5. 抖动、位置及失效')
    add(table(['组别','静止姿态散布RMS/°','静止位置散布RMS/mm','有效完整位姿'],[
        [g,f(s['models'][g]['static_rotation_rms_median_deg'],3),f(s['models'][g]['static_position_rms_median_mm'],3),s['models'][g]['valid_pose']]for g in 'ABCDEF']))
    add('静止散布为3段静止片段RMS的中位数，包含人手真实微动。位置没有独立真值，不能据此宣称毫米级定位准确性。完整位姿同时要求姿态和位置有效。')
    reasons=[]
    for label in ['earlier','later']:
        for g in 'ABCDEF':
            x=next(x for x in s['coverage']if x['session']==label and x['group']==g)
            reasons.append([label,g,str(x['reasons'])if x['reasons']else'无姿态失败',x['longest_observed_failure_frames']])
    add(table(['录像','组别','姿态失败原因及数量','最长连续失败时刻数'],reasons))
    add('详细位置失败和检测器原始手别分歧仍保存在逐帧明细。原始分类分歧不代表本轮采用了右手，也不作为删帧条件。F缺真实深度仍然失败。')
    fig('05_稳定性与有效率','固定左手没有重新开启时间滤波或对失败补值。')
    add('## 6. 处理时间与长期零位')
    add(table(['组别','处理时间中位数/ms','P95/ms'],[[g,f(timing[g]['median_ms']),f(timing[g]['p95_ms'])]for g in 'ABCDEF']))
    fig('06_实际处理耗时','重新用96张相同图像进行三轮处理计时；不是机器人端到端延迟。')
    add(table(['录像：固定一个零位的MAE/°']+list('ABCDEF'),[
        [label]+[f(next(x['mae_deg']for x in fixed if x['session']==label and x['group']==g))for g in 'ABCDEF']for label in ['earlier','later']]))
    add('固定左手不会自动解决长期零位参考不一致。IMU融合角与陀螺积分之间仍需独立核对；逐动作小误差不能替代长期固定零位精度。原参考问题的证据边界继续适用。')
    fig('08_整段录像固定零位','整段固定零位下的长期一致性，未为各动作重新归零。')
    add('## 7. 实现、区域和旧结果的关系')
    add('D组仍使用原中性模型上固定的四个12顶点区域，未按新结果重新选区。所有建轴方法、深度规则、IMU同步和统计口径沿用原实验定义。')
    fig('07_固定网格区域','D组区域未更改。')
    equal=all(x[g]['validity_changes']==0 and x[g]['max_rotation_matrix_difference']==0 for x in invariance.values()for g in 'EF')
    add('逐帧核对E/F与原实验：'+('有效状态和旋转矩阵完全相同。'if equal else'存在差异，详见手别应用核对.json中的实际差值。'))
    if equal:
        add('E/F汇总误差的小幅变化来自六组共同有效样本范围变化，并非E/F逐帧估计发生变化。原共同有效参考样本3,713个；本轮数量见第3节。')
    add(f'''原自动手别报告：[原六组报告]({OLD/'六组无滤波对比实验报告.md'})。原结果对应本地提交`bc532af`，本轮结果另存，不覆盖原表。

本轮代码分支：`codex/fixed-left-hand-20260917`。逐动作、逐帧、失败、速度分层及整段零位CSV均在本报告同目录。模型原始预测保存于本轮独立运行目录。没有上传GitHub。

复现命令：

```bash
cd {ROOT}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/mediapipe_env/bin/python perception_mediapipe/six_group_experiment.py mp
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/six_group_experiment.py hamer
OPENBLAS_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/analyze_six_groups.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/benchmark_six_groups.py
/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/plot_six_groups.py
/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/report_fixed_left.py
```

复现会重写本轮输出；要保留多次运行应先备份。实时入口仍为`bash scripts/run_grasp_feedback_recording.sh --no-recording`；选择HaMeR可在命令前设置`TELEOP_HAND_BACKEND=hamer`。本轮已在这两个入口固定左手。''')
    (P/'固定左手六组对比报告.md').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':main()
