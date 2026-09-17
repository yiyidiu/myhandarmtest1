"""Build a Chinese evidence report from measured results and explicit limits."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'docs/six_group_unfiltered_20260917'
OUT=ROOT/'.runtime/six_group_20260917'
NAMES={'A':'HaMeR 原始整体朝向','B':'HaMeR 腕环配准','C':'HaMeR 五点连线','D':'HaMeR 四区域中心','E':'MediaPipe 预测三维','F':'MediaPipe 二维＋真实深度'}


def f(x,n=2):return '—'if x is None or not np.isfinite(float(x))else f'{float(x):.{n}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v)for v in row)+' |'for row in rows])+'\n'


def main():
    s=json.load(open(P/'summary.json'));m=s['models'];r=json.load(open(P/'segment_metrics.json'))
    speed=json.load(open(P/'processing_timing.json'))['models']
    fixed=list(csv.DictReader(open(P/'整段录像固定零位指标.csv',encoding='utf-8-sig')))
    geom=json.load(open(OUT/'geometry_definition.json'))['False']
    text=[]
    def add(x):text.append(x.strip()+'\n')
    def fig(name,caption):add(f'![{caption}]({P/(name+".png")})\n\n{caption}')
    add('''# 六组手部姿态对比实验报告：关闭滤波与掌面辅助

日期：2026年9月17日。对象：同一工程中 A—F 六组离线原始手部姿态，不含机械臂控制误差。

**最直接的结果：更换 HaMeR 的建轴依据，没有在这两段录像里产生稳定的角度精度增益。腕环、原始朝向、五点连线和四区域方法的平均误差很接近。MediaPipe 3D 更快，但部分真实俯仰、横滚被低估；二维加真实深度在不同录像中的表现差别很大。**

**重要限制：下文约3°的HaMeR平均误差是逐动作片段内的相对转角差，不是绝对姿态精度，也不是长期保持同一个零位的精度。整段固定零位的差异明显更大，IMU融合角与陀螺积分本身也存在长期不一致，见第6节。**''')
    add('## 1. 实际做了什么，哪些处理已关闭')
    add(table(['组别','建轴输入','实现'],[
        ['A','同一次HaMeR预测的global_orient','左右手双侧镜像，再变换到共同解剖坐标'],
        ['B','HaMeR腕口16个顶点','复用工程的稳健相似配准，统一轴定义'],
        ['C','HaMeR的0、5、9、13、17点','腕点到四指根中心为纵轴；小指根到食指根为横轴'],
        ['D','四个固定掌部网格区域','每区12个顶点取中心，建立掌长、掌宽和正交轴'],
        ['E','MediaPipe预测三维的同五点','与C使用同一个建轴函数'],
        ['F','MediaPipe二维点和对齐真实深度','五点逐一反投影后调用同一个建轴函数；缺深度即失败']]))
    add('''时间滤波、姿态平滑、位置平滑、死区、跳变抑制、跨帧保持、掌面辅助、视角偏置校正和IMU拟合修正全部关闭。实验直接调用原始模型和建轴函数，没有经过实时入口的滤波器或机器人控制链。

保留原工程的同帧深度空间采样：每个点取5×5邻域，至少8个有效深度，取其中位数；深度范围0.15—2.5 m，邻域深度中位绝对偏差超过12 mm则失败。这是同一帧中的空间取样，不是时间平滑。F还检查腕点到指根的距离是否处于25—180 mm，防止把背景深度组成一只手。参数没有按本次误差调优。

F的实际帧检查：把预测三维数组替换为无效数，F的有效姿态完全不变；把真实深度清空，F返回深度不足。它不采用预测三维深度补洞。

A—D共用每帧同一次HaMeR推理；使用相同权重、裁剪图像、fp16精度和固定的每录像形状参数。两种模型使用同一次MediaPipe静态检测提供的目标框。E/F也共享同一次MediaPipe预测。所有组不做时间跟踪加速。HaMeR位置计算保留原有真实深度与模型表面的关联，未使用历史深度保持。''')
    add('## 2. 数据范围和有效样本')
    add(table(['录像','原缓存行数','去重后时刻','动作片段','IMU时间有效','六组与IMU共同有效'],[
        ['录像一',3402,3401,66,3121,3111],['录像二',652,652,13,611,602],['合计',4054,4053,79,3732,3713]]))
    add('''录像一覆盖慢速、中速、快速的左右、前后、上下平移，以及俯仰、横滚、偏航、组合动作和静止；录像二包含操作反馈指出的误转、主动旋转及原裁剪丢失片段。

数据为已有两段录像中全部已缓存的带标注采样时刻，典型采样间隔约67 ms；不是相机完整30 Hz视频，也不是跨受试者数据。原缓存1条重复图像已去重；缺源帧保留为失败。79段均进入分析，不裁掉每段开头或结尾，不删除大误差点。

六组共同产生姿态的时刻为4,027个；再要求同步IMU时间标记有效后为3,713个，组成公平配对主分析。参考时间无效321个、六组未共同有效26个，两类重叠7个。模型有效率仍以全部4,053个时刻为分母，不因参考缺失而抬高。

动作分类：平移39段，俯仰7段，横滚14段，偏航10段，静止3段，组合动作4段，原标注丢失2段。统计均为描述性结果；没有把视频帧当作独立受试者计算显著性。''')
    add('## 3. 坐标、零位和误差到底是什么')
    add('''共同坐标：X指向食指侧，Y从腕部指向四指根中心，Z=X×Y；采用右手坐标系。先确定Y，再去掉X在Y上的分量。C/E/F调用同一个函数，B/D只用中性模型的固定基变换统一轴名。A的左手镜像采用SRS而非只乘一个反射矩阵。

每个动作片段取六组和IMU共同有效的同一首帧为零位。某时刻的转角是“当前姿态相对该零位的三维总旋转角”；误差是模型总旋转角减IMU总旋转角。MAE是误差绝对值的平均，RMSE会更重地惩罚少量大误差。

主表先在每个片段内计算MAE/RMSE，再对79个片段等权平均。另给出所有有效帧直接汇总的数值，它会更重地反映长片段和异常帧。两者不是同一个统计量。

总转角比较不需要知道IMU贴在手上的精确轴向；但它不能证明三根轴都正确，也不能识别所有方向颠倒。三轴分量使用旋转向量，即转轴方向乘旋转角度，避免混淆欧拉角顺序。缺少独立的IMU—手部外参，三轴指标只能描述非目标轴活动，不能宣称逐轴真值误差。

网络矩阵先按原适配器数值容差检查，再做逐帧正交化以消除浮点舍入；没有跨帧运算。首轮新增校验过严的问题已经纠正，并用保存的同一批原始输出重算；初始输出另存。仍不满足原适配器要求的1帧保留为推理失败。''')
    add('## 4. 整体结果：短动作相对转角')
    add(table(['组别','片段MAE均值/°','片段RMSE均值/°','逐帧RMSE/°','逐帧绝对误差P95/°','有效姿态/4053','实测处理中位数/ms'],[
        [g+' '+NAMES[g],f(m[g]['macro_mae_deg'],3),f(m[g]['macro_rmse_deg'],3),f(m[g]['pooled']['rmse_deg'],3),
         f(m[g]['pooled']['p95_abs_deg'],2),m[g]['valid_orientation'],f(speed[g]['median_ms'],2)]for g in 'ABCDEF']))
    add('''四种HaMeR方法的片段MAE都约3.25—3.27°，目前没有可支持“腕环显著提高角度准确性”的结果。

不能只读平均误差：腕环B的逐帧最大误差为126.40°，E为26.51°，F为39.70°。B的最大异常发生在录像二约173.81 s，正好处于左右手判别反转的帧；该异常没有通过滤波抹掉。它提示手身份判别是需要进一步隔离的错误来源，但这里没有做强制正确手别的因果干预，不能把全部异常都归因于该因素。

E的平均绝对误差更高，但逐帧RMSE低于HaMeR四组，说明平均精度与极端异常的排序不同。现有结果不能压缩成“某模型全面更好”。''')
    fig('01_整体误差与腕环增益','同一批动作配对比较；每个散点是一个片段，不是一个独立受试者。')
    add('### 腕环究竟有没有增益')
    add(table(['比较','平均MAE差/°','差值中位数/°','腕环误差较低片段','降低超过0.1°','升高超过0.1°'],[
        [x['comparison'],f(x['mean_delta_mae_deg'],4),f(x['median_delta_mae_deg'],4),str(x['B_lower_error_count'])+'/79',x['B_lower_by_more_than_0_1deg'],x['B_higher_by_more_than_0_1deg']]
        for x in s['paired']if x['session']=='all']))
    add('''差值为腕环B减对照，负值才代表腕环更好。0.1°只是方便读表的描述刻度，不是通过/失败界线。

基于现有观测：腕环相对A的平均差约+0.006°，片段胜负接近一半；相对C、D也没有一致优势。静止姿态散布只从A的约2.183°变成B的约2.173°，不能把这样的小差别包装为实用增益。

解释层面，四组共同依赖同一网络预测的整体朝向，掌部关节点和网格随其变化，建轴不等于增加了独立观测。实测A与C的相对关系变化很小，B/D略大，符合共同误差来源的判断。该解释不是“所有姿态下数学上完全相等”的证明，也不代表腕环原点、机械手解耦或控制工程用途没有价值；本轮只检验它带来的手部运动测量差异。''')
    add('## 5. 平移、旋转与动作是否跟完整')
    cats=['平移','俯仰','横滚','偏航','静止','组合动作','丢失片段']
    add(table(['动作','片段数']+list('ABCDEF'),[[c,sum(x['category']==c and x['group']=='A'for x in r)]+[
        f(np.mean([x['mae_deg']for x in r if x['category']==c and x['group']==g]))for g in 'ABCDEF']for c in cats]))
    add('上表数值均为片段MAE均值，单位°。')
    fig('02_按动作类别的误差','不同动作类别的误差；不把组合动作误归为单纯平移。')
    picks=[(4,'向左平移'),(8,'主动俯仰'),(9,'主动横滚'),(10,'偏航')]
    add(table(['录像二动作','IMU最大相对角/°']+list('ABCDEF'),[
        [title,f(next(x['imu_peak_deg']for x in r if x['session']=='later'and x['id']==i))]+[
          f(next(x['model_peak_deg']for x in r if x['session']=='later'and x['id']==i and x['group']==g))for g in 'ABCDEF']for i,title in picks]))
    add('''向左平移的例子中，参考仍有约9.88°真实转动，所以不能把模型测到的所有转角都叫误转。HaMeR四组产生约48—53°的最大转角，E约22.77°，其多余转动较少。

主动俯仰中，参考约53.57°，B跟到52.30°，E仅38.40°，约保留72%的峰值；主动横滚中，参考约46.38°，B跟到43.10°，E仅31.12°，约67%。这直接显示E在部分动作中漏掉真实转动。

F在录像二的这两个旋转例子中幅值接近参考，但在录像一更不稳定。不能因总体均值较差就否定它在所有场景的价值，也不能只选录像二的好结果作为最终结论。

曲线以原始观测绘制；共同有效观测间隔超过0.20 s时断线，缺失处不造点。''')
    fig('03_动作跟随与误转实例','六组原始动作曲线；HaMeR四条曲线大量重合，本身就是实验结果。')
    add('## 6. 长期固定零位：必须单独说明的参考问题')
    add(table(['录像','零位录像时间/s']+list('ABCDEF'),[
        ['录像一'if label=='earlier'else'录像二',f(next(float(x['reference_time'])for x in fixed if x['session']==label))]+[
            f(next(float(x['mae_deg'])for x in fixed if x['session']==label and x['group']==g))for g in 'ABCDEF']for label in ['earlier','later']]))
    add('上表是整段只设一次零位、随后所有采样时刻的MAE，单位°；与第4节逐动作零位的结果不能互换。')
    discrepancy=[float(next(x['reference_disagreement_mae_deg']for x in fixed if x['session']==label))for label in ['earlier','later']]
    add(f'''录像一中约25—28°、录像二中约15—17°的长期差异，说明目前不能宣称“设一次零位后长期维持3°精度”。

在完全相同的有效时刻，IMU融合角与直接陀螺积分的长期相对角MAE分别为 **{f(discrepancy[0])}°、{f(discrepancy[1])}°**；短片段内两种参考的平均差则约0.216°。视觉估计、IMU长期漂移、佩戴变化等因素尚未被独立分离。

因此，第4节用于比较短动作相对变化；第6节用于揭示长期一致性的限制。不能把长期全部偏差判成视觉误差，也不能宣称已证明全部来自IMU。用于SCI长期零位精度结论前，需要独立姿态参照与佩戴刚性检查。''')
    fig('08_整段录像固定零位','整段固定零位的曲线同时展示两种IMU参考；没有重置每个动作的零位来隐藏长期偏差。')
    add('## 7. 抗串轴：不该动的轴有没有动')
    add(table(['录像二动作','非目标轴定义']+list('ABCDEF'),[
        [title,off]+[f(next(x['off_intent_axis_rms_deg']for x in r if x['session']=='later'and x['id']==i and x['group']==g))for g in 'ABCDEF']
        for i,title,off in [(8,'俯仰','Y/Z'),(9,'横滚','X/Z'),(10,'偏航','X/Y')]]))
    add('''上表为非目标轴旋转向量的合成RMS，单位°，没有把它命名为“串轴误差”。它同时包含真实伴随转动、解剖轴初始对齐差和模型误差。

B在这些动作中没有一致地减少非目标轴活动。E在俯仰例子中非目标活动较小，但横滚例子更大，而且主轴幅值被低估；单看“另一根轴动得少”可能误判一个整体变钝的方法。

逐动作CSV另附IMU及各模型的非主转轴能量比例，这是与坐标旋转无关的转轴集中度；它不能判断主轴是否指向正确方向。真正逐轴误差仍需要独立IMU—手部轴向标定。''')
    fig('04_非目标轴活动','三个主动旋转例子的非目标轴活动；不作为逐轴真值精度声明。')
    add('## 8. 位置、稳定性和抖动')
    add(table(['组别','静止姿态散布RMS/°','静止位置散布RMS/mm'],[
        [g,f(m[g]['static_rotation_rms_median_deg'],3),f(m[g]['static_position_rms_median_mm'],3)]for g in 'ABCDEF']))
    add('''数值是3个静止片段各自RMS的中位数，包含人的真实微动；样本很少，不能当作传感器噪声标定结果，更不是绝对定位精度。

A/C/D的位置原点都设在同一个MANO腕点，使用同一相机平移估计，因此它们的位置数值相同；更换坐标轴本身不会改变这个原点的位置。B采用腕环中心，旋转时与腕点的轨迹自然不同。E/F采用相同二维腕点和真实深度，所以在共同有效帧中位置也相同。

逐动作CSV保留X/Y/Z位移跨度、位置散布、相邻帧角度跳变的中位数/P95和超过10°的跳变次数。位置跨度能描述动作和异常，但没有独立位置真值，本轮不能给出“平移误差为多少毫米”或定位准确率。不同原点的轨迹差也不能直接当作错误。''')
    add(table(['动作','代表原点','X跨度/mm','Y跨度/mm','Z跨度/mm'],[
        [title,{'A':'A/C/D腕点','B':'B腕环中心','E':'E/F深度腕点'}[g]]+[
            f(next(x[k]for x in r if x['session']==sess and x['id']==idx and x['group']==g))
            for k in ['position_span_x_mm','position_span_y_mm','position_span_z_mm']]
        for sess,idx,title in [('earlier',2,'录像一向左'),('earlier',15,'录像一向前'),('later',2,'录像二向下')]for g in 'ABE']))
    add('位置坐标为相机光学坐标：X向图像右、Y向图像下、Z沿深度方向。上表是每个坐标的最大值减最小值，不是末点减起点，也不是与外部位置真值的误差。前三种原点都测到了主要平移；伴随位移是否全为误差尚不能确定。')
    fig('05_稳定性与有效率','无时间滤波时的静止散布与有效率；有效率图显式标明截断纵轴。')
    add('## 9. 可用率、检测丢失和配准失败')
    add(table(['组别','有效姿态','有效完整位姿','完整位姿率'],[
        [g,str(m[g]['valid_orientation'])+'/4053',str(m[g]['valid_pose'])+'/4053',f(100*m[g]['valid_pose']/4053,3)+'%']for g in 'ABCDEF']))
    add('''本次失败分类：

- 1个源图像/深度配对时刻缺失：六组共同无法从该源帧输出。
- 1个HaMeR输出的旋转矩阵未满足原推理适配器数值要求：A—D共同失败，属于推理输出无效，不是腕环配准失败。
- F另外有12帧五点深度不足、5帧邻域深度混杂、7帧五点深度对应的掌部长度不一致，共24帧；加源帧缺失后总失败25帧。
- E有2帧能给出方向，但腕点真实深度不足，不能提供完整位姿。
- 当前缓存内重新静态检测未出现NO_HAND_DETECTION；腕环数值配准失败为0。这个条件有效率不能推广成完整视频或实际运行的零丢失率。
- 有14个时刻发生左右手类别与本段左手操作身份不一致，仍计入结果，没有静默删除。误判即使产生了“有效矩阵”，仍可能导致很大的错误。

F最长已观测连续失败为9个采样时刻，首末相隔约0.567 s。这里的持续时间仅覆盖已采样观察，不能推断缺口外相机帧情况。

“检测丢失”是图像没有合格手候选；“配准失败”是B的腕环几何拟合无法形成有效旋转；“深度失败”是F无法从真实深度恢复五点。它们不能合并成同一种失败。逐帧原因和时间已导出。''')
    add('## 10. 处理速度、表观滞后和端到端延迟')
    add(table(['组别','中位数/ms','P95/ms','三轮中位数/ms'],[
        [g,f(speed[g]['median_ms']),f(speed[g]['p95_ms']),' / '.join(f(x)for x in speed[g]['repeat_median_ms'])]for g in 'ABCDEF']))
    add('''实测采用同一96张预加载RGB-D图像，三轮随机排列组别顺序，每条路线先4次预热，再重新运行检测、模型和该组建轴。每组288次计时。包括相同本地进程通信边界；CPU约束相同，GPU为RTX 2070。原图读取、相机等待、GUI、ROS和机械臂执行不计入此表。

E/F的计算耗时约为HaMeR路线的43%；A—D差别主要是小量建轴开销，无法通过换建轴方法消除HaMeR网络自身耗时。

对30个转角超过15°且采样足够的旋转片段，另外计算了原始曲线的表观时间偏移：A—D中位数约0 ms，E约+5 ms，F约−10 ms。E有3段、F有7段到达±250 ms搜索边界，不能当作可靠滞后估计。典型图像采样间隔约67 ms，10 ms搜索步长不等于10 ms测量精度。

负的表观偏移不表示模型能预测未来；幅值失真、曲线形状及参考时钟误差都会影响相关峰。主误差指标没有按各组最佳偏移重新对齐。机器人端到端延迟本次没有测量，不能由表中的20 ms或47 ms替代。''')
    fig('06_实际处理耗时','直接重复计时，包含重新检测和当前组的姿态计算。')
    add('## 11. 一致性和时间对齐敏感性')
    add(table(['分录像MAE/°']+list('ABCDEF'),[
        ['录像一'if label=='earlier'else'录像二']+[f(np.mean([x['mae_deg']for x in r if x['session']==label and x['group']==g]))for g in 'ABCDEF']
        for label in ['earlier','later']]))
    add('F在录像一约7.28°，录像二约2.98°；排序会随录像改变，不能只选一段作为投稿证据。慢/中/快分层结果和每个动作的幅值比例、相关性均已保存；已有数据不构成独立受试者重复性实验。')
    add(table(['动作速度','片段数']+list('ABCDEF'),[
        [v,sum(x['speed']==v and x['group']=='A'for x in r)]+[
            f(np.mean([x['mae_deg']for x in r if x['speed']==v and x['group']==g]))for g in 'ABCDEF']
        for v in ['慢速','中速','快速','未标速度']]))
    add('速度来自原有动作文字标注，不是按真实角速度划分的受控速度档；此表不能据以推导严格的速度响应曲线。')
    add(table(['组别','参考偏移−40 ms','原偏移','参考偏移+40 ms'],[
        [g,f(m[g]['macro_mae_minus40ms_deg'],3),f(m[g]['macro_mae_deg'],3),f(m[g]['macro_mae_plus40ms_deg'],3)]for g in 'ABCDEF']))
    add('''表中为MAE，单位°。沿用原分析的全局时间偏移：录像一+20 ms，录像二约0 ms；其来源是历史腕环角速度与IMU的相关分析，不是独立硬件同步。主分析没有为任一组重新优化偏移，敏感性范围也不能当作已知的时钟误差界限。

改变时间偏移带来的指标变化，比B相对A的约0.006°差值大。结合片段胜负接近一半，本轮支持“没有观察到稳定腕环精度增益”，不支持精确到千分之一度的物理优劣声明。

去掉左右手类别不一致帧的附加统计也保留：A/B/C/D片段MAE分别约3.140/3.147/3.140/3.151°，四者仍然接近。主表没有采用这个删帧结果。''')
    add('## 12. D组区域及可复现细节')
    add('四区顶点在中性模型上事先固定，取区域中心而不是随每帧重新选“最稳的点”。区域允许掌厚两侧表面；本次不是仅选掌心一侧皮肤。候选范围、蒙皮权重阈值及目标中心见《实验约定》。')
    add(table(['区域','MANO顶点编号'],[[{'near':'近腕','distal':'远端掌部','thumb':'拇指侧','little':'小指侧'}[k],', '.join(map(str,v))]for k,v in geom['regions'].items()]))
    fig('07_固定网格区域','四个固定区域与五个解剖关键点的位置；区域大小和位置未按测得误差搜索。')
    add('''## 13. 这轮实验能支持什么研究结论

可以作为消融结果：同一HaMeR输出上替换四种建轴方式，在当前短动作数据中未得到稳定精度增益。可以明确展示平移误转与主动旋转衰减的矛盾，以及真实深度路径对数据质量的敏感性。

还不能写成：腕环显著提高姿态精度、系统长期保持3°零位精度、已经证明逐轴准确、位置达到某毫米绝对精度、机器人延迟为20 ms、结果可推广到不同人或遮挡环境。

下一项最有价值的补充，是把独立姿态参照刚性固定到手上，在一次固定零位下重复相同的平移和三轴旋转，同时保留完整相机时间流。它能直接区分视觉偏差、参考漂移、手别误判与真实伴随运动；当前六组脚本可以复用。''')
    add('## 14. 结果文件与复现')
    add(table(['文件','内容'],[
        ['逐动作六组指标.csv','79片段×6组：角误差、幅值、三轴活动、位置散布、跳变、表观滞后'],
        ['逐帧误差与三轴变化.csv','共同有效时刻的IMU角、模型角、误差、三轴分量和位置'],
        ['整段录像固定零位指标.csv / 整段录像固定零位逐帧.csv','整段只设一次零位的指标与逐帧结果'],
        ['失败和左右手误判明细.csv','每个失败/身份不一致的时间、组别和原因'],
        ['按录像动作速度汇总.csv','录像、动作类别、速度的分层结果'],
        ['腕环配对增益.csv','腕环相对A/C/D逐动作配对差值'],
        ['processing_timing.json','三轮计时顺序、每次耗时和汇总'],
        ['实验约定.md','实验变量、坐标定义、D区域选择、指标边界']]))
    add(f'''本地工程分支：`codex/six-group-unfiltered-20260917`。源基线提交：`ab621bc`。原始预测和建轴记录位于工程的 `.runtime/six_group_20260917`；原始录像未修改。本次未上传GitHub。

复现顺序（在工程目录执行）：

```bash
cd {ROOT}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/mediapipe_env/bin/python perception_mediapipe/six_group_experiment.py mp
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/six_group_experiment.py hamer
OPENBLAS_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/analyze_six_groups.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/benchmark_six_groups.py
/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/plot_six_groups.py
/home/dongtian/anaconda3/envs/hamer_rtx2060/bin/python perception_mediapipe/report_six_groups.py
```

这些命令会重写本实验的结果文件；需要保留多次运行时先备份本实验结果目录。它们不启动机械臂。

证据记录：使用实际录像重新推理；结果为离线工程对照，非预注册、多受试者验证。没有统计显著性或人群泛化声明。参考角、独立逐轴真值、位置真值和端到端延迟的边界已经分别列出。''')
    (P/'六组无滤波对比实验报告.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
