import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import math


def _false_positive_weights(num_predictions: int) -> float:
    return 1.0 - 1.0 / num_predictions if num_predictions else 0.0

def continuous_segments(array: np.ndarray, value: int) -> np.ndarray:

    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    a = array
    if a.ndim != 1:
        a = a.ravel()

    eq = (a == value)
    w = eq.copy()
    w[1:] &= ~eq[:-1]
    starts = np.flatnonzero(w)

    w[:-1] = eq[:-1] & ~eq[1:]
    w[-1] = eq[-1]
    ends = np.flatnonzero(w) + 1

    out = np.empty((starts.size, 2), dtype=np.int64)
    out[:, 0] = starts
    out[:, 1] = ends

    return out

def larm_score(predictions: np.ndarray,targets: np.ndarray) -> float:
    """Compute the LARM score for discrete anomaly predictions.
    
    公式对应：
    LARM = (1/|I₁(g)|) * Σ_{A∈I₁(g):|I₁(p_A)|>0} [(α(p_A)+1)/2^|I₁(p_A)|] 
           - 2*Σ_{A∈I₀(g)} |I₁(p_A)| 
           - Σ_{A∈I₀(g)} β(|p_A⁻¹(1)|)
    
    符号说明：
    - I₁(g): 真实异常段集合（ground truth anomaly segments）
    - I₀(g): 真实正常段集合（ground truth normal segments）
    - I₁(p_A): 段A内的预测阳性子段数量
    - α(p_A): 段A内预测阳性的指数和（Σ2^-(i+1)，i为阳性位置索引）
    - β: 假阳性权重函数（复用_false_positive_weights）
    
    Args:
        predictions (npt.NDArray[np.integer]): 预测的二值异常数组
        targets (npt.NDArray[np.integer]): 真实的二值异常数组
    
    Returns:
        float: LARM score值
    """
    # 类型转换与二值化（与ALARM保持一致）
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    predictions_b = (predictions != 0)  # 预测阳性为True
    targets_b = (targets != 0)          # 真实阳性为True

    # 1. 分割真实段：I₁(g)（真实异常段）、I₀(g)（真实正常段）
    true_anomaly_segments = continuous_segments(targets_b, True)   # I₁(g)
    true_normal_segments = continuous_segments(targets_b, False)   # I₀(g)
    num_true_anomalies = len(true_anomaly_segments)                # |I₁(g)|

    # 2. 计算第一项：(1/|I₁(g)|) * Σ_{A∈I₁(g):|I₁(p_A)|>0} [(α(p_A)+1)/2^|I₁(p_A)|]
    term1 = 0.0
    if num_true_anomalies > 0:
        sum_term1 = 0.0
        for seg in true_anomaly_segments:
            start, end = seg
            # 该真实异常段内的预测结果
            seg_pred = predictions_b[start:end]
            if not np.any(seg_pred):
                continue  # |I₁(p_A)|=0，跳过
            
            # 计算|I₁(p_A)|：段内预测阳性的连续子段数量
            pred_segs_in_anomaly = continuous_segments(seg_pred, True)
            num_pred_segs = len(pred_segs_in_anomaly)  # |I₁(p_A)|
            
            # 计算α(p_A) = Σ2^-(i+1)（i为阳性位置在段内的索引）
            pred_pos_idx = np.nonzero(seg_pred)[0]
            alpha = np.exp2(-(pred_pos_idx + 1)).sum()  # α(p_A)
            
            # 累加(α+1)/2^|I₁(p_A)|
            sum_term1 += (alpha + 1) / (2 ** num_pred_segs)
        
        term1 = sum_term1 / num_true_anomalies  # 除以|I₁(g)|

    # 3. 计算第二项：2*Σ_{A∈I₀(g)} |I₁(p_A)|
    term2 = 0.0
    # 第三项：Σ_{A∈I₀(g)} β(|p_A⁻¹(1)|)
    term3 = 0.0
    for seg in true_normal_segments:
        start, end = seg
        # 该真实正常段内的预测结果
        seg_pred = predictions_b[start:end]
        # |I₁(p_A)|：真实正常段内预测阳性（误报）的连续子段数量，事件级误报
        pred_segs_in_normal = continuous_segments(seg_pred, True)
        # print(f'pred_segs_in_normal:{pred_segs_in_normal}')
        num_pred_segs_normal = len(pred_segs_in_normal)
        term2 += num_pred_segs_normal  # 累加|I₁(p_A)|
        
        # |p_A⁻¹(1)|：段内预测阳性的总数量，单点级误报
        num_pred_pos_normal = int(np.count_nonzero(seg_pred))
        # 计算β并累加
        beta = _false_positive_weights(num_pred_pos_normal)
        term3 += beta
    larm = term1 - 2 * term2 - term3
    print(f'term1:{term1:.2f},  term2:{term2:.2f},  term3:{term3:.2f}')
    false_alarm_tolerance_rate=0.2
    if term2>len(true_normal_segments)*false_alarm_tolerance_rate:
        score_event=1.0
    else:
        score_event=term2/(len(true_normal_segments)*false_alarm_tolerance_rate)

    score_beta=term3/len(true_normal_segments) 
    # 4. 计算最终LARM score
    
    larm1= 0.5*term1 - 0.5 *( score_event + score_beta)
    print(f'term1:{term1:.2f},  term2:{score_event:.2f},  term3:{score_beta:.2f}')
    print(f'larm:{larm:.2f},    larm1:{larm1:.2f}')
    return float(larm)

def VISA_score(predictions: np.ndarray, targets: np.ndarray, false_alarm_tolerance: int = 2) -> float:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    predictions_b = (predictions != 0)
    targets_b = (targets != 0)

    if int(np.count_nonzero(predictions_b)) == 0:
        return 0.0,0.0,0.0

    ground_truth_alarms = []
    predicted_alarms = []

    start = None
    for i, val in enumerate(targets_b):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ground_truth_alarms.append([start, i])
            start = None
    if start is not None:
        ground_truth_alarms.append([start, len(targets_b)])
    ground_truth_alarms = np.array(ground_truth_alarms)

    start = None
    for i, val in enumerate(predictions_b):
        if val and start is None:
            start = i
        elif not val and start is not None:
            predicted_alarms.append([start, i])
            start = None
    if start is not None:
        predicted_alarms.append([start, len(predictions_b)])
    predicted_alarms = np.array(predicted_alarms)

    first_predictions = predictions_b.copy()
    n_first_predictions = first_predictions.size
    target_positions = np.nonzero(targets_b)[0]

    if predicted_alarms.shape[0] and target_positions.size:
        predicted_alarms_start = predicted_alarms[:, 0]
        predicted_alarms_end = predicted_alarms[:, 1]
        first_target = np.searchsorted(target_positions, predicted_alarms_start, side='left')
        first_target_safe = np.minimum(first_target, target_positions.size - 1)
        target_candidates = target_positions[first_target_safe]
        prediction_overlap = (first_target < target_positions.size) & (target_candidates < predicted_alarms_end)

        if np.any(prediction_overlap):
            left = (target_candidates[prediction_overlap] + 1).astype(np.int64, copy=False)
            right = predicted_alarms_end[prediction_overlap]
            diff = np.zeros(n_first_predictions + 1, dtype=np.int32)
            np.add.at(diff, left, 1)
            np.add.at(diff, right, -1)
            first_predictions[np.cumsum(diff[:-1]) > 0] = False

    p = np.empty(first_predictions.size + 1, dtype=np.int32)
    p[0] = 0
    np.cumsum(first_predictions, out=p[1:])
    detected_mask = (p[ground_truth_alarms[:, 1]] - p[ground_truth_alarms[:, 0]]) != 0
    detected_anomalies = ground_truth_alarms[detected_mask]

    np.cumsum(targets_b, out=p[1:])#计算数组 targets_b 的前缀累加和，并将结果直接存入数组 p 的切片 p[1:] 中
    true_false_alarms = np.count_nonzero((p[predicted_alarms[:, 1]] - p[predicted_alarms[:, 0]]) == 0)
    num_fp = int(np.count_nonzero(predictions_b & ~targets_b))#统计预测为正但实际为负的位置（即假阳性）
    num_tp = int(np.count_nonzero(predictions_b & targets_b))#真阳性
    num_positives = int(np.count_nonzero(targets_b))
    print(f'num_fp:{num_fp},num_tp:{num_tp},num_positives:{num_positives}')

    fpw = float(_false_positive_weights(num_fp))

    #基础得分计算
    score_DA = float(len(detected_anomalies))                   #DA奖励分,1段真实异常内如果检测到多个异常，只算1次异常
    #------后续改进，如果延迟太久不计分-------------#####
    score_TA = float(true_false_alarms) / false_alarm_tolerance #TA惩罚分
    score3 = fpw                                                #β函数惩罚分
    # score3=num_tp/num_positives

    #计算公式中α
    score2=0.0
    for anomaly in detected_anomalies:
        start, end = anomaly
        anomaly_prediction = predictions_b[start:end]
        if anomaly_prediction.size == 0:
            continue
        num_alarms = int(anomaly_prediction[0]) + np.count_nonzero(anomaly_prediction[1:] & ~anomaly_prediction[:-1])   #该段内报警事件的个数 num_alarms=|I1(pA)|
        idx = np.nonzero(anomaly_prediction)[0]                         #获取报警位置（段内偏移）
        nom = np.ldexp(1 + np.exp2(-(idx + 1)).sum(), -num_alarms)      #α函数，nom=(1 + Σ 2^{-(idx+1)}) * 2^{-num_alarms}
        score2 += nom / len(detected_anomalies)

    #计算公式中EA
    false_alarms_early_overflow = 0
    score_EA=0.0
    if len(ground_truth_alarms) > 0:
        idx = ground_truth_alarms[:, 0]
        if idx[0] > 0 or len(idx) > 1:
            if idx[0] == 0:
                idx = idx[1:]
            false_alarms_early_overflow = np.count_nonzero(predictions_b[idx - 1] & predictions_b[idx])
    score_EA += 1.5 * false_alarms_early_overflow / false_alarm_tolerance

    #计算公式中LA
    false_alarms_late_overflow = 0
    score_LA=0.0
    if len(ground_truth_alarms) > 0:
        idx = ground_truth_alarms[:, 1]
        if idx[-1] < len(predictions_b) or len(idx) > 1:
            if idx[-1] == len(predictions_b):
                idx = idx[:-1]
            false_alarms_late_overflow = np.count_nonzero(predictions_b[idx - 1] & predictions_b[idx])
    score_LA += 1.5 * false_alarms_late_overflow / false_alarm_tolerance

    ALARM=score_DA+score2-score3-score_TA-score_EA-score_LA
    len_ground_truth_alarms=float(len(ground_truth_alarms))


    if score_TA > 2*len_ground_truth_alarms:
        score_DTELA=(score_DA-score_TA-score_EA-score_LA)/score_TA
    else:
        score_DTELA=(score_DA-score_TA-score_EA-score_LA)/len_ground_truth_alarms

    if score_DTELA<0:
        score_DTELA=0
        improve_score=0
        DQEi=0
    else:
        precision_pA=score2
        recall_FPA=num_tp/num_positives

        ppA_rFPA_f1=2*precision_pA*recall_FPA/(precision_pA+recall_FPA)#类似F1分数
        beta=0.5
        ppA_rFPA_fbeta=(1+beta*beta)*precision_pA*recall_FPA/(beta*beta*precision_pA+recall_FPA)#Fβ分数

        m=0.5
        VISA=m*score_DTELA+(1-m)*ppA_rFPA_f1
        VISAi=m*score_DTELA+(1-m)*ppA_rFPA_fbeta

        DQEi=math.sqrt((0.5*score_DTELA+0.5*precision_pA)*recall_FPA)

        print(f'score_DTELA:{score_DTELA:.2f},precision_pA:{precision_pA:.2f},recall_FPA:{recall_FPA:.2f},score3:{score3:.2f}')
        print(f'ppA_rFPA_f1:{ppA_rFPA_f1:.2f},beta:{beta},ppA_rFPA_fbeta:{ppA_rFPA_fbeta:.2f}')
        print(f'm:{m:.2f},VISA:{VISA:.2f},VISAi:{VISAi:.2f}')


    return float(ALARM),float(VISA),float(DQEi)

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)

    # F1pate=PATE(labels, score, e_buffer=60, d_buffer=60,binary_scores = False)

    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA= grader.metric_PointF1PA(labels, score, preds=pred)
    PointF1ibaPA=grader.metric_PointF1ibaPA(labels, score, preds=pred,adjust_window=10)
    PointF1fPA,latency_ave_fpa=grader.metric_PointF1fPA(labels, score, preds=pred)
    print(f'latency_ave_fpa:{latency_ave_fpa:.2f}')
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    if pred is not None:
        larm = larm_score(pred, labels)
        metrics['LARM'] = round(larm, 2)
    else:
        metrics['LARM'] = None

    if pred is not None:
        alarm_pa,latency_ave=grader._adjust_back_only_predicts(score, labels, pred=pred,calc_latency=True)#-------merge_len设置
        print(f'latency_ave:{latency_ave:.2f}')
        alarm,_,_ = alarm_score(pred, labels,false_alarm_tolerance=2)
        _,alarm_f1,DQEi = alarm_score(alarm_pa, labels,false_alarm_tolerance=2)
        metrics['ALARM'] = round(alarm, 2)
        metrics['ALARM_F1'] = round(alarm_f1, 2)
        metrics['DQEi'] = round(DQEi, 2)
    else:
        metrics['ALARM'] = None
        metrics['ALARM_F1'] = None
        metrics['DQEi'] = None

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC
    # metrics['F1pate'] = F1pate

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['ibaPA-F1'] = PointF1ibaPA
    metrics['fPA-F1'] = PointF1fPA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    # 保留4位小数
    for key in metrics:
        if isinstance(metrics[key], (int, float, np.number)):
            metrics[key] = round(float(metrics[key]), 2)

    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics
