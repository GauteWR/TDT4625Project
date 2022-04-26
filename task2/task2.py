import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    px1, py1, px2, py2 = prediction_box
    tx1, ty1, tx2, ty2 = gt_box

    if px1 > tx2 or ty1 > py2 or py1 > ty2 or tx1 > px2:
        return 0.0

    x1 = max(px1, tx1)
    x2 = min(px2, tx2)
    y1 = max(py1, ty1)
    y2 = min(py2, ty2)

    intersection = ( x2 - x1 ) * ( y2 - y1 )
    pred_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1] )
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1] )
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    check = num_fp + num_tp
    if check == 1:
        return 1
    
    if num_tp == 0 and num_fp == 0:
        return 1
    
    precision = float(num_tp / (num_tp + num_fp))
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    check = num_tp + num_fn 
    if check == 0:
        return 0

    if num_tp == 0 and num_fn == 0:
        return 0

    recall = float(num_tp / (num_tp + num_fn))
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    matched_pred_boxes = []
    matched_gt_boxes = []
    # Find all possible matches with a IoU >= iou threshold
    for gt in gt_boxes:
        # box, index in prediction_boxes, iou with gt
        best_match = [None, None, 0]
        for i, pred in enumerate(prediction_boxes):
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
               if iou > best_match[2]:
                   best_match = [pred, i, iou]

        # Add pair to matched boxes. Also remove best match, since 1 prediction box can match only 1 gt box. 
        if(best_match[0] is not None):
            matched_gt_boxes.append(gt)
            matched_pred_boxes.append(best_match[0])
            np.delete(prediction_boxes, best_match[1])

    return np.array(matched_pred_boxes), np.array(matched_gt_boxes)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    tp = 0
    fp = 0
    fn = 0

    matched_gt = np.zeros(len(gt_boxes))
    for pred in prediction_boxes:
        found_match = False
        for i, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
                tp += 1
                matched_gt[i] = 1
                found_match = True

        if not found_match:
            fp += 1

    for val in matched_gt:
        if val == 0:
            fn += 1

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(all_prediction_boxes)):   
        res_dict = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += res_dict["true_pos"]
        fp += res_dict["false_pos"]
        fn += res_dict["false_neg"]

    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return precision, recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    print("We calling on the curve")
    confidence_thresholds = np.linspace(0, 1, 500)
    #print(confidence_thresholds)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    for i in confidence_thresholds:
        # print("Confidence curve index", i)
        temp = []
        for j in range(len(all_prediction_boxes)):
            boxes = all_prediction_boxes[j]
            scores = confidence_scores[j]
            prune = []
            for k in range(len(scores)):
                if scores[k] >= i:
                    prune.append(True)
                else:
                    prune.append(False)
            temp.append(boxes[prune])
        precision, recall = calculate_precision_recall_all_images(temp, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    print("Plotting")
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    print("Mean avg")
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    temp_precisions = []
    for i in recall_levels:
        precision = precisions[recalls >= i]
        if precision.size == 0:
            precision = 0
        else:
            precision = max(precision)
        temp_precisions.append(precision)
    ap = np.mean(temp_precisions)
    return ap


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        print("Inside image", image_id)
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    print("Done with images")
    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    print("Values found")
    plot_precision_recall_curve(precisions, recalls)
    print("Plotting done")
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    print("Load ground")
    predicted_boxes = read_predicted_boxes()
    print("Load pred")
    mean_average_precision(ground_truth_boxes, predicted_boxes)
    print("Calculated")
