import cv2
import time
import tensorflow as tf
from glob import glob
from detection_test_pb import detection
from recognition_test_pb import recognition
from text_recognition.model import TextRecognition
import os
import numpy as np
from tqdm import tqdm
import argparse
import csv

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between reference (ground truth) and hypothesis (predicted) strings.

    Args:
    - reference (str): The correct reference text.
    - hypothesis (str): The hypothesis text to compare.

    Returns:
    - float: The CER expressed as a fraction of the total number of characters in the reference.
    """
    distance = levenshtein_distance(reference, hypothesis)
    cer = distance / max(len(reference), 1)
    return cer

def load_ground_truths(ground_truth_path):
    """
    Reads a ground truth file and concatenates the last element (assumed to be text) from each comma-separated line.
    
    Args:
        ground_truth_path (str): Path to the ground truth file.
    
    Returns:
        str: A continuous string of all last elements concatenated together.
    
    Note:
        This function assumes each line in the file is formatted as comma-separated values,
        with the text of interest being the last element in each line.
    """
    try:
        with open(ground_truth_path, 'r') as file:
            lines = file.readlines()
            # Extract the last comma-separated value from each line as the target text
            target_texts = [line.strip().split(',')[-1] for line in lines]
            continuous_text = ''.join(target_texts)
        return continuous_text
    except Exception as e:
        # Handle possible errors
        print(f"Error reading ground truth file {ground_truth_path}: {e}")
        return ""

def evaluate_model(img_path, ground_truth):

    detection_model_path = "pb_models/detection.pb"
    recognition_model_h_path = "ckpt/recognition_h/model_all.ckpt-8000"
    recognition_model_v_path = "ckpt/recognition_v/model_all.ckpt-146000"

    with tf.Graph().as_default():
        detection_graph_def = tf.GraphDef()
        with open(detection_model_path, "rb") as f:
            detection_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(detection_graph_def, name="")

        sess_d=tf.Session()
        init = tf.global_variables_initializer()
        sess_d.run(init)
        input_x = sess_d.graph.get_tensor_by_name("Placeholder:0")
        segm_logits = sess_d.graph.get_tensor_by_name("model/segm_logits/add:0")
        link_logits = sess_d.graph.get_tensor_by_name("model/link_logits/Reshape:0")

    bs = 4
    model = TextRecognition(is_training=False, num_classes=37)

    images_ph_h = tf.placeholder(tf.float32, [bs, 32, 240, 1])
    model_out_h = model(inputdata=images_ph_h)
    saver_h = tf.train.Saver()
    sess_r_h=tf.Session()
    saver_h.restore(sess=sess_r_h, save_path=recognition_model_h_path)
    decoded_h, _ = tf.nn.ctc_beam_search_decoder(model_out_h, 60 * np.ones(bs), merge_repeated=False)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        images_ph_v = tf.placeholder(tf.float32, [bs, 32, 320, 1])
        model_out_v = model(inputdata=images_ph_v)
        saver_v = tf.train.Saver()
        sess_r_v=tf.Session()
        saver_v.restore(sess=sess_r_v, save_path=recognition_model_v_path)
        decoded_v, _ = tf.nn.ctc_beam_search_decoder(model_out_v, 80 * np.ones(bs), merge_repeated=False)

    with open('evaluation_result.txt', 'w', newline='') as file:    
        writer = csv.writer(file)
        img_paths = glob(os.path.join(img_path, "*.jpg"))

        config = {}
        config['segm_conf_thr'] = 0.8
        config['link_conf_thr'] = 0.8
        config['min_area'] = 300
        config['min_height'] = 10

        total_cer = 0
        count = 0

        for impath in tqdm(img_paths):
            imname = os.path.splitext(os.path.basename(impath))[0]
            im = cv2.imread(impath)
            bboxs = detection(im, sess_d, input_x, segm_logits, link_logits, config)
            for bbox in bboxs:
                pts = [int(p) for p in bbox.split(",")]
                cv2.rectangle(im, (pts[0], pts[1]), (pts[4], pts[5]), (0, 255, 0), 2)

            try:
                predicted = recognition(im, sess_r_h, sess_r_v , bboxs, (240, 32), images_ph_h, images_ph_v, model_out_h, model_out_v, decoded_h, decoded_v)
                ground_truth_text = load_ground_truths(os.path.join(ground_truth, (imname + '.txt')))
                writer.writerow([imname, predicted, ground_truth_text])
                count += 1
            except ValueError as e:
                print(f"Error processing frame {imname}: {e}")
                writer.writerow([imname, e])
                continue

            cer = calculate_cer(ground_truth_text, predicted)
            total_cer += cer
        
        # Calculate the overall Character Error Rate (CER) in percentage
        overall_cer = (total_cer / max(count, 1)) * 100
        print(f"Character Error Rate: {overall_cer:.2f}%")

def main(img_path, ground_truth):
    # Using CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    evaluate_model(img_path, ground_truth)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Evaluate the OCR model')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input image directory.')
    parser.add_argument('-l', '--labels', type=str, required=True, help='Path to the ground truth directory.')

    args = parser.parse_args()
    main(args.input, args.labels)